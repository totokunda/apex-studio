#include <napi.h>

#include <framework/mlt.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

enum class CommandType : uint32_t {
  kUpsertClip = 1,
  kRemoveClip = 2,
  kSetPlayState = 3,
  kSetPlayhead = 4,
  kSetHoleRect = 5,
  kSetViewport = 6,
  kResetGraph = 7,
  kShutdown = 8,
};

enum class ClipKind : uint32_t {
  kUnknown = 0,
  kVideo = 1,
  kImage = 2,
  kModel = 3,
  kShape = 4,
  kText = 5,
  kDrawing = 6,
  kAudio = 7,
};

struct ClipRecord {
  std::string clip_id;
  uint32_t clip_kind = 0;
  std::string media_path;
  int64_t start_frame = 0;
  int64_t end_frame = 0;
  int64_t trim_start = 0;
  int64_t trim_end = 0;
  double speed = 1.0;
  bool visible = true;
  int32_t z_index = 0;
};

struct EngineState {
  int id = 0;
  int width = 0;
  int height = 0;
  double fps = 24.0;
  uintptr_t surface_handle = 0;
  bool has_surface = false;

  std::string last_command = "init";
  uint64_t last_sequence = 0;

  bool is_playing = false;
  int64_t focus_frame = 0;
  bool playhead_accurate_seek = false;

  bool hole_visible = false;
  int hole_x = 0;
  int hole_y = 0;
  int hole_width = 0;
  int hole_height = 0;

  std::unordered_map<std::string, ClipRecord> clips;
  std::string active_clip_id;

  mlt_profile profile = nullptr;
  mlt_producer producer = nullptr;
  mlt_consumer consumer = nullptr;

  uint64_t total_decode_requests_enqueued = 0;
};

std::mutex g_mutex;
std::unordered_map<int, EngineState> g_engines;
std::atomic<int> g_next_id{1};

std::atomic<bool> g_mlt_initialized{false};

void EnsureMltFactoryInitialized() {
  bool expected = false;
  if (g_mlt_initialized.compare_exchange_strong(expected, true)) {
    mlt_factory_init(nullptr);
  }
}

double GetNumberOr(const Napi::Object& obj, const char* key, double fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsNumber()) return fallback;
  return value.As<Napi::Number>().DoubleValue();
}

int64_t GetInt64Or(const Napi::Object& obj, const char* key, int64_t fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsNumber()) return fallback;
  return static_cast<int64_t>(value.As<Napi::Number>().Int64Value());
}

int32_t GetInt32Or(const Napi::Object& obj, const char* key, int32_t fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsNumber()) return fallback;
  return value.As<Napi::Number>().Int32Value();
}

uint32_t GetUInt32Or(const Napi::Object& obj,
                     const char* key,
                     uint32_t fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsNumber()) return fallback;
  return value.As<Napi::Number>().Uint32Value();
}

bool GetBoolOr(const Napi::Object& obj, const char* key, bool fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsBoolean()) return fallback;
  return value.As<Napi::Boolean>().Value();
}

std::string GetStringOr(const Napi::Object& obj,
                        const char* key,
                        std::string fallback = {}) {
  if (!obj.Has(key)) return fallback;
  Napi::Value value = obj.Get(key);
  if (!value.IsString()) return fallback;
  return value.As<Napi::String>().Utf8Value();
}

Napi::Object GetObjectOr(const Napi::Object& obj, const char* key, Napi::Env env) {
  if (!obj.Has(key)) return Napi::Object::New(env);
  Napi::Value value = obj.Get(key);
  if (!value.IsObject()) return Napi::Object::New(env);
  return value.As<Napi::Object>();
}

std::string CommandTypeToLabel(CommandType type) {
  switch (type) {
    case CommandType::kUpsertClip:
      return "UpsertClip";
    case CommandType::kRemoveClip:
      return "RemoveClip";
    case CommandType::kSetPlayState:
      return "SetPlayState";
    case CommandType::kSetPlayhead:
      return "SetPlayhead";
    case CommandType::kSetHoleRect:
      return "SetHoleRect";
    case CommandType::kSetViewport:
      return "SetViewport";
    case CommandType::kResetGraph:
      return "ResetGraph";
    case CommandType::kShutdown:
      return "Shutdown";
    default:
      return "Unknown";
  }
}

bool IsPlayableClipKind(uint32_t clip_kind) {
  const auto kind = static_cast<ClipKind>(clip_kind);
  return kind == ClipKind::kVideo || kind == ClipKind::kAudio;
}

bool IsClipActiveAtFrame(const ClipRecord& clip, int64_t focus_frame) {
  if (!clip.visible) return false;
  if (clip.end_frame <= clip.start_frame) return false;
  return focus_frame >= clip.start_frame && focus_frame < clip.end_frame;
}

int64_t ComputeSourceFrameForClip(const ClipRecord& clip, int64_t focus_frame) {
  if (clip.end_frame <= clip.start_frame) {
    return std::max<int64_t>(0, clip.trim_start);
  }
  const int64_t timeline_offset = std::max<int64_t>(0, focus_frame - clip.start_frame);
  const double safe_speed = clip.speed > 0.0 ? clip.speed : 1.0;
  const int64_t source_offset =
      static_cast<int64_t>(std::floor(static_cast<double>(timeline_offset) * safe_speed));
  int64_t source_frame = clip.trim_start + source_offset;
  if (clip.trim_end > clip.trim_start) {
    source_frame = std::min<int64_t>(source_frame, clip.trim_end - 1);
  }
  return std::max<int64_t>(0, source_frame);
}

ClipRecord ParseUpsertClipPayload(const Napi::Object& payload, Napi::Env env) {
  ClipRecord clip{};
  clip.clip_id = GetStringOr(payload, "clip_id");
  clip.clip_kind = GetUInt32Or(payload, "clip_kind", 0);
  clip.media_path = GetStringOr(payload, "media_path");
  clip.z_index = GetInt32Or(payload, "z_index", 0);

  const Napi::Object timeline = GetObjectOr(payload, "timeline", env);
  clip.start_frame = GetInt64Or(timeline, "start_frame", 0);
  clip.end_frame = GetInt64Or(timeline, "end_frame", 0);
  clip.trim_start = GetInt64Or(timeline, "trim_start", 0);
  clip.trim_end = GetInt64Or(timeline, "trim_end", 0);
  clip.speed = GetNumberOr(timeline, "speed", 1.0);

  const Napi::Object transform = GetObjectOr(payload, "transform", env);
  clip.visible = GetBoolOr(transform, "visible", true);

  return clip;
}

void DestroyPipeline(EngineState& st) {
  if (st.consumer) {
    mlt_consumer_stop(st.consumer);
    mlt_consumer_close(st.consumer);
    st.consumer = nullptr;
  }
  if (st.producer) {
    mlt_producer_close(st.producer);
    st.producer = nullptr;
  }
  st.active_clip_id.clear();
}

const ClipRecord* SelectClipForFocus(const EngineState& st) {
  const ClipRecord* selected = nullptr;

  for (const auto& [_, clip] : st.clips) {
    if (!IsPlayableClipKind(clip.clip_kind)) continue;
    if (clip.media_path.empty()) continue;
    if (!IsClipActiveAtFrame(clip, st.focus_frame)) continue;
    if (!selected || clip.z_index > selected->z_index) {
      selected = &clip;
    }
  }

  if (selected) return selected;

  for (const auto& [_, clip] : st.clips) {
    if (!IsPlayableClipKind(clip.clip_kind)) continue;
    if (clip.media_path.empty()) continue;
    if (!clip.visible) continue;
    if (!selected || clip.z_index > selected->z_index) {
      selected = &clip;
    }
  }

  return selected;
}

mlt_producer CreateProducerForPath(mlt_profile profile, const std::string& path) {
  if (path.empty()) return nullptr;

  mlt_producer producer = mlt_factory_producer(profile, nullptr, path.c_str());
  if (producer) return producer;

  producer = mlt_factory_producer(profile, "avformat-novalidate", path.c_str());
  if (producer) return producer;

  producer = mlt_factory_producer(profile, "avformat", path.c_str());
  return producer;
}

mlt_consumer CreatePreviewConsumer(mlt_profile profile) {
  mlt_consumer consumer = mlt_factory_consumer(profile, "sdl2", nullptr);
  if (!consumer) {
    consumer = mlt_factory_consumer(profile, "sdl", nullptr);
  }
  if (!consumer) {
    consumer = mlt_factory_consumer(profile, "multi", nullptr);
  }
  return consumer;
}

void SeekActiveProducer(EngineState& st, const ClipRecord& clip) {
  if (!st.producer) return;
  const int64_t source_frame = ComputeSourceFrameForClip(clip, st.focus_frame);
  mlt_producer_seek(st.producer, source_frame);
}

void ApplyPlaybackState(EngineState& st, const ClipRecord& clip) {
  if (!st.producer || !st.consumer) return;

  const double safe_speed = clip.speed > 0.0 ? clip.speed : 1.0;
  const double next_speed = st.is_playing ? safe_speed : 0.0;
  mlt_producer_set_speed(st.producer, next_speed);

  mlt_properties consumer_props = MLT_CONSUMER_PROPERTIES(st.consumer);
  mlt_properties_set_int(consumer_props, "pause", st.is_playing ? 0 : 1);

  // Start the consumer even when paused so the current frame can be presented.
  if (mlt_consumer_is_stopped(st.consumer)) {
    mlt_consumer_start(st.consumer);
  }
}

bool EnsurePipelineForCurrentFocus(EngineState& st) {
  const ClipRecord* selected_clip = SelectClipForFocus(st);
  if (!selected_clip) {
    DestroyPipeline(st);
    return false;
  }

  if (!st.profile) {
    st.profile = mlt_profile_init(nullptr);
    if (st.profile) {
      if (st.width > 0) st.profile->width = st.width;
      if (st.height > 0) st.profile->height = st.height;
      if (st.fps > 0.0) {
        st.profile->frame_rate_num =
            std::max(1, static_cast<int>(std::llround(st.fps * 1000.0)));
        st.profile->frame_rate_den = 1000;
      }
    }
  }

  const bool need_rebuild =
      st.active_clip_id != selected_clip->clip_id || !st.producer || !st.consumer;

  if (need_rebuild) {
    DestroyPipeline(st);

    st.producer = CreateProducerForPath(st.profile, selected_clip->media_path);
    if (!st.producer) {
      st.active_clip_id.clear();
      return false;
    }

    if (selected_clip->trim_end > selected_clip->trim_start) {
      mlt_producer_set_in_and_out(st.producer,
                                  selected_clip->trim_start,
                                  selected_clip->trim_end - 1);
    }

    st.consumer = CreatePreviewConsumer(st.profile);
    if (!st.consumer) {
      mlt_producer_close(st.producer);
      st.producer = nullptr;
      st.active_clip_id.clear();
      return false;
    }

    mlt_properties consumer_props = MLT_CONSUMER_PROPERTIES(st.consumer);
    mlt_properties_set_int(consumer_props, "real_time", 1);
    mlt_properties_set_int(consumer_props, "rescale", 1);
#if !defined(__APPLE__)
    if (st.has_surface && st.surface_handle != 0) {
      mlt_properties_set_int64(
          consumer_props, "window_id", static_cast<int64_t>(st.surface_handle));
    }
#endif

    if (mlt_consumer_connect(st.consumer, MLT_PRODUCER_SERVICE(st.producer)) != 0) {
      mlt_consumer_close(st.consumer);
      st.consumer = nullptr;
      mlt_producer_close(st.producer);
      st.producer = nullptr;
      st.active_clip_id.clear();
      return false;
    }

    st.active_clip_id = selected_clip->clip_id;
  }

  SeekActiveProducer(st, *selected_clip);
  ApplyPlaybackState(st, *selected_clip);
  return true;
}

void ClearEngineState(EngineState& st) {
  st.clips.clear();
  st.is_playing = false;
  st.focus_frame = 0;
  st.playhead_accurate_seek = false;
  st.hole_visible = false;
  st.hole_x = 0;
  st.hole_y = 0;
  st.hole_width = 0;
  st.hole_height = 0;
  DestroyPipeline(st);
}

Napi::Value CreateEngine(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  EnsureMltFactoryInitialized();

  int width = 0;
  int height = 0;
  double fps = 24.0;

  if (info.Length() > 0 && info[0].IsObject()) {
    Napi::Object config = info[0].As<Napi::Object>();
    width = GetInt32Or(config, "width", 0);
    height = GetInt32Or(config, "height", 0);
    fps = GetNumberOr(config, "fps", 24.0);
  }

  const int id = g_next_id.fetch_add(1);

  std::lock_guard<std::mutex> lock(g_mutex);
  auto [it, _inserted] = g_engines.try_emplace(id);
  EngineState& st = it->second;
  st.id = id;
  st.width = width;
  st.height = height;
  st.fps = fps > 0.0 ? fps : 24.0;
  st.last_command = "init";

  return Napi::Number::New(env, id);
}

Napi::Value DestroyEngine(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected (engineId:number)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const int id = info[0].As<Napi::Number>().Int32Value();

  bool removed = false;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_engines.find(id);
    if (it != g_engines.end()) {
      DestroyPipeline(it->second);
      if (it->second.profile) {
        mlt_profile_close(it->second.profile);
        it->second.profile = nullptr;
      }
      g_engines.erase(it);
      removed = true;
    }
  }

  return Napi::Boolean::New(env, removed);
}

Napi::Value AttachSurface(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsBuffer()) {
    Napi::TypeError::New(env, "Expected (engineId:number, nativeHandle:Buffer)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const int id = info[0].As<Napi::Number>().Int32Value();
  auto handle = info[1].As<Napi::Buffer<char>>();
  if (handle.Length() < sizeof(void*)) {
    Napi::Error::New(env, "nativeHandle buffer too small")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  uintptr_t raw = reinterpret_cast<uintptr_t>(
      *reinterpret_cast<void**>(handle.Data()));

  bool ok = false;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_engines.find(id);
    if (it != g_engines.end()) {
      it->second.surface_handle = raw;
      it->second.has_surface = true;
      it->second.last_command = "attachSurface";
      ok = true;
    }
  }

  return Napi::Boolean::New(env, ok);
}

Napi::Value SubmitCommand(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsObject()) {
    Napi::TypeError::New(env, "Expected (engineId:number, command:object)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const int id = info[0].As<Napi::Number>().Int32Value();
  Napi::Object command = info[1].As<Napi::Object>();

  const uint32_t type_u32 = GetUInt32Or(command, "type", 0);
  const auto command_type = static_cast<CommandType>(type_u32);
  const uint64_t sequence =
      static_cast<uint64_t>(GetInt64Or(command, "sequence", 0));
  const Napi::Object payload = GetObjectOr(command, "payload", env);

  ClipRecord upsert_clip{};
  std::string remove_clip_id;
  bool has_upsert_clip = false;
  bool has_remove_clip = false;

  if (command_type == CommandType::kUpsertClip) {
    upsert_clip = ParseUpsertClipPayload(payload, env);
    has_upsert_clip = !upsert_clip.clip_id.empty();
  } else if (command_type == CommandType::kRemoveClip) {
    remove_clip_id = GetStringOr(payload, "clip_id");
    has_remove_clip = !remove_clip_id.empty();
  }

  bool ok = false;

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_engines.find(id);
    if (it == g_engines.end()) {
      return Napi::Boolean::New(env, false);
    }

    EngineState& st = it->second;
    st.last_command = CommandTypeToLabel(command_type);
    st.last_sequence = sequence;

    switch (command_type) {
      case CommandType::kUpsertClip:
        if (!has_upsert_clip) {
          ok = false;
          break;
        }
        st.clips[upsert_clip.clip_id] = std::move(upsert_clip);
        st.total_decode_requests_enqueued += 1;
        EnsurePipelineForCurrentFocus(st);
        ok = true;
        break;

      case CommandType::kRemoveClip:
        if (!has_remove_clip) {
          ok = false;
          break;
        }
        st.clips.erase(remove_clip_id);
        st.total_decode_requests_enqueued += 1;
        EnsurePipelineForCurrentFocus(st);
        ok = true;
        break;

      case CommandType::kSetPlayState: {
        st.is_playing = GetBoolOr(payload, "is_playing", st.is_playing);
        EnsurePipelineForCurrentFocus(st);
        ok = true;
        break;
      }

      case CommandType::kSetPlayhead:
        st.focus_frame = GetInt64Or(payload, "focus_frame", st.focus_frame);
        st.fps = GetNumberOr(payload, "fps", st.fps);
        st.playhead_accurate_seek =
            GetBoolOr(payload, "accurate_seek", st.playhead_accurate_seek);
        EnsurePipelineForCurrentFocus(st);
        ok = true;
        break;

      case CommandType::kSetHoleRect: {
        const Napi::Object rect = GetObjectOr(payload, "rect", env);
        st.hole_x = GetInt32Or(rect, "x", st.hole_x);
        st.hole_y = GetInt32Or(rect, "y", st.hole_y);
        st.hole_width = GetInt32Or(rect, "width", st.hole_width);
        st.hole_height = GetInt32Or(rect, "height", st.hole_height);
        st.hole_visible = GetBoolOr(payload, "visible", st.hole_visible);
        ok = true;
        break;
      }

      case CommandType::kSetViewport:
        st.width = GetInt32Or(payload, "width", st.width);
        st.height = GetInt32Or(payload, "height", st.height);
        if (st.profile) {
          if (st.width > 0) st.profile->width = st.width;
          if (st.height > 0) st.profile->height = st.height;
        }
        ok = true;
        break;

      case CommandType::kResetGraph:
        ClearEngineState(st);
        ok = true;
        break;

      case CommandType::kShutdown:
        ClearEngineState(st);
        ok = true;
        break;

      default:
        ok = true;
        break;
    }
  }

  return Napi::Boolean::New(env, ok);
}

Napi::Value DrainDecodedVideoFrames(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected (engineId:number)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return Napi::Array::New(env, 0);
}

Napi::Value DrainDecodedAudioChunks(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected (engineId:number)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return Napi::Array::New(env, 0);
}

Napi::Value DrainMixedAudioBlocks(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected (engineId:number)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return Napi::Array::New(env, 0);
}

Napi::Value GetStats(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expected (engineId:number)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const int id = info[0].As<Napi::Number>().Int32Value();
  Napi::Object out = Napi::Object::New(env);

  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_engines.find(id);
  if (it == g_engines.end()) {
    out.Set("exists", Napi::Boolean::New(env, false));
    return out;
  }

  const EngineState& st = it->second;
  out.Set("exists", Napi::Boolean::New(env, true));
  out.Set("id", Napi::Number::New(env, st.id));
  out.Set("width", Napi::Number::New(env, st.width));
  out.Set("height", Napi::Number::New(env, st.height));
  out.Set("fps", Napi::Number::New(env, st.fps));
  out.Set("hasSurface", Napi::Boolean::New(env, st.has_surface));
  out.Set("lastCommand", Napi::String::New(env, st.last_command));
  out.Set("lastSequence", Napi::Number::New(env, static_cast<double>(st.last_sequence)));
  out.Set("isPlaying", Napi::Boolean::New(env, st.is_playing));
  out.Set("focusFrame", Napi::Number::New(env, static_cast<double>(st.focus_frame)));
  out.Set("holeVisible", Napi::Boolean::New(env, st.hole_visible));

  out.Set("decodeSchedulerTicks", Napi::Number::New(env, 0));
  out.Set("decodeSchedulerLastClockFrame", Napi::Number::New(env, static_cast<double>(st.focus_frame)));
  out.Set("clipCount", Napi::Number::New(env, static_cast<double>(st.clips.size())));
  out.Set("activeClipCount", Napi::Number::New(env, st.active_clip_id.empty() ? 0 : 1));
  out.Set("pendingDecodeCount", Napi::Number::New(env, 0));
  out.Set("totalDecodeRequestsEnqueued",
          Napi::Number::New(env, static_cast<double>(st.total_decode_requests_enqueued)));

  out.Set("videoExecutorSubmittedRequests", Napi::Number::New(env, 0));
  out.Set("videoExecutorProcessedRequests", Napi::Number::New(env, 0));
  out.Set("videoExecutorSucceededRequests", Napi::Number::New(env, 0));
  out.Set("videoExecutorFailedRequests", Napi::Number::New(env, 0));
  out.Set("videoExecutorDroppedRequests", Napi::Number::New(env, 0));
  out.Set("videoExecutorQueueDepth", Napi::Number::New(env, 0));

  out.Set("audioExecutorSubmittedRequests", Napi::Number::New(env, 0));
  out.Set("audioExecutorProcessedRequests", Napi::Number::New(env, 0));
  out.Set("audioExecutorSucceededRequests", Napi::Number::New(env, 0));
  out.Set("audioExecutorFailedRequests", Napi::Number::New(env, 0));
  out.Set("audioExecutorDroppedRequests", Napi::Number::New(env, 0));
  out.Set("audioExecutorQueueDepth", Napi::Number::New(env, 0));

  out.Set("audioMixerSubmittedChunks", Napi::Number::New(env, 0));
  out.Set("audioMixerMixedBlocksEnqueued", Napi::Number::New(env, 0));
  out.Set("audioMixerMixedBlocksDrained", Napi::Number::New(env, 0));
  out.Set("audioMixerDroppedChunks", Napi::Number::New(env, 0));
  out.Set("audioMixerInputQueueDepth", Napi::Number::New(env, 0));

  out.Set("audioSinkStarted", Napi::Boolean::New(env, st.consumer != nullptr));
  out.Set("audioSinkPlaying", Napi::Boolean::New(env, st.is_playing));
  out.Set("audioSinkSampleRate", Napi::Number::New(env, 48000));
  out.Set("audioSinkChannels", Napi::Number::New(env, 2));
  out.Set("audioSinkQueuedSamples", Napi::Number::New(env, 0));
  out.Set("audioSinkSubmittedBlocks", Napi::Number::New(env, 0));
  out.Set("audioSinkSubmittedSamples", Napi::Number::New(env, 0));
  out.Set("audioSinkConsumedSamples", Napi::Number::New(env, 0));
  out.Set("audioSinkDroppedSamples", Napi::Number::New(env, 0));

  const double audio_clock_seconds =
      st.producer && st.fps > 0.0
          ? static_cast<double>(mlt_producer_position(st.producer)) / st.fps
          : 0.0;
  out.Set("audioClockSeconds", Napi::Number::New(env, audio_clock_seconds));

  out.Set("videoPresenterQueuedFrames", Napi::Number::New(env, 0));
  out.Set("videoPresenterPresentedFrames", Napi::Number::New(env, 0));
  out.Set("videoPresenterDroppedFrames", Napi::Number::New(env, 0));
  out.Set("videoPresenterHasPresentedFrame", Napi::Boolean::New(env, false));
  out.Set("videoPresenterLastFocusFrame", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastSourceFrame", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastPts", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastWidth", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastHeight", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastPixelFormat", Napi::Number::New(env, 0));
  out.Set("videoPresenterLastZIndex", Napi::Number::New(env, 0));
  out.Set("videoPresenterTargetFocusFrame",
          Napi::Number::New(env, static_cast<double>(st.focus_frame)));
  out.Set("videoPresenterAudioClockSeconds", Napi::Number::New(env, audio_clock_seconds));

  out.Set("holeRendererAttached", Napi::Boolean::New(env, st.has_surface));
  out.Set("holeRendererVisible", Napi::Boolean::New(env, st.hole_visible));
  out.Set("holeRendererPresentedFrames", Napi::Number::New(env, 0));
  out.Set("holeRendererFailedFrames", Napi::Number::New(env, 0));
  out.Set("holeRendererWidth", Napi::Number::New(env, st.hole_width));
  out.Set("holeRendererHeight", Napi::Number::New(env, st.hole_height));

  out.Set("decodedVideoQueueDepth", Napi::Number::New(env, 0));
  out.Set("decodedVideoFramesEnqueued", Napi::Number::New(env, 0));
  out.Set("decodedVideoFramesDropped", Napi::Number::New(env, 0));
  out.Set("decodedVideoFramesDrained", Napi::Number::New(env, 0));
  out.Set("decodedAudioQueueDepth", Napi::Number::New(env, 0));
  out.Set("decodedAudioChunksEnqueued", Napi::Number::New(env, 0));
  out.Set("decodedAudioChunksDropped", Napi::Number::New(env, 0));
  out.Set("decodedAudioChunksDrained", Napi::Number::New(env, 0));

  out.Set("demuxWorkerSubmittedRequests", Napi::Number::New(env, 0));
  out.Set("demuxWorkerProcessedRequests", Napi::Number::New(env, 0));
  out.Set("demuxWorkerSucceededRequests", Napi::Number::New(env, 0));
  out.Set("demuxWorkerFailedRequests", Napi::Number::New(env, 0));
  out.Set("demuxWorkerDroppedRequests", Napi::Number::New(env, 0));
  out.Set("demuxWorkerQueueDepth", Napi::Number::New(env, 0));

  return out;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("createEngine", Napi::Function::New(env, CreateEngine));
  exports.Set("destroyEngine", Napi::Function::New(env, DestroyEngine));
  exports.Set("attachSurface", Napi::Function::New(env, AttachSurface));
  exports.Set("submitCommand", Napi::Function::New(env, SubmitCommand));
  exports.Set("drainDecodedVideoFrames", Napi::Function::New(env, DrainDecodedVideoFrames));
  exports.Set("drainDecodedAudioChunks", Napi::Function::New(env, DrainDecodedAudioChunks));
  exports.Set("drainMixedAudioBlocks", Napi::Function::New(env, DrainMixedAudioBlocks));
  exports.Set("getStats", Napi::Function::New(env, GetStats));
  return exports;
}

}  // namespace

NODE_API_MODULE(media_native_addon, Init)

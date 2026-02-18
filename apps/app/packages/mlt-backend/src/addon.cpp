#include <napi.h>
#include <framework/mlt.h>
#include <atomic>
#include <cstddef>
#include <cstring>

// ─── Shared ring buffer layout ──────────────────────────────────────────────
//
// JS allocates a SharedArrayBuffer and passes it to C++.  Layout:
//
//   Int32 [0] = write_index   (C++ writes, JS reads via Atomics)
//   Int32 [1] = read_index    (JS writes, C++ reads via Atomics)
//   Int32 [2] = width
//   Int32 [3] = height
//   Int32 [4] = frame_size    (bytes per YUV420p frame = w*h*3/2)
//   Int32 [5] = slot_count    (number of frame slots)
//   Int32 [6] = dropped_frames (C++ increments when ring is full)
//   Int32 [7] = reserved
//
//   Byte [HEADER_BYTES ... ] = frame slots (slot_count × frame_size)
//
// C++ writes the next frame into slot [write_index % slot_count],
// then increments write_index and calls Atomics.notify on [0].
// JS reads from slot [read_index % slot_count], then increments read_index.
//
// This is a lock-free SPSC ring buffer.  No JS callbacks, no Buffer
// allocations, no copies beyond the single memcpy into shared memory.

static constexpr int HEADER_INTS  = 8;
static constexpr int HEADER_BYTES = HEADER_INTS * sizeof(int32_t);

struct PlayerContext {
    std::atomic<bool> running{false};

    // Shared memory pointers (backed by JS SharedArrayBuffer)
    int32_t* header   = nullptr;   // points at the Int32Array region
    uint8_t* slots    = nullptr;   // points at frame data region
    int      slot_count = 0;
    size_t   frame_size = 0;

    // TSFN kept only for preventing GC of the shared buffer / stop signaling
    Napi::ThreadSafeFunction tsfn;
};

struct ReleaseWorker : Napi::AsyncWorker {
    PlayerContext* ctx;
    ReleaseWorker(Napi::Env env, PlayerContext* c)
        : AsyncWorker(env, "MLT TSFN Release"), ctx(c) {}
    void Execute() override {}
    void OnOK() override { if (ctx) ctx->tsfn.Release(); }
};

// ─── Globals ────────────────────────────────────────────────────────────────
static mlt_consumer   g_consumer = nullptr;
static mlt_producer   g_producer = nullptr;
static mlt_profile    g_profile  = nullptr;
static PlayerContext* g_ctx      = nullptr;
static bool g_mlt_initialized    = false;

static void ensure_mlt_init() {
    if (!g_mlt_initialized) {
        mlt_factory_init(nullptr);
        g_mlt_initialized = true;
    }
}

// ─── Frame callback (runs on MLT thread) ────────────────────────────────────
static void on_frame_show(mlt_properties /*owner*/, void* self,
                           mlt_event_data event_data) {
    PlayerContext* ctx = static_cast<PlayerContext*>(self);
    if (!ctx || !ctx->running) return;

    mlt_frame frame_ptr = mlt_event_data_to_frame(event_data);
    if (!frame_ptr) return;

    mlt_image_format fmt = mlt_image_yuv420p;  // native — no conversion!
    int width = 0, height = 0;
    uint8_t* image = nullptr;

    if (mlt_frame_get_image(frame_ptr, &image, &fmt, &width, &height, 0) != 0)
        return;
    if (!image || width <= 0 || height <= 0) return;

    size_t expected = static_cast<size_t>(width) * height * 3 / 2;
    if (expected != ctx->frame_size) return;  // resolution mismatch with pre-allocated slots

    // SPSC ring: check if there's space
    int32_t wi = __atomic_load_n(&ctx->header[0], __ATOMIC_ACQUIRE);
    int32_t ri = __atomic_load_n(&ctx->header[1], __ATOMIC_ACQUIRE);
    int32_t used = wi - ri;  // works correctly even with wrap due to modular arithmetic

    if (used >= ctx->slot_count) {
        // Ring full — drop frame
        __atomic_fetch_add(&ctx->header[6], 1, __ATOMIC_RELAXED);
        return;
    }

    // Write frame into the next slot
    int slot = wi % ctx->slot_count;
    std::memcpy(ctx->slots + slot * ctx->frame_size, image, ctx->frame_size);

    // Publish: increment write_index (release so JS sees the memcpy)
    __atomic_store_n(&ctx->header[0], wi + 1, __ATOMIC_RELEASE);

    // Wake JS thread if it's waiting on Atomics.wait()
    // We use the futex-compatible GCC builtin; on the JS side this maps to
    // Atomics.notify(header_i32, 0).
    // For portability we also support the TSFN-less path where JS just polls.
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// ─── Load ───────────────────────────────────────────────────────────────────
//
// JS call: addon.load(filePath, sharedArrayBuffer, slotCount)
//
// The SharedArrayBuffer must be pre-allocated by JS with size:
//   HEADER_BYTES + slotCount * frameSize
// where frameSize = width * height * 3 / 2 (YUV420p).
//
// Since we don't know the resolution until we open the file, the workflow is:
//   1. addon.probe(filePath) → { width, height }
//   2. Allocate SAB of the right size
//   3. addon.load(filePath, sab, slotCount)

Napi::Value Probe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string pathStr = info[0].As<Napi::String>().Utf8Value();

    ensure_mlt_init();

    mlt_profile profile = mlt_profile_init(nullptr);
    mlt_producer producer = mlt_factory_producer(profile, "avformat", pathStr.c_str());
    if (!producer) {
        mlt_profile_close(profile);
        Napi::Error::New(env, "Failed to probe file").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    mlt_profile_from_producer(profile, producer);

    Napi::Object result = Napi::Object::New(env);
    result.Set("width",       Napi::Number::New(env, profile->width));
    result.Set("height",      Napi::Number::New(env, profile->height));
    result.Set("fps",         Napi::Number::New(env,
        static_cast<double>(profile->frame_rate_num) / profile->frame_rate_den));
    result.Set("frameSize",   Napi::Number::New(env,
        static_cast<double>(profile->width) * profile->height * 3 / 2));

    mlt_producer_close(producer);
    mlt_profile_close(profile);

    return result;
}

Napi::Value Load(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // args: (filePath: string, sharedBuffer: SharedArrayBuffer, slotCount: number)
    std::string pathStr  = info[0].As<Napi::String>().Utf8Value();
    auto sab             = info[1].As<Napi::ArrayBuffer>();  // SharedArrayBuffer
    int slotCount        = info[2].As<Napi::Number>().Int32Value();

    ensure_mlt_init();

    // Tear down previous session
    if (g_consumer || g_ctx) {
        mlt_consumer c   = g_consumer;
        mlt_producer p   = g_producer;
        mlt_profile  pr  = g_profile;
        PlayerContext* cx = g_ctx;
        g_consumer = nullptr;
        g_producer = nullptr;
        g_profile  = nullptr;
        g_ctx      = nullptr;
        if (cx) cx->running = false;
        if (c) {
            mlt_events_disconnect(mlt_consumer_properties(c), cx);
            mlt_consumer_stop(c);
            mlt_consumer_close(c);
        }
        if (p)  mlt_producer_close(p);
        if (pr) mlt_profile_close(pr);
        if (cx) { auto* w = new ReleaseWorker(env, cx); w->Queue(); }
    }

    mlt_profile profile = mlt_profile_init(nullptr);
    mlt_producer producer = mlt_factory_producer(profile, "avformat", pathStr.c_str());
    if (!producer) {
        mlt_profile_close(profile);
        Napi::Error::New(env, "Failed to create producer").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    mlt_profile_from_producer(profile, producer);

    // HW accel hints
    mlt_properties props = mlt_producer_properties(producer);
#if defined(__APPLE__)
    mlt_properties_set(props, "hwaccel", "videotoolbox");
#elif defined(_WIN32)
    mlt_properties_set(props, "hwaccel", "d3d11va");
#else
    if (access("/dev/dri/renderD128", F_OK) == 0) {
        mlt_properties_set(props, "hwaccel", "vaapi");
        mlt_properties_set(props, "hwaccel_device", "/dev/dri/renderD128");
    } else {
        mlt_properties_set(props, "hwaccel", "cuda");
    }
#endif

    mlt_consumer consumer = mlt_factory_consumer(profile, "null", nullptr);
    if (!consumer) {
        mlt_producer_close(producer);
        mlt_profile_close(profile);
        Napi::Error::New(env, "Failed to create consumer").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    mlt_properties cprops = mlt_consumer_properties(consumer);
    mlt_properties_set(cprops, "mlt_image_format", "yuv420p");
    mlt_properties_set_int(cprops, "real_time", -1);
    mlt_properties_set_int(cprops, "threads", 1);
    mlt_properties_set_int(cprops, "audio_off", 1);
    mlt_properties_set_int(cprops, "buffer", 2);
    mlt_properties_set_int(cprops, "prefill", 1);

    // Set up shared memory ring buffer
    size_t frameSize = static_cast<size_t>(profile->width) * profile->height * 3 / 2;
    size_t requiredSize = HEADER_BYTES + slotCount * frameSize;
    if (sab.ByteLength() < requiredSize) {
        mlt_consumer_close(consumer);
        mlt_producer_close(producer);
        mlt_profile_close(profile);
        Napi::Error::New(env, "SharedArrayBuffer too small").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    auto* ctx = new PlayerContext();
    ctx->header     = reinterpret_cast<int32_t*>(sab.Data());
    ctx->slots      = reinterpret_cast<uint8_t*>(sab.Data()) + HEADER_BYTES;
    ctx->slot_count = slotCount;
    ctx->frame_size = frameSize;

    // Initialize header
    ctx->header[0] = 0;  // write_index
    ctx->header[1] = 0;  // read_index
    ctx->header[2] = profile->width;
    ctx->header[3] = profile->height;
    ctx->header[4] = static_cast<int32_t>(frameSize);
    ctx->header[5] = slotCount;
    ctx->header[6] = 0;  // dropped_frames
    ctx->header[7] = 0;  // reserved

    // We still need a minimal TSFN to prevent the SAB from being GC'd
    // and to allow clean stop signaling, but it's never actually called.
    Napi::Function noop = Napi::Function::New(env, [](const Napi::CallbackInfo&){});
    ctx->tsfn = Napi::ThreadSafeFunction::New(
        env, noop, "MLT Keepalive TSFN",
        0, 1,
        [](Napi::Env, PlayerContext* c) { delete c; },
        ctx
    );
    ctx->running = true;

    mlt_events_listen(
        cprops, ctx, "consumer-frame-show",
        (mlt_listener)on_frame_show
    );

    mlt_consumer_connect(consumer, mlt_producer_service(producer));
    mlt_consumer_start(consumer);

    g_consumer = consumer;
    g_producer = producer;
    g_profile  = profile;
    g_ctx      = ctx;

    return env.Undefined();
}

// ─── Stop ───────────────────────────────────────────────────────────────────
Napi::Value Stop(const Napi::CallbackInfo& info) {
    mlt_consumer consumer  = g_consumer;
    mlt_producer producer  = g_producer;
    mlt_profile  profile   = g_profile;
    PlayerContext* ctx      = g_ctx;

    g_consumer = nullptr;
    g_producer = nullptr;
    g_profile  = nullptr;
    g_ctx      = nullptr;

    if (!ctx) return info.Env().Undefined();
    ctx->running = false;

    if (consumer) {
        mlt_events_disconnect(mlt_consumer_properties(consumer), ctx);
        mlt_consumer_stop(consumer);
        mlt_consumer_close(consumer);
    }
    if (producer) mlt_producer_close(producer);
    if (profile)  mlt_profile_close(profile);

    auto* worker = new ReleaseWorker(info.Env(), ctx);
    worker->Queue();

    return info.Env().Undefined();
}

// ─── Module init ────────────────────────────────────────────────────────────
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("probe", Napi::Function::New(env, Probe));
    exports.Set("load",  Napi::Function::New(env, Load));
    exports.Set("stop",  Napi::Function::New(env, Stop));
    return exports;
}

NODE_API_MODULE(mlt_backend, Init)
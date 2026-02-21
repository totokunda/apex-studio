#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace media::contract {

// Bump when command semantics or field meaning changes.
inline constexpr uint32_t kControlAbiVersion = 1;

// Coordinate conventions shared with the renderer/Konva side:
// - Units: logical preview pixels (CSS pixel space).
// - Origin: top-left of the preview content rect.
// - Rotation: degrees, clockwise.
// - Opacity: 0..100.
// - Crop: normalized 0..1 in source-space.

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

enum class MaskKind : uint32_t {
  kUnknown = 0,
  kShape = 1,
  kLasso = 2,
  kTouch = 3,
};

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

struct RectI {
  int32_t x = 0;
  int32_t y = 0;
  int32_t width = 0;
  int32_t height = 0;
};

struct NormalizedCrop {
  double x = 0.0;
  double y = 0.0;
  double width = 1.0;
  double height = 1.0;
};

struct ClipTransform {
  double x = 0.0;
  double y = 0.0;
  double width = 0.0;
  double height = 0.0;
  double scale_x = 1.0;
  double scale_y = 1.0;
  double rotation_deg = 0.0;
  double opacity = 100.0;
  double corner_radius = 0.0;
  bool visible = true;
  bool has_crop = false;
  NormalizedCrop crop{};
};

struct FilterParams {
  double brightness = 0.0;
  double contrast = 0.0;
  double hue = 0.0;
  double saturation = 0.0;
  double blur = 0.0;
  double sharpness = 0.0;
  double noise = 0.0;
  double vignette = 0.0;
  double scan_lines = 0.0;
  double chromatic_aberration = 0.0;
  double interlace = 0.0;
  double pixelate = 0.0;
  double jitter = 0.0;
  std::string color_tint_color_hex{};
  double color_tint_intensity = 0.0;
};

struct LutParams {
  bool enabled = false;
  std::string lut_path{};
  double intensity = 1.0;
};

struct MaskParams {
  std::string mask_id{};
  MaskKind kind = MaskKind::kUnknown;
  bool enabled = true;
  bool inverted = false;
  double feather = 0.0;
  // Geometry/shape-specific payload is expected to be transport-defined for now.
  // Keep this as JSON text until binary mask payload format is finalized.
  std::string payload_json{};
};

struct TimelineRange {
  int64_t start_frame = 0;
  int64_t end_frame = 0;
  int64_t trim_start = 0;
  int64_t trim_end = 0;
  double speed = 1.0;
};

struct UpsertClipCommand {
  std::string clip_id{};
  ClipKind clip_kind = ClipKind::kUnknown;
  std::string asset_id{};
  std::string media_path{};
  TimelineRange timeline{};
  ClipTransform transform{};
  FilterParams filters{};
  std::vector<LutParams> luts{};
  std::vector<MaskParams> masks{};
  int32_t z_index = 0;
};

struct RemoveClipCommand {
  std::string clip_id{};
};

struct SetPlayStateCommand {
  bool is_playing = false;
};

struct SetPlayheadCommand {
  int64_t focus_frame = 0;
  double fps = 24.0;
  bool accurate_seek = false;
};

struct SetHoleRectCommand {
  RectI rect{};
  bool visible = false;
};

struct SetViewportCommand {
  int32_t width = 0;
  int32_t height = 0;
  double scale = 1.0;
  double stage_x = 0.0;
  double stage_y = 0.0;
};

struct ResetGraphCommand {};

struct ShutdownCommand {};

using CommandPayload = std::variant<UpsertClipCommand,
                                    RemoveClipCommand,
                                    SetPlayStateCommand,
                                    SetPlayheadCommand,
                                    SetHoleRectCommand,
                                    SetViewportCommand,
                                    ResetGraphCommand,
                                    ShutdownCommand>;

struct CommandEnvelope {
  uint32_t abi_version = kControlAbiVersion;
  uint64_t sequence = 0;
  CommandType type = CommandType::kResetGraph;
  CommandPayload payload{};
};

}  // namespace media::contract

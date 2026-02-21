#pragma once

namespace media {

struct EngineConfig {
  int preview_width = 0;
  int preview_height = 0;
  double fps = 24.0;
};

class Engine {
 public:
  Engine() = default;
  ~Engine() = default;

  bool Init(const EngineConfig& config);
};

}  // namespace media

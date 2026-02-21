#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "media_types.h"

namespace media_native {

struct ImageExecutorStats {
  uint64_t submitted_requests = 0;
  uint64_t processed_requests = 0;
  uint64_t cache_hits = 0;
  uint64_t failed_requests = 0;
  uint64_t dropped_requests = 0;
  uint64_t queue_depth = 0;
  uint64_t cached_images = 0;
};

class ImageExecutor {
 public:
  ImageExecutor();
  ~ImageExecutor();

  ImageExecutor(const ImageExecutor&) = delete;
  ImageExecutor& operator=(const ImageExecutor&) = delete;

  void Submit(std::vector<DecodeRequest>&& requests);
  void Reset();
  ImageExecutorStats Stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace media_native

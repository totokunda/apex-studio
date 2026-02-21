#include "image_executor.h"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

namespace media_native {

struct ImageExecutor::Impl {
  mutable std::mutex mu;
  std::condition_variable cv;
  bool stop_requested = false;
  std::deque<DecodeRequest> queue;
  std::unordered_set<std::string> image_cache;
  ImageExecutorStats stats{};
  std::thread thread;

  Impl() : thread(&Impl::ThreadMain, this) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stop_requested = true;
      queue.clear();
    }
    cv.notify_all();
    if (thread.joinable()) thread.join();
  }

  void Submit(std::vector<DecodeRequest>&& requests) {
    if (requests.empty()) return;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (stop_requested) return;
      for (auto& req : requests) {
        queue.push_back(std::move(req));
      }
      stats.submitted_requests += static_cast<uint64_t>(requests.size());
      stats.queue_depth = static_cast<uint64_t>(queue.size());
    }
    cv.notify_one();
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mu);
    stats.dropped_requests += static_cast<uint64_t>(queue.size());
    queue.clear();
    stats.queue_depth = 0;
  }

  ImageExecutorStats GetStats() const {
    std::lock_guard<std::mutex> lock(mu);
    ImageExecutorStats out = stats;
    out.queue_depth = static_cast<uint64_t>(queue.size());
    out.cached_images = static_cast<uint64_t>(image_cache.size());
    return out;
  }

  void ThreadMain() {
    for (;;) {
      DecodeRequest req{};
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&]() { return stop_requested || !queue.empty(); });
        if (stop_requested) break;
        req = std::move(queue.front());
        queue.pop_front();
        stats.queue_depth = static_cast<uint64_t>(queue.size());
      }

      bool ok = false;
      bool cache_hit = false;
      ProcessRequest(req, &ok, &cache_hit);

      {
        std::lock_guard<std::mutex> lock(mu);
        stats.processed_requests += 1;
        if (cache_hit) stats.cache_hits += 1;
        if (!ok) stats.failed_requests += 1;
      }
    }

    std::lock_guard<std::mutex> lock(mu);
    stats.dropped_requests += static_cast<uint64_t>(queue.size());
    queue.clear();
    stats.queue_depth = 0;
  }

  void ProcessRequest(const DecodeRequest& req, bool* ok, bool* cache_hit) {
    *ok = false;
    *cache_hit = false;
    if (req.media_path.empty()) return;

    {
      std::lock_guard<std::mutex> lock(mu);
      const auto it = image_cache.find(req.media_path);
      if (it != image_cache.end()) {
        *ok = true;
        *cache_hit = true;
        return;
      }
    }

    std::error_code ec;
    const bool exists = std::filesystem::exists(req.media_path, ec);
    if (ec || !exists) return;

    {
      std::lock_guard<std::mutex> lock(mu);
      image_cache.insert(req.media_path);
    }
    *ok = true;
  }
};

ImageExecutor::ImageExecutor() : impl_(std::make_unique<Impl>()) {}
ImageExecutor::~ImageExecutor() = default;

void ImageExecutor::Submit(std::vector<DecodeRequest>&& requests) {
  impl_->Submit(std::move(requests));
}

void ImageExecutor::Reset() { impl_->Reset(); }

ImageExecutorStats ImageExecutor::Stats() const { return impl_->GetStats(); }

}  // namespace media_native

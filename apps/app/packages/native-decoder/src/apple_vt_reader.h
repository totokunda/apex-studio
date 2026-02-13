#pragma once

#include <atomic>
#include <functional>
#include <string>

namespace apex {

bool AppleVTDecodeOnlyIterate(
    const std::string& filePath,
    double startTime,
    double endTime,
    std::atomic<bool>& cancelled,
    const std::function<bool(double timestamp, double duration)>& onFrame,
    std::string& error
);

} // namespace apex

#pragma once

#include <napi.h>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "decoder_instance.h"

namespace apex {

/**
 * Global registry of all active decoder instances, keyed by integer handle.
 * Thread-safe: all access goes through the registry mutex.
 */
class DecoderRegistry {
public:
    static DecoderRegistry& instance();

    int create();
    DecoderInstance* get(int handle);
    void remove(int handle);
    void removeAll();

private:
    DecoderRegistry() = default;
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<DecoderInstance>> decoders_;
    int nextHandle_ = 1;
};

// N-API exported functions
Napi::Value CreateDecoder(const Napi::CallbackInfo& info);
Napi::Value Configure(const Napi::CallbackInfo& info);
Napi::Value Seek(const Napi::CallbackInfo& info);
Napi::Value Iterate(const Napi::CallbackInfo& info);
Napi::Value AckFrame(const Napi::CallbackInfo& info);
Napi::Value CancelCurrent(const Napi::CallbackInfo& info);
Napi::Value Dispose(const Napi::CallbackInfo& info);
Napi::Value DisposeAll(const Napi::CallbackInfo& info);
Napi::Value GetCapabilities(const Napi::CallbackInfo& info);

} // namespace apex

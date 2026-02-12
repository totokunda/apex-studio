#include "addon.h"
#include "hwaccel.h"
#include <sstream>

namespace apex {

// ---------------------------------------------------------------------------
// DecoderRegistry — singleton managing all decoder instances
// ---------------------------------------------------------------------------

DecoderRegistry& DecoderRegistry::instance() {
    static DecoderRegistry reg;
    return reg;
}

int DecoderRegistry::create() {
    std::lock_guard<std::mutex> lock(mutex_);
    int handle = nextHandle_++;
    decoders_[handle] = std::make_unique<DecoderInstance>();
    return handle;
}

DecoderInstance* DecoderRegistry::get(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = decoders_.find(handle);
    return (it != decoders_.end()) ? it->second.get() : nullptr;
}

void DecoderRegistry::remove(int handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = decoders_.find(handle);
    if (it != decoders_.end()) {
        it->second->dispose();
        decoders_.erase(it);
    }
}

void DecoderRegistry::removeAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [handle, decoder] : decoders_) {
        decoder->dispose();
    }
    decoders_.clear();
}

// ---------------------------------------------------------------------------
// N-API exported functions
// ---------------------------------------------------------------------------

/**
 * createDecoder() → number (handle)
 */
Napi::Value CreateDecoder(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int handle = DecoderRegistry::instance().create();
    return Napi::Number::New(env, handle);
}

/**
 * configure(handle, {
 *   filePath: string,
 *   width: number,
 *   height: number,
 *   bufferPool: SharedArrayBuffer[],
 *   onFrame: (bufferIndex, width, height, timestamp, duration, requestId) => void,
 *   onError: (message: string) => void,
 *   onReady: () => void,
 * })
 */
Napi::Value Configure(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsObject()) {
        Napi::TypeError::New(env, "Expected (handle: number, options: object)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    int handle = info[0].As<Napi::Number>().Int32Value();
    auto opts = info[1].As<Napi::Object>();

    DecoderInstance* decoder = DecoderRegistry::instance().get(handle);
    if (!decoder) {
        Napi::Error::New(env, "Invalid decoder handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Extract configuration from JS options object
    DecoderConfig config;
    config.filePath = opts.Get("filePath").As<Napi::String>().Utf8Value();
    config.outputWidth = opts.Get("width").As<Napi::Number>().Int32Value();
    config.outputHeight = opts.Get("height").As<Napi::Number>().Int32Value();

    // Extract SharedArrayBuffer pool
    auto poolArray = opts.Get("bufferPool").As<Napi::Array>();
    for (uint32_t i = 0; i < poolArray.Length(); i++) {
        auto sab = poolArray.Get(i).As<Napi::ArrayBuffer>();
        config.bufferPool.push_back(static_cast<uint8_t*>(sab.Data()));
        config.bufferSizes.push_back(sab.ByteLength());
    }
    config.poolSize = static_cast<int>(config.bufferPool.size());

    // Create ThreadSafeFunctions for callbacks
    config.onFrame = Napi::ThreadSafeFunction::New(
        env,
        opts.Get("onFrame").As<Napi::Function>(),
        "NativeDecoderOnFrame",
        0,  // unlimited queue
        1   // initial thread count
    );

    config.onError = Napi::ThreadSafeFunction::New(
        env,
        opts.Get("onError").As<Napi::Function>(),
        "NativeDecoderOnError",
        0,
        1
    );

    config.onReady = Napi::ThreadSafeFunction::New(
        env,
        opts.Get("onReady").As<Napi::Function>(),
        "NativeDecoderOnReady",
        0,
        1
    );

    std::string error;
    bool ok = decoder->configure(config, error);

    if (!ok) {
        Napi::Error::New(env, "Configure failed: " + error)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return env.Undefined();
}

/**
 * seek(handle, timestamp, forceAccurate, requestId) → Promise<void>
 */
Napi::Value Seek(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int handle = info[0].As<Napi::Number>().Int32Value();
    double timestamp = info[1].As<Napi::Number>().DoubleValue();
    bool forceAccurate = info[2].As<Napi::Boolean>().Value();
    uint64_t requestId = static_cast<uint64_t>(info[3].As<Napi::Number>().Int64Value());

    DecoderInstance* decoder = DecoderRegistry::instance().get(handle);
    if (!decoder) {
        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Reject(Napi::Error::New(env, "Invalid decoder handle").Value());
        return deferred.Promise();
    }

    auto deferred = Napi::Promise::Deferred::New(env);

    // Store the deferred so the async operation can resolve/reject it
    decoder->seek(timestamp, forceAccurate, requestId,
        // onComplete callback (called from thread pool via TSFN)
        [deferred](bool success, const std::string& error) mutable {
            if (success) {
                deferred.Resolve(deferred.Env().Undefined());
            } else {
                deferred.Reject(
                    Napi::Error::New(deferred.Env(), error).Value());
            }
        }
    );

    return deferred.Promise();
}

/**
 * iterate(handle, startTime, endTime, requestId) → Promise<void>
 */
Napi::Value Iterate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int handle = info[0].As<Napi::Number>().Int32Value();
    double startTime = info[1].As<Napi::Number>().DoubleValue();
    double endTime = info[2].As<Napi::Number>().DoubleValue();
    uint64_t requestId = static_cast<uint64_t>(info[3].As<Napi::Number>().Int64Value());

    DecoderInstance* decoder = DecoderRegistry::instance().get(handle);
    if (!decoder) {
        auto deferred = Napi::Promise::Deferred::New(env);
        deferred.Reject(Napi::Error::New(env, "Invalid decoder handle").Value());
        return deferred.Promise();
    }

    auto deferred = Napi::Promise::Deferred::New(env);

    decoder->iterate(startTime, endTime, requestId,
        [deferred](bool success, const std::string& error) mutable {
            if (success) {
                deferred.Resolve(deferred.Env().Undefined());
            } else {
                deferred.Reject(
                    Napi::Error::New(deferred.Env(), error).Value());
            }
        }
    );

    return deferred.Promise();
}

/**
 * ackFrame(handle, bufferIndex) — release a buffer back to the pool
 */
Napi::Value AckFrame(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int handle = info[0].As<Napi::Number>().Int32Value();
    int bufferIndex = info[1].As<Napi::Number>().Int32Value();

    DecoderInstance* decoder = DecoderRegistry::instance().get(handle);
    if (decoder) {
        decoder->ackFrame(bufferIndex);
    }

    return env.Undefined();
}

/**
 * cancelCurrent(handle) — cancel in-progress seek/iterate
 */
Napi::Value CancelCurrent(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int handle = info[0].As<Napi::Number>().Int32Value();
    DecoderInstance* decoder = DecoderRegistry::instance().get(handle);
    if (decoder) {
        decoder->cancel();
    }

    return env.Undefined();
}

/**
 * dispose(handle) — close and release a single decoder
 */
Napi::Value Dispose(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    int handle = info[0].As<Napi::Number>().Int32Value();
    DecoderRegistry::instance().remove(handle);

    return env.Undefined();
}

/**
 * disposeAll() — close and release all decoders
 */
Napi::Value DisposeAll(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    DecoderRegistry::instance().removeAll();
    return env.Undefined();
}

/**
 * getCapabilities() → { hwAccelMethods: string[] }
 */
Napi::Value GetCapabilities(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    auto result = Napi::Object::New(env);
    auto methods = Napi::Array::New(env);

    auto available = HwAccelManager::getAvailableMethods();
    for (size_t i = 0; i < available.size(); i++) {
        methods.Set(static_cast<uint32_t>(i),
                    Napi::String::New(env, available[i]));
    }

    result.Set("hwAccelMethods", methods);
    result.Set("preferredMethod",
               Napi::String::New(env, HwAccelManager::getPreferredMethodName()));

    return result;
}

// ---------------------------------------------------------------------------
// Module initialization
// ---------------------------------------------------------------------------

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("createDecoder", Napi::Function::New(env, CreateDecoder));
    exports.Set("configure", Napi::Function::New(env, Configure));
    exports.Set("seek", Napi::Function::New(env, Seek));
    exports.Set("iterate", Napi::Function::New(env, Iterate));
    exports.Set("ackFrame", Napi::Function::New(env, AckFrame));
    exports.Set("cancelCurrent", Napi::Function::New(env, CancelCurrent));
    exports.Set("dispose", Napi::Function::New(env, Dispose));
    exports.Set("disposeAll", Napi::Function::New(env, DisposeAll));
    exports.Set("getCapabilities", Napi::Function::New(env, GetCapabilities));
    return exports;
}

NODE_API_MODULE(native_decoder, Init)

} // namespace apex

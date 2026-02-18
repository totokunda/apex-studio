#include <napi.h>
#include <framework/mlt.h>
#include <atomic>
#include <cstddef>
#include <cstring>


static bool g_mlt_initialized    = false;

static void ensure_mlt_init() {
    if (!g_mlt_initialized) {
        mlt_factory_init(nullptr);
        g_mlt_initialized = true;
    }
}


// ─── Module init ────────────────────────────────────────────────────────────
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return exports;
}

NODE_API_MODULE(mlt_backend, Init)
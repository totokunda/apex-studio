# @app/mlt-backend

N-API native addon for **MLT** (Media Lovin' Toolkit). Provides Node.js bindings to the MLT framework for video editing, transcoding, and composition.

## Prerequisites

### macOS (Homebrew)

```bash
brew install mlt
```

### Linux (apt)

```bash
sudo apt install libmlt-dev
```

### Verify MLT

```bash
pkg-config --cflags --libs mlt-framework-7
# Should output: -I/opt/homebrew/.../include/mlt-7 -L... -lmlt-7
```

## Build

From the `mlt-backend` directory:

```bash
cd apps/app/packages/mlt-backend
npm install
npm run build
```

Or from the app root (after `npm install` in mlt-backend):

```bash
cd apps/app/packages/mlt-backend && node-gyp rebuild
```

## Usage

```javascript
import { getVersion, getRepositoryDirectory, testConnection, testLite } from "@app/mlt-backend";

// Lite check (no MLT init - safe headless)
console.log(testLite()); // true

// Full MLT init (may segfault when run headless - no display)
console.log(testConnection());        // true
console.log(getVersion());             // "7.x (MLT framework initialized)"
console.log(getRepositoryDirectory()); // "/opt/homebrew/.../lib/mlt"
```

## Headless / CI

`mlt_factory_init()` can segfault when run without a display (Node worker, CI, etc.). Use `testLite()` to verify the addon loads. For full MLT, run in a process with a display (e.g. Electron main, interactive terminal). Set `QT_PLUGIN_PATH` on macOS if needed (see [melt-transcode.sh](../../../scripts/melt-transcode.sh)).

## API

| Function | Returns | Description |
|----------|---------|-------------|
| `testLite()` | `boolean` | Addon loaded, no MLT init (headless-safe) |
| `testConnection()` | `boolean` | Full MLT init + verify |
| `getVersion()` | `string` | Placeholder version info |
| `getRepositoryDirectory()` | `string \| null` | Path to MLT module directory |

## Project Structure

```
mlt-backend/
├── binding.gyp      # node-gyp build config (links mlt-framework-7)
├── package.json
├── README.md
└── src/
    ├── addon.cpp    # N-API + MLT C API
    └── index.ts     # Node.js wrapper
```

## IDE / C++ Linter

To remove "napi.h" / "node_api.h file not found" and MLT type errors:

1. **compile_flags.txt** (clangd): In `mlt-backend/` – includes node-addon-api, Node headers, MLT. Reload the window (Cmd+Shift+P → "Developer: Reload Window") if needed.

2. **c_cpp_properties.json** (MS C++ extension): `.vscode/c_cpp_properties.json` at workspace root configures include paths.

3. **nvm users**: If Node is via nvm, add your Node include path to `compile_flags.txt`, e.g. `-I$HOME/.nvm/versions/node/v25.6.1/include/node` (replace version as needed).

4. **Linux**: Use `-I/usr/include/node` and `-I/usr/include/mlt-7` (or from `pkg-config --cflags-only-I mlt-framework-7`) in `compile_flags.txt`.

## Extending

The addon initializes MLT via `mlt_factory_init(nullptr)` on first use. You can extend `addon.cpp` to:

- Create producers: `mlt_factory_producer(profile, service, resource)`
- Create consumers: `mlt_factory_consumer(profile, service, input)`
- Create filters: `mlt_factory_filter(profile, service, input)`
- Build compositions and export (e.g. avformat consumer)

See [MLT C API docs](https://www.mltframework.org/doxygen/) and [melt-transcode.sh](../../../scripts/melt-transcode.sh) for reference.

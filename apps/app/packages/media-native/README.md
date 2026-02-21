# @app/media-native

Scaffolding package for the cross-platform native media renderer pipeline.

## Current scope

- Buildable N-API addon shell (`src/addon.cc`)
- TypeScript wrapper API (`src/index.ts`)
- Native control contract for renderer/Konva -> engine commands:
  - `native/include/media/control_contract.h`
- Placeholder engine lifecycle:
  - `createEngine`
  - `attachSurface`
  - `submitCommand`
  - `getStats`
  - `destroyEngine`

## Near-term plan

1. Add native preview-surface binding for macOS/Windows/Linux.
2. Implement single-clip decode + OpenGL texture presentation.
3. Port existing WebGL mask/filter passes into native OpenGL nodes.
4. Integrate timeline command stream from renderer.

## Build

```bash
npm --prefix apps/app/packages/media-native run build
```

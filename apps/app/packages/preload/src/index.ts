// Barrel file that re-exports all preload helpers, split into feature modules.
// This keeps the external API identical while avoiding a single giant implementation file.

// Core crypto/versioning
export { sha256sum } from "./nodeCrypto.js";
export { versions } from "./versions.js";
export * from "./remoteVersions.js";

// Installer helpers (server bundle extraction, ffmpeg install)
export * from "./installer.js";

// App updates (Electron auto-updater)
export * from "./appUpdates.js";

// Engine/API updates (apex-engine self-update)
export * from "./apiUpdates.js";

// Media processing primitives
export { processMediaTo24 } from "./media/process.js";
export { getMediaRootAbsolute } from "./media/root.js";
export {
  AUDIO_EXTS,
  IMAGE_EXTS,
  VIDEO_EXTS,
  getLowercaseExtension,
} from "./media/fileExts.js";

import { pathToFileURL as pathToFileURLNode, fileURLToPath as fileURLToPathNode } from "node:url";

export const pathToFileURL = (path: string) => {
  return pathToFileURLNode(path).href;
};

export const fileURLToPath = (url: string) => {
  return fileURLToPathNode(url);
};

// Filters
export { fetchFilters } from "./filters/fetch.js";

// Shared types
export type { ConfigResponse } from "./types.js";
export type { ConvertedMediaItem } from "./media/library.js";

// Core IPC & filesystem helpers
export * from "./core/ipc.js";

// Media library and previews
export * from "./media/library.js";
export * from "./media/previews.js";
// Explicit re-export to ensure the virtual browser shim includes this new API.
export { exportVideoTranscodeWithFfmpeg } from "./media/previews.js";

// External (non-bundled) UI assets (downloaded on-demand)
export * from "./externalAssets.js";

// Configuration/system settings
export * from "./config.js";

// Projects, timelines, clips, masks metadata
export * from "./projects.js";

// Mask creation and tracking streams
export * from "./masks.js";

// Preprocessor management and jobs
export * from "./preprocessors.js";

// Components (model assets) management
export * from "./components.js";

// Generic WebSocket helpers
export * from "./ws.js";

// Manifest/model metadata helpers
export * from "./manifest.js";

// Engine run/status/result helpers
export * from "./engine.js";

// Postprocessor helpers
export * from "./postprocessor.js";

// Unified download helpers
export * from "./downloads.js";

// Generic jobs and Ray job helpers
export * from "./jobs.js";
export * from "./rayJobs.js";

// Settings helpers
export * from "./settings.js";

// Global offload defaults (persisted in main process, keyed by backend URL + manifest id)
export * from "./offload.js";

// Export result caching (persistent, electron-store)
export * from "./exportCache.js";

// Python process management (bundled API control)
export * from "./python.js";

// (uv removed) - we ship the full Python environment in the server bundle / python-api.
// Launcher helpers (gate the main UI behind a setup/launcher screen)
export * from "./launcher.js";

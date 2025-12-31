// Barrel file that re-exports all preload helpers, split into feature modules.
// This keeps the external API identical while avoiding a single giant implementation file.

// Core crypto/versioning
export { sha256sum } from "./nodeCrypto.ts";
export { versions } from "./versions.ts";

// Media processing primitives
export { processMediaTo24 } from "./media/process.ts";
export { getMediaRootAbsolute } from "./media/root.ts";
export {
  AUDIO_EXTS,
  IMAGE_EXTS,
  VIDEO_EXTS,
  getLowercaseExtension,
} from "./media/fileExts.ts";

import { pathToFileURL as pathToFileURLNode, fileURLToPath as fileURLToPathNode } from "node:url";

export const pathToFileURL = (path: string) => {
  return pathToFileURLNode(path).href;
};

export const fileURLToPath = (url: string) => {
  return fileURLToPathNode(url);
};

// Filters
export { fetchFilters } from "./filters/fetch.ts";

// Shared types
export type { ConfigResponse } from "./types.ts";
export type { ConvertedMediaItem } from "./media/library.ts";

// Core IPC & filesystem helpers
export * from "./core/ipc.ts";

// Media library and previews
export * from "./media/library.ts";
export * from "./media/previews.ts";

// Configuration/system settings
export * from "./config.ts";

// Projects, timelines, clips, masks metadata
export * from "./projects.ts";

// Mask creation and tracking streams
export * from "./masks.ts";

// Preprocessor management and jobs
export * from "./preprocessors.ts";

// Components (model assets) management
export * from "./components.ts";

// Generic WebSocket helpers
export * from "./ws.ts";

// Manifest/model metadata helpers
export * from "./manifest.ts";

// Engine run/status/result helpers
export * from "./engine.ts";

// Postprocessor helpers
export * from "./postprocessor.ts";

// Unified download helpers
export * from "./downloads.ts";

// Generic jobs and Ray job helpers
export * from "./jobs.ts";
export * from "./rayJobs.ts";

// Settings helpers
export * from "./settings.ts";

// Export result caching (persistent, electron-store)
export * from "./exportCache.ts";

// Python process management (bundled API control)
export * from "./python.ts";

import { ZoomLevel } from "./types";

export const MAX_ZOOM:ZoomLevel = 10 as ZoomLevel;          
export const MIN_ZOOM:ZoomLevel = 1 as ZoomLevel;
export const MIN_DURATION = 5;
export const BASE_LONG_SIDE = 600; // world units
export const TIMELINE_DURATION_SECONDS = 60;
export const DEFAULT_FPS = 16;
export const MAX_DURATION = TIMELINE_DURATION_SECONDS * DEFAULT_FPS;
export const FRAMES_CACHE_MAX_BYTES = 250 * 1024 * 1024; // 250 MB cap for decoded frames in frontend
import type {
  AnyClipProps,
  TimelineProps,
  PreprocessorClipProps,
  MaskClipProps,
  MaskData,
  ZoomLevel,
  Point,
  ViewTool,
  ShapeTool,
} from "../types";

export type ProjectSummary = {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
};

export type SerializedMask = Omit<MaskClipProps, "keyframes"> & {
  keyframes?: Record<string, MaskData>;
};

export type SerializedPreprocessor = Omit<
  PreprocessorClipProps,
  "startFrame" | "endFrame"
> & {
  startTicks?: number;
  endTicks?: number;
};

export type SerializedClip = Omit<
  AnyClipProps,
  | "startFrame"
  | "endFrame"
  | "trimStart"
  | "trimEnd"
  | "preprocessors"
  | "masks"
> & {
  startTicks: number;
  endTicks: number;
  trimStartTicks?: number;
  trimEndTicks?: number;
  preprocessors?: SerializedPreprocessor[];
  masks?: SerializedMask[];
};

export type PersistedClipState = {
  clips: SerializedClip[];
  timelines: TimelineProps[];
};

export type PersistedControlState = {
  fps: number;
  maxZoomLevel: ZoomLevel;
  minZoomLevel: ZoomLevel;
  zoomLevel: ZoomLevel;
  totalTicks: number;
  timelineStartTicks: number;
  timelineEndTicks: number;
  focusTicks: number;
  focusAnchorRatio: number;
  selectedClipIds: string[];
  isFullscreen: boolean;
};

export type PersistedViewportState = {
  scale: number;
  minScale: number;
  maxScale: number;
  position: Point;
  tool: ViewTool;
  shape: ShapeTool;
  clipPositions: Record<string, Point>;
  viewportSize: { width: number; height: number };
  contentBounds: { x: number; y: number; width: number; height: number } | null;
  aspectRatio: { width: number; height: number; id: string };
  isAspectEditing: boolean;
};

export type PersistedProjectSnapshot = {
  tickRate: number;
  clips?: PersistedClipState;
  controls?: PersistedControlState;
  viewport?: PersistedViewportState;
};

export type ProjectStateEnvelope = {
  success: boolean;
  data?: any;
  error?: string;
};

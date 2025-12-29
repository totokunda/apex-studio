import type {
  TimelineProps,
  AnyClipProps,
  ClipType,
  ClipSpecificProps,
  MaskClipProps,
  PreprocessorClipProps,
} from "../../types";

export type ClipPersistenceSlice = {
  setTimelines: (timelines: TimelineProps[]) => void;
  setClips: (clips: AnyClipProps[]) => void;
  _updateZoomLevel: (clips: AnyClipProps[], clipDuration: number) => void;
};

export type PersistedClipState = {
  clipId: string;
  timelineId: string;
  type: ClipType;
  groupId: string | null;
  startFrame: number;
  endFrame: number;
  trimStart: number;
  trimEnd: number;
  clipPadding: number;
  width?: number;
  height?: number;
  transform?: AnyClipProps["transform"];
  originalTransform?: AnyClipProps["originalTransform"];
  props: ClipSpecificProps | null;
};

export type MaskSnapshot = {
  id: string;
  clipId: string;
  mask: MaskClipProps;
};

export type PreprocessorSnapshot = {
  id: string;
  clipId: string;
  preprocessorName: string | null;
  preprocessor: PreprocessorClipProps;
};


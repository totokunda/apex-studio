import type {
  AnyClipProps,
  ClipSpecificProps,
  MaskClipProps,
  MaskData,
  PreprocessorClipProps,
  ClipType,
} from "../../types";
import type {
  PersistedClipState,
  MaskSnapshot,
  PreprocessorSnapshot,
} from "./types";

const BASE_PROP_KEYS = new Set([
  "clipId",
  "timelineId",
  "startFrame",
  "endFrame",
  "trimStart",
  "trimEnd",
  "clipPadding",
  "width",
  "height",
  "transform",
  "originalTransform",
  "type",
  "groupId",
  // Stored separately in their own tables
  "masks",
  "preprocessors",
]);

export function buildClipPropsForPersistence(clip: AnyClipProps): ClipSpecificProps | null {
  const props: Record<string, any> = {};

  for (const [key, value] of Object.entries(clip as any)) {
    if (BASE_PROP_KEYS.has(key)) continue;
    // Skip undefined to avoid bloating JSON
    if (value === undefined) continue;
    (props as any)[key] = value;
  }

  // Ensure hidden flag is always persisted via props (no dedicated DB column)
  if (typeof (clip as any).hidden === "boolean") {
    (props as any).hidden = (clip as any).hidden;
  }

  return Object.keys(props).length > 0 ? (props as ClipSpecificProps) : null;
}

export function buildPersistedStateMap(
  clips: AnyClipProps[],
): Map<string, PersistedClipState> {
  const map = new Map<string, PersistedClipState>();

  for (const clip of clips) {
    const key = String(clip.clipId);
    const timelineId = String(clip.timelineId ?? "");
    const startFrame = Number(clip.startFrame ?? 0) || 0;
    const endFrame = Number(clip.endFrame ?? 0) || 0;

    const trimStartRaw = (clip as any).trimStart;
    const trimEndRaw = (clip as any).trimEnd;

    const trimStart =
      typeof trimStartRaw === "number" && Number.isFinite(trimStartRaw)
        ? trimStartRaw
        : 0;
    const trimEnd =
      typeof trimEndRaw === "number" && Number.isFinite(trimEndRaw)
        ? trimEndRaw
        : 0;

    const clipPaddingRaw = (clip as any).clipPadding;
    const clipPadding =
      typeof clipPaddingRaw === "number" && Number.isFinite(clipPaddingRaw)
        ? clipPaddingRaw
        : 24;


    const state: PersistedClipState = {
      clipId: key,
      timelineId,
      type: clip.type as ClipType,
      groupId: (clip as any).groupId ?? null,
      startFrame,
      endFrame,
      trimStart,
      trimEnd,
      clipPadding,
      transform: clip.transform,
      originalTransform: clip.originalTransform,
      props: buildClipPropsForPersistence(clip),
    };
    map.set(key, state);
  }

  return map;
}

export function normalizeMaskKeyframesForPersistence(
  keyframes: MaskClipProps["keyframes"],
): Record<string | number, MaskData> {
  if (!keyframes) return {};
  if (keyframes instanceof Map) {
    const out: Record<number, MaskData> = {};
    for (const [frame, data] of keyframes.entries()) {
      out[frame] = data;
    }
    return out;
  }
  return { ...(keyframes as Record<string | number, MaskData>) };
}

export function buildMaskSnapshot(
  clips: AnyClipProps[],
): Map<string, MaskSnapshot> {
  const map = new Map<string, MaskSnapshot>();

  for (const clip of clips) {
    if (clip.type !== "video" && clip.type !== "image") continue;
    const clipKey = String(clip.clipId);
    const masks = (clip as any).masks as MaskClipProps[] | undefined;
    if (!Array.isArray(masks) || masks.length === 0) continue;

    for (const mask of masks) {
      if (!mask || !mask.id) continue;
      map.set(mask.id, {
        id: mask.id,
        clipId: clipKey,
        mask,
      });
    }
  }

  return map;
}

export function buildPreprocessorSnapshot(
  clips: AnyClipProps[],
): Map<string, PreprocessorSnapshot> {
  const map = new Map<string, PreprocessorSnapshot>();

  for (const clip of clips) {
    if (clip.type !== "video" && clip.type !== "image") continue;
    const clipKey = String(clip.clipId);

    const preprocessors = (clip as any)
      .preprocessors as PreprocessorClipProps[] | undefined;
    if (!Array.isArray(preprocessors) || preprocessors.length === 0) continue;

    for (const p of preprocessors) {
      if (!p || !p.id) continue;
      const name =
        (p as any).preprocessor?.id ??
        (p as any).preprocessor?.name ??
        null;
      map.set(p.id, {
        id: p.id,
        clipId: clipKey,
        preprocessorName: name,
        preprocessor: p,
      });
    }
  }

  return map;
}


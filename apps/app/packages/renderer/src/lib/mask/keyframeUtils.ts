import { getLocalFrame } from "@/lib/clip";
import { AnyClipProps, MaskClipProps, MaskData } from "@/lib/types";

type MaskKeyframes = Map<number, MaskData> | Record<number, MaskData>;

interface UpsertMaskKeyframeArgs {
  mask: MaskClipProps;
  clip?: AnyClipProps;
  focusFrame: number;
  updater: (previous: MaskData) => MaskData;
}

interface UpsertMaskKeyframeResult {
  keyframes: MaskKeyframes;
  frame: number;
  created: boolean;
  previousFrame: number | undefined;
}

const cloneKeyframes = (keyframes: MaskKeyframes): MaskKeyframes =>
  keyframes instanceof Map ? new Map(keyframes) : { ...keyframes };

const getSortedFrames = (keyframes: MaskKeyframes): number[] => {
  const frames =
    keyframes instanceof Map
      ? Array.from(keyframes.keys())
      : Object.keys(keyframes).map((key) => Number(key));

  return frames.sort((a, b) => a - b);
};

const hasFrame = (keyframes: MaskKeyframes, frame: number): boolean => {
  if (keyframes instanceof Map) {
    return keyframes.has(frame);
  }
  const record = keyframes as Record<number, MaskData>;
  return Object.prototype.hasOwnProperty.call(record, frame);
};

const getFrameData = (
  keyframes: MaskKeyframes,
  frame: number,
): MaskData | undefined => {
  if (keyframes instanceof Map) {
    return keyframes.get(frame);
  }
  const record = keyframes as Record<number, MaskData>;
  return record[frame];
};

export const upsertMaskKeyframe = ({
  mask,
  clip,
  focusFrame,
  updater,
}: UpsertMaskKeyframeArgs): UpsertMaskKeyframeResult | null => {
  const keyframes =
    mask.keyframes instanceof Map
      ? mask.keyframes
      : (mask.keyframes as Record<number, MaskData>);

  if (!keyframes) {
    return null;
  }

  const frames = getSortedFrames(keyframes);
  const hasExistingFrames = frames.length > 0;
  const isVideoClip = clip?.type === "video";
  const targetFrame = clip
    ? isVideoClip
      ? Math.max(0, Math.round(getLocalFrame(focusFrame, clip)))
      : 0
    : Math.max(0, Math.round(focusFrame));

  const frameExists = hasFrame(keyframes, targetFrame);
  const fallbackFrame = hasExistingFrames
    ? (frames.filter((frame) => frame <= targetFrame).pop() ?? frames[0])
    : undefined;

  const baseFrame = frameExists ? targetFrame : fallbackFrame;
  const baseData =
    baseFrame !== undefined ? getFrameData(keyframes, baseFrame) : undefined;
  const baseClone: MaskData = baseData ? { ...baseData } : {};
  const updatedData = updater(baseClone);

  const finalFrame = clip ? (isVideoClip ? targetFrame : 0) : targetFrame;

  const clonedKeyframes = cloneKeyframes(keyframes);
  if (clonedKeyframes instanceof Map) {
    clonedKeyframes.set(finalFrame, updatedData);
  } else {
    (clonedKeyframes as Record<number, MaskData>)[finalFrame] = updatedData;
  }

  return {
    keyframes: clonedKeyframes,
    frame: finalFrame,
    created: !frameExists && finalFrame !== baseFrame,
    previousFrame: baseFrame,
  };
};

import type { WebGLHaldClut } from "../../../renderer/src/components/preview/webgl-filters/hald-clut";
import type { BaseClipApplicator } from "../../../renderer/src/components/preview/clips/apply/base";
import { FilterPreview } from "../../../renderer/src/components/preview/clips/apply/filter";

export type ClipType =
  | "video"
  | "image"
  | "audio"
  | "model"
  | "text"
  | "lora"
  | "shape"
  | "draw"
  | "filter"
  | "group";

export interface AnyClipProps {
  type: ClipType;
  clipId: string;
  groupId?: string;
  timelineId?: string;
  startFrame?: number;
  endFrame?: number;
  children?: string[][]; // for group
}

export interface TimelineLike {
  timelineId: string;
  timelineY: number;
  hidden?: boolean;
}

export interface FilterClipProps extends AnyClipProps {
  type: "filter";
  smallPath?: string;
  fullPath?: string;
  category?: string;
  intensity?: number;
}

export interface ApplicatorFactoryConfig {
  haldClutInstance?: WebGLHaldClut | null;
  focusFrameOverride?: number;
}

export function createApplicatorFromClip(
  clip: AnyClipProps,
  config: ApplicatorFactoryConfig,
): BaseClipApplicator | null {
  switch (clip.type) {
    case "filter": {
      if (!config.haldClutInstance) {
        console.warn(
          "[Export ApplicatorFactory] Cannot create FilterPreview: haldClutInstance is null",
        );
        return null;
      }
      const filterApplicator = new FilterPreview(
        clip as FilterClipProps,
      ) as unknown as BaseClipApplicator & {
        setHaldClutInstance?: (h: WebGLHaldClut) => void;
        setFocusFrameOverride?: (f: number) => void;
        setStrength?: (s: number) => void;
      };
      if (filterApplicator.setHaldClutInstance) {
        filterApplicator.setHaldClutInstance(config.haldClutInstance);
      }
      if (
        typeof config.focusFrameOverride === "number" &&
        filterApplicator.setFocusFrameOverride
      ) {
        filterApplicator.setFocusFrameOverride(config.focusFrameOverride);
      }
      const intensity = (clip as FilterClipProps).intensity as
        | number
        | undefined;
      if (
        typeof intensity === "number" &&
        Number.isFinite(intensity) &&
        filterApplicator.setStrength
      ) {
        const normalized = Math.max(0, Math.min(1, intensity / 100));
        filterApplicator.setStrength(normalized);
      }
      return filterApplicator as BaseClipApplicator;
    }
    default:
      console.warn(
        `[Export ApplicatorFactory] Unsupported applicator type: ${clip.type}`,
      );
      return null;
  }
}

export function getApplicableClips(
  targetClip: AnyClipProps,
  allClips: AnyClipProps[],
  timelines: TimelineLike[],
  applicatorTypes: ClipType[] = ["filter"],
  focusFrame?: number,
): AnyClipProps[] {
  const getEffectiveTimelineY = (c: AnyClipProps): number => {
    const own = timelines.find((t) => t.timelineId === c.timelineId);
    if (own) return own.timelineY ?? 0;
    if (c.groupId) {
      const groupClip = allClips.find((x) => x.clipId === c.groupId);
      if (groupClip) {
        const groupTl = timelines.find(
          (t) => t.timelineId === groupClip.timelineId,
        );
        if (groupTl) return groupTl.timelineY ?? 0;
      }
    }
    return 0;
  };

  const targetY = getEffectiveTimelineY(targetClip);
  const targetGroupId = targetClip.groupId;
  const targetGroup = targetGroupId
    ? (allClips.find((c) => c.clipId === targetGroupId) as
        | AnyClipProps
        | undefined)
    : undefined;
  const targetChildrenNested =
    (targetGroup &&
      ((targetGroup as any).children as string[][] | undefined)) ??
    [];
  const targetChildrenFlat = targetChildrenNested.flat();
  const targetChildIndex = targetChildrenFlat.indexOf(targetClip.clipId);

  const applicableClips: AnyClipProps[] = allClips.filter((c) => {
    if (!applicatorTypes.includes(c.type)) return false;
    const effY = getEffectiveTimelineY(c);
    let allowedByVertical = effY < targetY;
    if (
      !allowedByVertical &&
      targetGroupId &&
      c.groupId === targetGroupId &&
      targetChildIndex !== -1
    ) {
      const idx = targetChildrenFlat.indexOf(c.clipId);
      if (idx !== -1 && idx < targetChildIndex) {
        allowedByVertical = true;
      }
    }
    if (!allowedByVertical) return false;
    const tlOwnerId =
      allClips.find((x) => x.clipId === c.groupId)?.timelineId || c.timelineId;
    const tl = timelines.find((t) => t.timelineId === tlOwnerId);
    if (tl?.hidden) return false;
    if (typeof focusFrame === "number") {
      const s = c.startFrame ?? 0;
      const e = c.endFrame ?? 0;
      return focusFrame >= s && focusFrame <= e;
    }
    return true;
  });

  applicableClips.sort((a, b) => {
    const yA = getEffectiveTimelineY(a);
    const yB = getEffectiveTimelineY(b);
    if (yA !== yB) return yA - yB;
    if (
      targetGroupId &&
      a.groupId === targetGroupId &&
      b.groupId === targetGroupId
    ) {
      const ia = targetChildrenFlat.indexOf(a.clipId);
      const ib = targetChildrenFlat.indexOf(b.clipId);
      if (ia !== ib) return ia - ib;
    }
    return (a.startFrame ?? 0) - (b.startFrame ?? 0);
  });

  return applicableClips;
}

export function getApplicatorsForClipExport(
  targetClip: AnyClipProps,
  allClips: AnyClipProps[],
  timelines: TimelineLike[],
  config: ApplicatorFactoryConfig,
  applicatorTypes?: ClipType[],
): BaseClipApplicator[] {
  const applicableClips = getApplicableClips(
    targetClip,
    allClips,
    timelines,
    applicatorTypes,
    config.focusFrameOverride,
  );
  const applicators = applicableClips
    .map((clip) => createApplicatorFromClip(clip, config))
    .filter((a): a is BaseClipApplicator => a !== null);
  return applicators;
}

import { useClipStore } from "./clip";
import { useControlsStore } from "./control";
import { AnyClipProps, ClipType, FilterClipProps } from "./types";
import { BaseClipApplicator } from "@/components/preview/clips/apply/base";
import { FilterPreview } from "@/components/preview/clips/apply/filter";
import { WebGLHaldClut } from "@/components/preview/webgl-filters/hald-clut";

/**
 * Supported applicator clip types that can be applied to other clips
 */
export const APPLICATOR_CLIP_TYPES: ClipType[] = ["filter"];

/**
 * Gets all applicable effect clips that should be applied to a given clip at the current frame.
 * Returns clips from timelines above the target clip's timeline, in the correct order (bottom to top).
 * Only includes clips of types that can be applied as effects (filters, masks, processors, etc.)
 *
 * @param clipId - The ID of the target clip to get applicators for
 * @param applicatorTypes - Optional array of clip types to filter by (defaults to all applicator types)
 * @returns Array of AnyClipProps sorted by timeline order (bottom to top for proper stacking)
 */
export function getApplicableClips(
  clipId: string,
  applicatorTypes: ClipType[] = APPLICATOR_CLIP_TYPES,
  focusFrameOverride?: number,
): AnyClipProps[] {
  const clipStore = useClipStore.getState();
  const controlStore = useControlsStore.getState();

  const clip = clipStore.getClipById(clipId);
  if (!clip) return [];

  const focusFrame =
    typeof focusFrameOverride === "number"
      ? focusFrameOverride
      : controlStore.focusFrame;
  const timelines = clipStore.timelines;
  const allClips = clipStore.clips;

  // Helper: effective timelineY for a clip, considering its group container
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

  // Determine the effective timelineY of the target clip
  const targetY = getEffectiveTimelineY(clip);
  const targetGroupId = clip.groupId;
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
  const targetChildIndex = targetChildrenFlat.indexOf(clip.clipId);

  // Collect applicable clips from timelines with smaller effective y (visually above target)
  const applicableClips: AnyClipProps[] = allClips.filter((c) => {
    if (!applicatorTypes.includes(c.type)) return false;
    // Ignore hidden timelines
    const effY = getEffectiveTimelineY(c);
    // Allow if strictly above (smaller y), OR if in the same group and ordered before target by child index
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
    const tl = timelines.find(
      (t) =>
        t.timelineId ===
        (allClips.find((x) => x.clipId === c.groupId)?.timelineId ||
          c.timelineId),
    );
    if (tl?.hidden) return false;
    // Time overlap
    const s = c.startFrame ?? 0;
    const e = c.endFrame ?? 0;
    return focusFrame >= s && focusFrame <= e;
  });

  // Sort top to bottom by effective timelineY (smaller y first), so lower (bigger y) end up later in the list and render later
  applicableClips.sort((a, b) => {
    const yA = getEffectiveTimelineY(a);
    const yB = getEffectiveTimelineY(b);
    // Smaller y first, larger y (lower on screen) last
    if (yA !== yB) return yA - yB;
    // If in the same group as target, order by child index ascending
    if (
      targetGroupId &&
      a.groupId === targetGroupId &&
      b.groupId === targetGroupId
    ) {
      const ia = targetChildrenFlat.indexOf(a.clipId);
      const ib = targetChildrenFlat.indexOf(b.clipId);
      if (ia !== ib) return ia - ib;
    }
    // Tie-breaker: earlier start first
    return (a.startFrame ?? 0) - (b.startFrame ?? 0);
  });

  return applicableClips;
}

/**
 * Configuration object for applicator factory
 */
export interface ApplicatorFactoryConfig {
  haldClutInstance?: WebGLHaldClut | null;
  focusFrameOverride?: number;
  // Add more shared resources here as needed (e.g., maskProcessor, etc.)
}

/**
 * Factory function to create an applicator instance from a clip.
 * Extend this function to support new applicator types.
 *
 * @param clip - The clip to create an applicator from
 * @param config - Shared resources and configuration for applicators
 * @returns BaseClipApplicator instance or null if clip type is not supported
 */
export function createApplicatorFromClip(
  clip: AnyClipProps,
  config: ApplicatorFactoryConfig,
): BaseClipApplicator | null {
  switch (clip.type) {
    case "filter": {
      // Don't create filter applicator if haldClutInstance is not available
      if (!config.haldClutInstance) {
        console.warn(
          "[ApplicatorFactory] Cannot create FilterPreview: haldClutInstance is null",
        );
        return null;
      }

      const filterApplicator = new FilterPreview(clip as FilterClipProps);
      filterApplicator.setHaldClutInstance(config.haldClutInstance);
      const intensity = (clip as FilterClipProps).intensity;
      if (typeof intensity === "number" && Number.isFinite(intensity)) {
        const normalized = Math.max(0, Math.min(1, intensity / 100));
        filterApplicator.setStrength(normalized);
      }
      return filterApplicator;
    }

    // Add more applicator types here as they are implemented
    // case 'mask': {
    //     const maskApplicator = new MaskPreview(clip as MaskClipProps);
    //     if (config.maskProcessor) {
    //         maskApplicator.setMaskProcessor(config.maskProcessor);
    //     }
    //     return maskApplicator;
    // }

    // case 'processor': {
    //     return new ProcessorPreview(clip as ProcessorClipProps);
    // }

    default:
      console.warn(
        `[ApplicatorFactory] Unsupported applicator type: ${clip.type}`,
      );
      return null;
  }
}

/**
 * Convenience function to get all applicators for a clip.
 * Combines getApplicableClips and createApplicatorFromClip.
 *
 * @param clipId - The ID of the target clip
 * @param config - Shared resources and configuration for applicators
 * @param applicatorTypes - Optional array of clip types to filter by
 * @returns Array of BaseClipApplicator instances ready to be applied
 */
export function getApplicatorsForClip(
  clipId: string,
  config: ApplicatorFactoryConfig,
  applicatorTypes?: ClipType[],
): BaseClipApplicator[] {
  const applicableClips = getApplicableClips(
    clipId,
    applicatorTypes,
    config.focusFrameOverride,
  );
  const applicators = applicableClips
    .map((clip) => createApplicatorFromClip(clip, config))
    .filter(
      (applicator): applicator is BaseClipApplicator => applicator !== null,
    )
    .map((app) => {
      if (typeof config.focusFrameOverride === "number") {
        (app as BaseClipApplicator).setFocusFrameOverride(
          config.focusFrameOverride,
        );
      }
      return app;
    });

  return applicators;
}

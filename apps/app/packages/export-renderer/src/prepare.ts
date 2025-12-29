import {
  createApplicatorFromClip,
  getApplicableClips,
  type AnyClipProps,
  type TimelineLike,
  type ClipType,
  type FilterClipProps,
} from "./applicators/utils";
import type {
  ExportClip,
  ExportApplicatorClip,
  ExportImageClip,
  ExportVideoClip,
  ExportTextClip,
  ExportShapeClip,
  ExportDrawClip,
} from "./exporter";
import type { ExportAudioClip } from "./exporter";

type ClipId = string;

function isGroup(clip: AnyClipProps): boolean {
  return clip.type === "group";
}

function isApplicator(clip: AnyClipProps): boolean {
  const applicatorTypes: ClipType[] = ["filter"];
  return applicatorTypes.includes(clip.type);
}

function isRenderableContent(clip: AnyClipProps): boolean {
  return (
    clip.type === "image" ||
    clip.type === "video" ||
    clip.type === "text" ||
    clip.type === "shape" ||
    clip.type === "draw" ||
    clip.type === "audio"
  );
}

function buildUngroupedView(
  allClips: AnyClipProps[],
  timelines: TimelineLike[],
  includeHiddenChildrenInGroups: boolean,
): { content: AnyClipProps[]; applicators: AnyClipProps[] } {
  const byId = new Map(allClips.map((c) => [c.clipId, c] as const));
  const content: AnyClipProps[] = [];
  const applicators: AnyClipProps[] = [];
  const seen = new Set<string>();

  const pushIfNew = (c: AnyClipProps) => {
    if (seen.has(c.clipId)) return;
    if (isTimelineHidden(timelines, c.timelineId)) return; // ignore hidden timelines entirely
    if (isApplicator(c)) applicators.push(c);
    else if (isRenderableContent(c)) content.push(c);
    seen.add(c.clipId);
  };

  for (const c of allClips) {
    if (isGroup(c)) {
      const children: string[][] = (c as any)?.children || [];
      for (const sub of children) {
        for (const id of sub) {
          const child = byId.get(id);
          if (!child) continue;
          if (!includeHiddenChildrenInGroups && (child as any).hidden) continue;
          // treat as ungrouped view: do not exclude hidden children when requested
          pushIfNew(child);
        }
      }
      continue;
    }
    // Non-group items
    pushIfNew(c);
  }

  return { content, applicators };
}

function getEffectiveTimelineY(
  clips: AnyClipProps[],
  timelines: TimelineLike[],
  clip: AnyClipProps,
): number {
  const own = timelines.find((t) => t.timelineId === clip.timelineId);
  if (own) return own.timelineY ?? 0;
  if (clip.groupId) {
    const group = clips.find((c) => c.clipId === clip.groupId);
    if (group) {
      const gt = timelines.find((t) => t.timelineId === group.timelineId);
      if (gt) return gt.timelineY ?? 0;
    }
  }
  return 0;
}

function isTimelineHidden(
  timelines: TimelineLike[],
  timelineId?: string,
): boolean {
  if (!timelineId) return false;
  const tl = timelines.find((t) => t.timelineId === timelineId);
  return !!tl?.hidden;
}

function flattenGroupChildrenBottomToTop(
  group: AnyClipProps | undefined,
  clips: AnyClipProps[],
  timelines: TimelineLike[],
): ClipId[] {
  if (!group) return [];
  const nested: string[][] = (group as any)?.children || [];
  if (!Array.isArray(nested) || nested.length === 0) return [];

  // Attach timelineY to each subarray via first present child
  const yForSub = (ids: string[]): number => {
    const first = ids
      .map((id) => clips.find((c) => c.clipId === id))
      .find(Boolean) as AnyClipProps | undefined;
    if (first)
      return getEffectiveTimelineY(clips, timelines, first as AnyClipProps);
    // fallback to group's own Y if no child found
    return getEffectiveTimelineY(clips, timelines, group);
  };

  // Sort subarrays by timelineY descending (bottom first), then within each by startFrame ascending
  const orderedSubs = [...nested]
    .map((sub) => ({ ids: sub.slice(), y: yForSub(sub) }))
    .sort((a, b) => b.y - a.y)
    .map(({ ids }) => ids);

  const mapById = new Map(clips.map((c) => [c.clipId, c] as const));
  const result: string[] = [];
  for (const sub of orderedSubs) {
    const sortedWithin = [...sub].sort((ida, idb) => {
      const a = mapById.get(ida);
      const b = mapById.get(idb);
      const sa = a?.startFrame ?? 0;
      const sb = b?.startFrame ?? 0;
      return sa - sb;
    });
    result.push(...sortedWithin);
  }
  return result;
}

function buildGroupOrderIndex(
  group: AnyClipProps | undefined,
  clips: AnyClipProps[],
  timelines: TimelineLike[],
): Map<string, number> {
  const flat = flattenGroupChildrenBottomToTop(group, clips, timelines);
  const idx = new Map<string, number>();
  flat.forEach((id, i) => idx.set(id, i));
  return idx;
}

function compareForRender(
  a: AnyClipProps,
  b: AnyClipProps,
  clips: AnyClipProps[],
  timelines: TimelineLike[],
  groupIndexCache: Map<string, Map<string, number>>,
): number {
  const ya = getEffectiveTimelineY(clips, timelines, a);
  const yb = getEffectiveTimelineY(clips, timelines, b);
  if (ya !== yb) return yb - ya; // bottom (larger y) first

  // Same effective Y; if in same group, use group's internal order bottom->top
  if (a.groupId && a.groupId === b.groupId) {
    const gid = a.groupId;
    let indexer = groupIndexCache.get(gid);
    if (!indexer) {
      const groupClip = clips.find((c) => c.clipId === gid);
      if (groupClip) {
        indexer = buildGroupOrderIndex(groupClip, clips, timelines);
        groupIndexCache.set(gid, indexer);
      }
    }
    const ia = indexer?.get(a.clipId) ?? 0;
    const ib = indexer?.get(b.clipId) ?? 0;
    if (ia !== ib) return ia - ib;
  }

  // Fallback tie-breakers
  const sa = a.startFrame ?? 0;
  const sb = b.startFrame ?? 0;
  if (sa !== sb) return sa - sb;
  return a.clipId.localeCompare(b.clipId);
}

function toExportApplicatorClip(clip: FilterClipProps): ExportApplicatorClip {
  return {
    clipId: clip.clipId,
    type: "filter",
    startFrame: clip.startFrame,
    endFrame: clip.endFrame,
    timelineId: clip.timelineId,
    // Pass through any known filter fields, ignore unknown
    ...(clip.smallPath ? ({ smallPath: clip.smallPath } as any) : {}),
    ...(clip.fullPath ? ({ fullPath: clip.fullPath } as any) : {}),
    ...(clip.category ? ({ category: clip.category } as any) : {}),
  } as ExportApplicatorClip;
}

function toExportClip(base: AnyClipProps): ExportClip | null {
  const common = {
    clipId: base.clipId,
    type: base.type as any,
    timelineId: base.timelineId,
    startFrame: base.startFrame,
    endFrame: base.endFrame,
    transform: (base as any).transform,
    originalTransform: (base as any).originalTransform,
  } as any;

  if (base.type === "image") {
    const src = (base as any).src as string | undefined;
    if (!src) return null;
    const out: ExportImageClip = {
      ...common,
      type: "image",
      src,
      brightness: (base as any).brightness,
      contrast: (base as any).contrast,
      hue: (base as any).hue,
      saturation: (base as any).saturation,
      blur: (base as any).blur,
      noise: (base as any).noise,
      sharpness: (base as any).sharpness,
      vignette: (base as any).vignette,
      masks: (base as any).masks,
      preprocessors: (base as any).preprocessors,
      trimStart: (base as any).trimStart,
    } as any;
    return out as ExportClip;
  }
  if (base.type === "video") {
    const src = (base as any).src as string | undefined;
    if (!src) return null;
    const out: ExportVideoClip = {
      ...common,
      type: "video",
      src,
      speed: (base as any).speed,
      trimStart: (base as any).trimStart,
      brightness: (base as any).brightness,
      contrast: (base as any).contrast,
      hue: (base as any).hue,
      saturation: (base as any).saturation,
      blur: (base as any).blur,
      noise: (base as any).noise,
      sharpness: (base as any).sharpness,
      vignette: (base as any).vignette,
      masks: (base as any).masks,
      preprocessors: (base as any).preprocessors,
    } as any;
    return out as ExportClip;
  }
  if (base.type === "audio") {
    const src = (base as any).src as string | undefined;
    if (!src) return null;
    const out: ExportAudioClip = {
      ...common,
      type: "audio",
      src,
      volume: (base as any).volume,
      fadeIn: (base as any).fadeIn,
      fadeOut: (base as any).fadeOut,
      speed: (base as any).speed,
      trimStart: (base as any).trimStart,
      trimEnd: (base as any).trimEnd,
    } as any;
    return out as ExportClip;
  }
  if (base.type === "text") {
    const out: ExportTextClip = {
      ...common,
      type: "text",
      text: (base as any).text,
    } as any;
    return out as ExportClip;
  }
  if (base.type === "shape") {
    const out: ExportShapeClip = {
      ...common,
      type: "shape",
      shapeType: (base as any).shapeType,
      fill: (base as any).fill,
      fillOpacity: (base as any).fillOpacity,
      stroke: (base as any).stroke,
      strokeOpacity: (base as any).strokeOpacity,
      strokeWidth: (base as any).strokeWidth,
      sides: (base as any).sides,
      points: (base as any).points,
    } as any;
    return out as ExportClip;
  }
  if (base.type === "draw") {
    const out: ExportDrawClip = {
      ...common,
      type: "draw",
      lines: (base as any).lines || [],
    } as any;
    return out as ExportClip;
  }
  return null;
}

export interface PrepareExportInput {
  // Selection to export: may include groups and/or individual clips
  clips: AnyClipProps[];
  // Universe of clips from which to resolve group children and applicators; defaults to clips
  allClips?: AnyClipProps[];
  timelines: TimelineLike[];
  includeHiddenChildrenInGroups?: boolean;
}

function flattenSelection(
  selection: AnyClipProps[],
  allClips: AnyClipProps[],
  timelines: TimelineLike[],
  includeHiddenChildrenInGroups: boolean,
): AnyClipProps[] {
  const byId = new Map(allClips.map((c) => [c.clipId, c] as const));
  const out: AnyClipProps[] = [];
  const seen = new Set<string>();
  const add = (c: AnyClipProps) => {
    if (!seen.has(c.clipId)) {
      out.push(c);
      seen.add(c.clipId);
    }
  };

  for (const item of selection) {
    if (isGroup(item)) {
      const children: string[][] = (item as any)?.children || [];
      for (const sub of children) {
        for (const id of sub) {
          const child = byId.get(id);
          if (!child) continue;
          if (!includeHiddenChildrenInGroups && (child as any).hidden) continue;
          if (isRenderableContent(child)) add(child);
        }
      }
    } else if (isRenderableContent(item)) {
      // Respect hidden unless explicitly selected
      add(item);
    }
  }
  return out;
}

export function prepareExportClips(input: PrepareExportInput): ExportClip[] {
  const { clips, timelines, includeHiddenChildrenInGroups = true } = input;
  const universe =
    Array.isArray(input.allClips) && input.allClips.length > 0
      ? input.allClips
      : clips;

  // Build ungrouped view (content + applicators), ignoring group containers entirely
  const { content: universeContent, applicators } = buildUngroupedView(
    universe,
    timelines,
    includeHiddenChildrenInGroups,
  );

  // Determine target content set from selection by flattening groups against the universe
  const targetContent = flattenSelection(
    clips,
    universe,
    timelines,
    includeHiddenChildrenInGroups,
  ).filter(
    (c) => isRenderableContent(c) && !isTimelineHidden(timelines, c.timelineId),
  );

  // Sort content clips for rendering order (bottom to top by timelineY, then by startFrame)
  const groupIndexCache = new Map<string, Map<string, number>>();
  targetContent.sort((a, b) =>
    compareForRender(a, b, universe, timelines, groupIndexCache),
  );

  // Compute applicators per clip with ungrouped vertical logic:
  // Any applicator on a timeline above the target's effective Y applies
  const out: ExportClip[] = [];
  for (const clip of targetContent) {
    const exportClip = toExportClip(clip);
    if (!exportClip) continue;
    const targetY = getEffectiveTimelineY(universe, timelines, clip);
    const appClips: ExportApplicatorClip[] = applicators
      .filter((a) => getEffectiveTimelineY(universe, timelines, a) < targetY)
      .sort((aClip, bClip) => {
        const ya = getEffectiveTimelineY(universe, timelines, aClip);
        const yb = getEffectiveTimelineY(universe, timelines, bClip);
        if (ya !== yb) return ya - yb; // top to bottom order for applicators
        const sa = aClip.startFrame ?? 0;
        const sb = bClip.startFrame ?? 0;
        return sa - sb;
      })
      .map((c) => toExportApplicatorClip(c as FilterClipProps));

    // Attach applicators list
    (exportClip as any).applicators = appClips;
    out.push(exportClip);
  }

  return out;
}

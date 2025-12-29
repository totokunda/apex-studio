import type { AnyClipProps, TimelineProps } from "@/lib/types";

/**
 * Shared clip stacking order helper used by both the preview canvas and
 * export preparation. It mirrors the logic previously inlined in
 * `Preview.tsx`:
 *
 * - Treat each group as a single sortable unit based on its timeline Y and
 *   startFrame.
 * - Non-group clips that are not group children are treated as single units.
 * - Units are sorted bottom-to-top (higher `timelineY` first), then by
 *   earlier `startFrame`.
 * - Groups are flattened back into their children in the order defined by the
 *   group's `children` nested arrays.
 */
export function sortClipsForStacking(
  clips: AnyClipProps[],
  timelines: TimelineProps[],
): AnyClipProps[] {
  type GroupUnit = {
    kind: "group";
    id: string;
    y: number;
    start: number;
    children: AnyClipProps[];
  };
  type SingleUnit = {
    kind: "single";
    y: number;
    start: number;
    clip: AnyClipProps;
  };

  const getTimelineY = (timelineId?: string) =>
    timelines.find((t) => t.timelineId === timelineId)?.timelineY ?? 0;

  const groups = clips.filter((c) => c.type === "group") as AnyClipProps[];
  const childrenSet = new Set<string>(
    groups.flatMap((g) => {
      const nested = ((g as any).children as string[][] | undefined) ?? [];
      return nested.flat();
    }),
  );

  // Build group units
  const groupUnits: GroupUnit[] = groups.map((g) => {
    const y = getTimelineY(g.timelineId);
    const start = g.startFrame ?? 0;
    const nested = ((g as any).children as string[][] | undefined) ?? [];
    const childIdsFlat = nested.flat();
    const children = childIdsFlat
      .map((id) => clips.find((c) => c.clipId === id))
      .filter(Boolean) as AnyClipProps[];
    return { kind: "group", id: g.clipId, y, start, children };
  });

  // Build single units for non-group, non-child clips
  const singleUnits: SingleUnit[] = clips
    .filter((c) => c.type !== "group" && !childrenSet.has(c.clipId))
    .map((c) => {
      const y = getTimelineY(c.timelineId);
      const start = c.startFrame ?? 0;
      return { kind: "single", y, start, clip: c };
    });

  // Sort units: lower on screen first (higher y), then earlier start
  const units = [...groupUnits, ...singleUnits].sort((a, b) => {
    if (a.y !== b.y) return b.y - a.y;
    return a.start - b.start;
  });

  // Flatten units back to clip list; for groups, expand children in their defined order
  const result: AnyClipProps[] = [];
  for (const u of units) {
    if (u.kind === "single") {
      result.push(u.clip);
    } else {
      // Ensure children are ordered as in group's children list (top-to-bottom)
      result.push(...u.children.reverse());
    }
  }

  return result;
}

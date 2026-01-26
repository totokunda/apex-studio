import type { ManifestComponent } from "@/lib/manifest/api";

/**
 * Scheduler components can appear multiple times in a manifest.
 * - If `component.name` is present, we use it as the stable key.
 * - Otherwise we generate keys:
 *   - first unnamed scheduler: "scheduler"
 *   - subsequent unnamed schedulers: "scheduler_2", "scheduler_3", ...
 *
 * This keeps backward compatibility for the common single-scheduler case
 * (where the key historically was just "scheduler").
 */
export function getSchedulerComponentKey(
  component: ManifestComponent,
  schedulerIndex: number,
): string {
  const named = String((component as any)?.name || "").trim();
  if (named) return named;
  if (schedulerIndex <= 0) return "scheduler";
  return `scheduler_${schedulerIndex + 1}`;
}



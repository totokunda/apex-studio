import { useJobProgress, useEngineJobActions } from "../../../lib/engine/hooks";
import { useClipStore } from "@/lib/clip";
import { ModelClipProps } from "@/lib/types";
import { useEffect, useRef, useState } from "react";
import { IoRefreshOutline } from "react-icons/io5";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ProgressPanelProps {
  clipId: string | null;
}

const ProgressPanel: React.FC<ProgressPanelProps> = ({ clipId }) => {
  const { getClipById } = useClipStore();
  const clip = getClipById(clipId ?? "") as ModelClipProps | undefined;
  const activeJobId = clip?.activeJobId ?? null;
  const job = useJobProgress(activeJobId);
  const { startTracking, stopTracking, fetchJobResult } = useEngineJobActions();
  const [spinning, setSpinning] = useState(false);

  // Ensure tracking stays active even if this panel unmounts
  useEffect(() => {
    if (!activeJobId) return;
    try {
      startTracking(activeJobId);
    } catch {}
  }, [activeJobId, startTracking]);

  // Build chronological (oldest -> newest) updates,
  // skipping preview frames and deduping while keeping the most recent duplicate
  const rawUpdates = (job?.updates || []) as Array<any>;
  const dedupeKeys = new Set<string>();
  const filteredFromEnd: Array<any> = [];

  for (let i = rawUpdates.length - 1; i >= 0; i--) {
    const u = rawUpdates[i];
    const pct =
      typeof u.progress === "number" ? Math.round(u.progress) : undefined;
    const msg = (u.message || job?.currentStep || "Working...")
      .toString()
      .trim()
      .replace(/\s+/g, " ");
    if (msg.includes("Preview frame")) continue;
    const status = (u as any)?.status || "";
    const key = `${msg}|${pct ?? ""}|${status}`;
    if (dedupeKeys.has(key)) continue;
    dedupeKeys.add(key);
    filteredFromEnd.push(u);
  }
  const displayUpdates = filteredFromEnd.reverse();

  // Auto-scroll to bottom as updates change (works within ScrollArea viewport)
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!bottomRef.current) return;
    bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [displayUpdates.length]);

  const handleRefresh = async () => {
    const id = job?.jobId || activeJobId;
    if (!id) return;
    try {
      setSpinning(true);
      await stopTracking(id);
      // brief delay to ensure socket cleanup
      await new Promise((r) => setTimeout(r, 50));
      await startTracking(id);
      await fetchJobResult(id);
    } finally {
      setTimeout(() => setSpinning(false), 500);
    }
  };

  return (
    <div className="p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="text-brand-light text-[11px] uppercase tracking-wide text-start font-medium">
          {" "}
          {clip?.manifest?.metadata?.name} Progress
        </div>
        <span
          className="text-brand-light text-sm cursor-pointer"
          onClick={handleRefresh}
        >
          <IoRefreshOutline
            className={spinning ? "animate-spin duration-500" : ""}
          />
        </span>
      </div>
      <ScrollArea className="w-full flex-1 min-h-0">
        {displayUpdates.length > 0 ? (
          <ul className="space-y-3">
            {displayUpdates.map((u, idx) => {
              const pct =
                typeof u.progress === "number"
                  ? Math.round(u.progress)
                  : undefined;
              const d = u.time ? new Date(u.time) : null;
              const dateStr = d
                ? d.toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })
                : "";
              const timeStr = d
                ? d.toLocaleTimeString(undefined, {
                    hour: "numeric",
                    minute: "2-digit",
                  })
                : "";
              const key = `${job?.jobId || clipId}-u-${idx}`;
              return (
                <li
                  key={key}
                  className="rounded-[6px] border border-brand-light/10 bg-brand p-2.5 w-full"
                >
                  <div className="flex items-start justify-between gap-3 min-w-0">
                    <div className="min-w-0 flex-1 overflow-hidden w-0">
                      <div
                        className="text-[10.5px] font-medium text-brand-light truncate text-start w-full max-w-full min-w-0"
                        title={(u.message || job?.currentStep || "Working...")
                          .toString()
                          .trim()
                          .replace(/\s+/g, " ")}
                      >
                        {(u.message || job?.currentStep || "Working...")
                          .trim()
                          .replace(/\s+/g, " ")}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {typeof pct === "number" && (
                        <div className="text-[10.5px] font-medium text-brand-light">
                          {pct}%
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[10px] text-brand-light/60">
                    <span className="text-start">{dateStr}</span>
                    <span className="text-end">{timeStr}</span>
                  </div>
                  {typeof pct === "number" && (
                    <div className="mt-2 h-1.5 w-full rounded bg-brand-light/10">
                      <div
                        className="h-1.5 rounded bg-brand-accent-two-shade"
                        style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                      />
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="text-[10.5px] text-brand-light/60 p-2.5 text-start w-full bg-brand rounded-[6px] border border-brand-light/10">
            No updates found.
          </div>
        )}
        <div ref={bottomRef} className="h-3" />
      </ScrollArea>
    </div>
  );
};

export default ProgressPanel;

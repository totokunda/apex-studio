import React, { useEffect, useMemo, useRef, useState } from "react";
import type { Preprocessor } from "@/lib/preprocessor/api";
import type { AnyClipProps } from "@/lib/types";
import { PreprocessorItem } from "@/components/menus/PreprocessorMenu";
import PreprocessorPage from "@/components/preprocessors/PreprocessorPage";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LuChevronDown, LuInfo } from "react-icons/lu";
import { useQuery } from "@tanstack/react-query";
import { fetchRayJobs, type RayJobStatus } from "@/lib/jobs/api";
import {
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  onUnifiedDownloadError,
  onUnifiedDownloadStatus,
  onUnifiedDownloadUpdate,
} from "@/lib/download/api";
import { useDownloadJobIdStore } from "@/lib/download/job-id-store";

interface PreprocessorPropertiesProps {
  preprocDetailId: string | null;
  setPreprocDetailId: (id: string | null) => void;
  preprocQuery: string;
  setPreprocQuery: (q: string) => void;
  compatiblePreprocessors: Preprocessor[];
  clip: AnyClipProps | undefined;
  currentPreprocessors:
    | Array<{ startFrame?: number; endFrame?: number }>
    | undefined;
  onAdd: (preproc: Preprocessor) => void;
}

const PreprocessorProperties: React.FC<PreprocessorPropertiesProps> = ({
  preprocDetailId,
  setPreprocDetailId,
  preprocQuery,
  setPreprocQuery,
  compatiblePreprocessors,
  clip,
  currentPreprocessors,
  onAdd,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>("__all__");
  const subscribedRef = useRef<Map<string, () => void>>(new Map());

  const { data: polledJobs = [] } = useQuery<RayJobStatus[]>({
    queryKey: ["rayJobs"],
    queryFn: fetchRayJobs,
    placeholderData: (prev) => prev ?? [],
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: 4000,
    refetchIntervalInBackground: true,
  });

  const sourceToJobId = useDownloadJobIdStore((s) => s.sourceToJobId);
  const addJobUpdate = useDownloadJobIdStore((s) => s.addJobUpdate);

  const categories = useMemo(() => {
    const set = new Set<string>();
    for (const p of compatiblePreprocessors) {
      if (p?.category) set.add(p.category);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [compatiblePreprocessors]);

  const filteredList = useMemo(() => {
    if (!selectedCategory || selectedCategory === "__all__")
      return compatiblePreprocessors;
    return compatiblePreprocessors.filter(
      (p) => p.category === selectedCategory,
    );
  }, [compatiblePreprocessors, selectedCategory]);

  const relevantDownloadJobIds = useMemo(() => {
    const visible = new Set(filteredList.map((p) => p.id));
    const ids: string[] = [];
    for (const [source, jobId] of Object.entries(sourceToJobId || {})) {
      if (!jobId) continue;
      if (!visible.has(source)) continue;
      ids.push(jobId);
    }
    return Array.from(new Set(ids));
  }, [filteredList, sourceToJobId]);

  // Ensure unified-download websocket updates are wired into the in-memory job store
  // while this panel is open. Without this, download completion may not be observed
  // until a detail page forces a refetch.
  useEffect(() => {
    const activeJobIds = new Set(relevantDownloadJobIds);

    // Connect to new jobs
    activeJobIds.forEach((jobId) => {
      if (subscribedRef.current.has(jobId)) return;
      const setup = async () => {
        try {
          await connectUnifiedDownloadWebSocket(jobId);
          const unsubUpdate = onUnifiedDownloadUpdate(jobId, (data) => {
            addJobUpdate(jobId, data);
          });
          const unsubStatus = onUnifiedDownloadStatus(jobId, (data) => {
            // PreprocessorItem primarily looks at `jobUpdates[*].status`,
            // so mirror status events into the updates stream.
            addJobUpdate(jobId, { status: data?.status } as any);
          });
          const unsubError = onUnifiedDownloadError(jobId, (data) => {
            addJobUpdate(jobId, { status: "error", error: data?.error } as any);
          });

          const cleanup = () => {
            try {
              unsubUpdate();
              unsubStatus();
              unsubError();
            } catch {}
            disconnectUnifiedDownloadWebSocket(jobId).catch(() => {});
          };

          subscribedRef.current.set(jobId, cleanup);
        } catch {
          // Best-effort: if WS cannot connect, polling can still drive completion.
        }
      };
      setup();
    });

    // Cleanup finished jobs
    for (const [jobId, cleanup] of subscribedRef.current.entries()) {
      if (!activeJobIds.has(jobId)) {
        try {
          cleanup();
        } catch {}
        subscribedRef.current.delete(jobId);
      }
    }
  }, [relevantDownloadJobIds, addJobUpdate]);

  useEffect(() => {
    return () => {
      for (const cleanup of subscribedRef.current.values()) {
        try {
          cleanup();
        } catch {}
      }
      subscribedRef.current.clear();
    };
  }, []);

  return (
    <>
      {preprocDetailId ? (
        <PreprocessorPage
          preprocessorId={preprocDetailId}
          onBack={() => setPreprocDetailId(null)}
        />
      ) : (
        <div className="flex flex-col">
          <div className="sticky top-0 z-10 bg-brand-background px-5 pt-3 pb-3 ">
            <div className="w-full flex items-center gap-2">
              <div className="relative bg-brand text-brand-light h-9 border border-brand-light/5 rounded-[6px] placeholder:text-brand-light/50 items-center flex w-full px-3 py-2.5 space-x-2 text-[10.5px]">
                <input
                  type="text"
                  placeholder="Search preprocessors…"
                  value={preprocQuery}
                  onChange={(e) => setPreprocQuery(e.target.value)}
                  className="w-full outline-none bg-brand"
                />
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="dark h-9 text-[10px] border-brand-light/5 bg-brand text-brand-light hover:bg-brand/80 rounded-[6px]"
                  >
                    {selectedCategory === "__all__"
                      ? "All categories"
                      : selectedCategory}
                    <LuChevronDown className="w-3.5! h-3.5!" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  className="bg-brand-background/95 backdrop-blur-sm text-brand-light border border-brand-light/10 font-poppins dark"
                >
                  <DropdownMenuRadioGroup
                    value={selectedCategory}
                    onValueChange={setSelectedCategory}
                  >
                    <DropdownMenuRadioItem
                      value="__all__"
                      className="text-[10.5px] font-medium"
                    >
                      All
                    </DropdownMenuRadioItem>
                    {categories.map((cat) => (
                      <DropdownMenuRadioItem
                        key={cat}
                        value={cat}
                        className="text-[10.5px] font-medium"
                      >
                        {cat}
                      </DropdownMenuRadioItem>
                    ))}
                  </DropdownMenuRadioGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <div className="px-5 py-3">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: "repeat(auto-fit, minmax(148px, 1fr))",
              }}
            >
              {filteredList.map((p) => (
                <div key={p.id} className="flex justify-center">
                  <PreprocessorItem
                    preprocessor={p}
                    onMoreInfo={(id) => setPreprocDetailId(id)}
                    onAdd={onAdd}
                    polledJobs={polledJobs}
                    addDisabled={(() => {
                      if (
                        !clip ||
                        (clip.type !== "video" && clip.type !== "image")
                      )
                        return true;
                      const duration = Math.max(
                        1,
                        (clip.endFrame ?? 0) - (clip.startFrame ?? 0),
                      );
                      if (duration <= 0) return true;
                      const intervals = (currentPreprocessors || [])
                        .map((pp: any) => {
                          const s = Math.max(0, pp.startFrame ?? 0);
                          const e = Math.max(
                            s + 1,
                            Math.min(duration, pp.endFrame ?? duration),
                          );
                          return [s, e] as [number, number];
                        })
                        .sort((a, b) => a[0] - b[0]);
                      let coverEnd = 0;
                      for (const [s, e] of intervals) {
                        if (s > coverEnd) {
                          return false; // gap exists
                        }
                        coverEnd = Math.max(coverEnd, e);
                        if (coverEnd >= duration) return true;
                      }
                      return coverEnd >= duration;
                    })()}
                    dimmed={(() => {
                      if (
                        !clip ||
                        (clip.type !== "video" && clip.type !== "image")
                      )
                        return false;
                      const duration = Math.max(
                        1,
                        (clip.endFrame ?? 0) - (clip.startFrame ?? 0),
                      );
                      if (duration <= 0) return true;
                      const intervals = (currentPreprocessors || [])
                        .map((pp: any) => {
                          const s = Math.max(0, pp.startFrame ?? 0);
                          const e = Math.max(
                            s + 1,
                            Math.min(duration, pp.endFrame ?? duration),
                          );
                          return [s, e] as [number, number];
                        })
                        .sort((a, b) => a[0] - b[0]);
                      let coverEnd = 0;
                      for (const [s, e] of intervals) {
                        if (s > coverEnd) {
                          return false;
                        }
                        coverEnd = Math.max(coverEnd, e);
                        if (coverEnd >= duration) return true;
                      }
                      return coverEnd >= duration;
                    })()}
                  />
                </div>
              ))}
              {compatiblePreprocessors.length === 0 && (
                <div className="text-[11px] text-brand-light/60 p-3.5 bg-brand rounded-md flex items-center gap-x-2">
                  <LuInfo className="w-4 h-4 text-brand-light/80" />
                  No preprocessors found
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default PreprocessorProperties;

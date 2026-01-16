import React, { useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import SemiGauge from "@/components/ui/semi-gauge";
import { fetchSystemMemory, SystemMemoryResponse } from "@/lib/system/api";
import { LuGauge, LuInfo, LuLoader } from "react-icons/lu";
import { useQuery } from "@tanstack/react-query";


const POLL_MS = 2000;
const SYSTEM_MEMORY_QUERY_KEY = ["systemMemory"] as const;

const SystemMemoryMenu: React.FC = () => {
  const [open, setOpen] = useState(false);
  const { data, isLoading } = useQuery<SystemMemoryResponse | null>({
    queryKey: SYSTEM_MEMORY_QUERY_KEY,
    queryFn: fetchSystemMemory,
    placeholderData: (prev) => prev ?? null,
    retry: false,
    refetchOnWindowFocus: false,
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });

  const fmt = (bytes?: number) => {
    if (!Number.isFinite(bytes || 0)) return "0 GB";
    const gb = (bytes as number) / 1024 ** 3;
    return `${gb.toFixed(1)} GB`;
  };

  const usageLine = (used?: number, total?: number, percent?: number) => {
    if (!Number.isFinite(used || 0) || !Number.isFinite(total || 0)) return "";
    const p = Number.isFinite(percent || NaN)
      ? ` (${Math.round(percent as number)}%)`
      : "";
    return `${fmt(used)} / ${fmt(total)}${p}`;
  };

  const summaryPercent = useMemo(() => {
    if (!data) return 0;
    if (data.unified) return data.unified.percent;
    if (data.gpu && data.gpu.count > 0) return data.gpu.percent;
    if (data.cpu) return data.cpu.percent;
    return 0;
  }, [data]);

  const gaugeCount = useMemo(() => {
    if (!data) return 1;
    if (data.unified) return 1;
    return 1 + (data.gpu ? 1 : 0); // CPU + optional GPU
  }, [data]);

  const contentWidth = Math.max(100 * gaugeCount, 320);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger className="text-brand-light/90 dark h-[34px] relative flex items-center space-x-2 w-32 px-3 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
        <LuGauge className="w-4 h-4" />
        <span className="text-[11px]">Memory</span>
        <span className="ml-1 text-[11px] text-brand-light/60">
          {Math.round(summaryPercent)}%
        </span>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px] p-4 font-poppins"
        style={{ width: contentWidth }}
      >
        {data ? (
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="text-[11px] uppercase tracking-wide text-brand-light/80 font-medium">
                System
              </div>
              <div className="text-[11px] text-brand-light/80 font-medium">
                {" "}
                {data.device_type.toUpperCase()}
              </div>
            </div>
            <div className="flex items-stretch justify-between gap-4">
              {data.unified ? (
                <div className="flex-1 flex flex-col items-center">
                  <SemiGauge
                    value={data.unified.percent}
                    label="Unified Memory"
                  />
                  <div className="mt-1 text-[11px] text-brand-light/80 font-medium">
                    {usageLine(data.unified.used, data.unified.total)}
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex-1 flex flex-col items-center">
                    <SemiGauge
                      value={data.cpu?.percent ?? 0}
                      label="System RAM"
                    />
                    {data.cpu ? (
                      <div className="mt-1 text-[11px] text-brand-light/60">
                        {usageLine(data.cpu.used, data.cpu.total)}
                      </div>
                    ) : null}
                  </div>
                  {data.gpu ? (
                    <div className="flex-1 flex flex-col items-center">
                      <SemiGauge value={data.gpu.percent} label="GPU VRAM" />
                      <div className="mt-1 text-[11px] text-brand-light/60">
                        {usageLine(data.gpu.used, data.gpu.total)}
                      </div>
                    </div>
                  ) : null}
                </>
              )}
            </div>

            {data.unified ? (
              <div className="mt-1">
                <div className="text-[11px] text-brand-light/90 font-medium mb-1">
                  Breakdown
                </div>
                <div className="grid grid-cols-2 gap-y-1 gap-x-3 text-[11px]">
                  <div className="text-brand-light/70">Physical Memory</div>
                  <div className="text-right text-brand-light">
                    {fmt(data.unified.total)}
                  </div>
                  <div className="text-brand-light/70">Memory Used</div>
                  <div className="text-right text-brand-light">
                    {usageLine(data.unified.used, data.unified.total)}
                  </div>
                  <div className="text-brand-light/70">Cached Files</div>
                  <div className="text-right text-brand-light">
                    {fmt((data.unified as any).details?.cached_files)}
                  </div>
                  <div className="text-brand-light/70">Wired Memory</div>
                  <div className="text-right text-brand-light">
                    {fmt((data.unified as any).details?.wired)}
                  </div>
                  <div className="text-brand-light/70">Compressed</div>
                  <div className="text-right text-brand-light">
                    {fmt((data.unified as any).details?.compressed)}
                  </div>
                  <div className="text-brand-light/70">Swap Used</div>
                  <div className="text-right text-brand-light">
                    {fmt((data.unified as any).details?.swap_used)}
                  </div>
                </div>
              </div>
            ) : data.gpu ? (
              <div className="mt-1">
                <div className="text-[11px] text-brand-light/70 mb-1">
                  GPU Adapters
                </div>
                <div className="space-y-1">
                  {data.gpu.adapters.map((a) => (
                    <div
                      key={a.index}
                      className="flex items-center justify-between text-[11px]"
                    >
                      <div className="text-brand-light/70">GPU {a.index}</div>
                      <div className="text-brand-light">
                        {usageLine(a.used, a.total, a.percent)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="text-[10px] text-brand-light/40 mt-1">
              Updates every 2s
            </div>
          </div>
        ) : (
          <div className="text-[12px] text-brand-light/70">
            <div className="flex items-center gap-x-2">
            {isLoading ? <LuLoader className="w-4 h-4 text-brand-light/80 animate-spin" /> : <LuInfo className="w-4 h-4 text-brand-light/80" />}
            <span>
              {isLoading ? "Loading…" : "Error loading system memory."}
            </span>
            </div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
};

export default SystemMemoryMenu;

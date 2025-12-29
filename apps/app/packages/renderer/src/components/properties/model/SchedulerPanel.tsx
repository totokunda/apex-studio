import React, { useEffect, useMemo, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ManifestComponent } from "@/lib/manifest/api";
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

interface SchedulerPanelProps {
  clipId: string;
  component: ManifestComponent;
  schedulerIndex: number;
}

const SchedulerPanel: React.FC<SchedulerPanelProps> = ({
  clipId,
  component,
  schedulerIndex,
}) => {
  const getClipById = useClipStore((s) => s.getClipById);
  const updateClip = useClipStore((s) => s.updateClip);
  const clip: any = getClipById(clipId);

  const componentKey = useMemo(() => {
    return getSchedulerComponentKey(component, schedulerIndex);
  }, [component, schedulerIndex]);

  const options = useMemo(() => {
    const all = Array.isArray((component as any)?.scheduler_options)
      ? ((component as any).scheduler_options as any[])
      : [];
    // Keep options stable but drop entries without a name.
    return all.filter((opt) => String(opt?.name || "").trim().length > 0) as Array<{
      name: string;
      label?: string;
      description?: string;
      base?: string;
      config_path?: string;
    }>;
  }, [component]);

  const selectedName: string | undefined = useMemo(() => {
    const fromKey = clip?.selectedComponents?.[componentKey]?.name;
    return String(fromKey ?? "") || undefined;
  }, [clip?.selectedComponents, componentKey]);

  // Ensure a default selection
  useEffect(() => {
    if (!clip || !options || options.length === 0) return;
    const names = options.map((o) => String(o?.name));
    const curr = selectedName;
    if (!curr || !names.includes(String(curr))) {
      const first = options[0];
      const next = {
        name: first?.name,
        base: first?.base,
        config_path: first?.config_path,
      };
      const nextSelected = {
        ...(clip.selectedComponents || {}),
        [componentKey]: next,
      };
      updateClip(clipId, { selectedComponents: nextSelected } as any);
    }
  }, [clip, clipId, options, selectedName, updateClip, componentKey]);

  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const selectedOption = useMemo(
    () => options.find((o) => String(o.name) === String(selectedName)),
    [options, selectedName],
  );

  return (
    <div className="w-full min-w-0">
      <div className="text-[10px] text-brand-light/70 mb-1.5 px-0.5 text-start">
        {String((component as any)?.label || (component as any)?.name || "").trim() ||
          (schedulerIndex > 0 ? `Scheduler ${schedulerIndex + 1}` : "Scheduler")}
      </div>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            ref={triggerRef}
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn(
              "w-full min-w-0 overflow-hidden justify-between rounded-[6px] dark bg-brand-background-light border border-brand-light/5 shadow-none px-3 text-brand-light text-[11px] h-13 hover:bg-brand-background-light/70",
            )}
            onClick={() => setOpen(true)}
          >
            <div className="flex flex-col min-w-0 w-0 flex-1 gap-y-0.5 overflow-hidden text-start">
              <span className="block truncate font-medium">
                {selectedOption?.label ||
                  selectedOption?.name ||
                  "Select scheduler"}
              </span>
              {selectedOption?.description && (
                <span className="block truncate text-[10px] text-brand-light/70 font-normal">
                  {selectedOption.description}
                </span>
              )}
            </div>
            <div className="flex flex-col items-center shrink-0">
              <ChevronUp className="h-3! w-3!  opacity-50" />
              <ChevronDown className="h-3! w-3!  opacity-50" />
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="p-0 bg-brand-background border-brand-light/10 font-poppins"
          style={{ width: triggerRef.current?.offsetWidth || 320 }}
        >
          <Command className="bg-brand-background">
            <CommandInput
              placeholder="Search scheduler"
              className="text-brand-light placeholder:text-brand-light/40 text-[11px]"
            />
            <CommandList>
              <CommandEmpty className="text-brand-light/40 p-3 text-xs">
                No scheduler found.
              </CommandEmpty>
              <CommandGroup className="px-1 py-1">
                {options.map((opt) => {
                  const isSelected = String(opt.name) === String(selectedName);
                  return (
                    <CommandItem
                      key={opt.name}
                      value={opt.name}
                      onSelect={(val) => {
                        const picked = options.find(
                          (o) => String(o.name) === String(val),
                        );
                        const next = picked
                          ? {
                              name: picked.name,
                              base: picked.base,
                              config_path: picked.config_path,
                            }
                          : { name: val };
                        const nextSelected = {
                          ...(clip?.selectedComponents || {}),
                          [componentKey]: next,
                        };
                        updateClip(clipId, {
                          selectedComponents: nextSelected,
                        } as any);
                        setOpen(false);
                      }}
                      className="flex items-center gap-2 px-2 py-2 hover:bg-brand-light/5 rounded-sm "
                    >
                      <div className="h-4 w-4 mt-0.5">
                        {isSelected ? (
                          <Check className="h-4 w-4 text-brand-lighter" />
                        ) : (
                          <span className="inline-block h-4 w-4" />
                        )}
                      </div>
                      <div className="flex flex-col min-w-0 max-w-full overflow-hidden">
                        <span className="block text-brand-light text-[11.5px] font-medium truncate">
                          {opt.label || opt.name}
                        </span>
                        {opt.description && (
                          <span className="block text-[10px] text-brand-light/70 truncate">
                            {opt.description}
                          </span>
                        )}
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default SchedulerPanel;

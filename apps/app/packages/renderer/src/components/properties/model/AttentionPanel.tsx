import React, { useEffect, useMemo, useRef, useState } from "react";
import { useClipStore } from "@/lib/clip";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Check } from "lucide-react";
import { cn } from "@/lib/utils";
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

interface AttentionPanelProps {
  clipId: string;
}

const AttentionPanel: React.FC<AttentionPanelProps> = ({ clipId }) => {
  const getClipById = useClipStore((s) => s.getClipById);
  const updateClip = useClipStore((s) => s.updateClip);
  const clip: any = getClipById(clipId);

  const options = useMemo(() => {
    const detail = (clip?.manifest?.spec?.attention_types_detail ||
      []) as Array<{ name: string; label?: string; description?: string }>;
    const seen = new Set<string>();
    return detail.filter((opt) => {
      const key = String(opt?.name || "");
      if (!key) return false;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [clip]);

  const selectedName: string | undefined = useMemo(() => {
    return String(clip?.selectedComponents?.attention?.name ?? "") || undefined;
  }, [clip?.selectedComponents]);

  // Ensure a default selection
  useEffect(() => {
    if (!clip || !options || options.length === 0) return;
    const names = options.map((o) => String(o?.name));
    const curr = selectedName;
    if (!curr || !names.includes(String(curr))) {
      const first = options[0];
      const next = { name: first?.name };
      console.log("next", next);
      const nextSelected = {
        ...(clip.selectedComponents || {}),
        attention: next,
      };
      updateClip(clipId, { selectedComponents: nextSelected } as any);
    }
  }, [clip, clipId, options, selectedName, updateClip]);

  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const selectedOption = useMemo(
    () => options.find((o) => String(o.name) === String(selectedName)),
    [options, selectedName],
  );

  return (
    <div className="w-full min-w-0">
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
            <div className="flex flex-col text-left min-w-0 w-0 flex-1 gap-y-0.5 overflow-hidden">
              <span className="block truncate font-medium">
                {selectedOption?.label ||
                  selectedOption?.name ||
                  "Select attention"}
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
              placeholder="Search attention"
              className="text-brand-light placeholder:text-brand-light/40 text-[11px]"
            />
            <CommandList>
              <CommandEmpty className="text-brand-light/40 p-3 text-xs">
                No attention found.
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
                          ? { name: picked.name }
                          : { name: val };
                        const nextSelected = {
                          ...(clip?.selectedComponents || {}),
                          attention: next,
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

export default AttentionPanel;

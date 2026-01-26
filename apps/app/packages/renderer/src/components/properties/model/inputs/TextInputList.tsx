import React, { useEffect, useMemo, useRef, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { LuInfo, LuPlus, LuTrash } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TextInputListProps {
  label?: string;
  description?: string;
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  defaultValue?: string[];
  maxItems?: number;
  maxScrollHeight?: number;
}

const TextInputList: React.FC<TextInputListProps> = ({
  label,
  description,
  value,
  onChange,
  placeholder,
  defaultValue,
  maxItems,
  maxScrollHeight = 260,
}) => {
  const [internalValue, setInternalValue] = useState<string[]>(
    Array.isArray(value)
      ? value
      : Array.isArray(defaultValue)
        ? defaultValue
        : [],
  );

  const debounceTimeoutRef = useRef<number | null>(null);
  const textareaRefs = useRef<Array<HTMLTextAreaElement | null>>([]);

  const canAddMore = useMemo(() => {
    if (typeof maxItems !== "number") return true;
    return internalValue.length < maxItems;
  }, [internalValue.length, maxItems]);

  // Keep internal value in sync if parent-controlled value changes externally
  useEffect(() => {
    setInternalValue(Array.isArray(value) ? value : []);
  }, [value]);

  // Auto-resize all textareas when values change
  useEffect(() => {
    textareaRefs.current.forEach((el) => {
      if (!el) return;
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";
    });
  }, [internalValue]);

  // Clear any pending debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current !== null) {
        window.clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  const emitDebounced = (next: string[]) => {
    if (debounceTimeoutRef.current !== null) {
      window.clearTimeout(debounceTimeoutRef.current);
    }
    debounceTimeoutRef.current = window.setTimeout(() => {
      onChange(next);
    }, 300);
  };

  const handleItemChange = (index: number, nextText: string) => {
    const next = internalValue.slice();
    next[index] = nextText;
    setInternalValue(next);
    emitDebounced(next);
  };

  const handleAdd = () => {
    if (!canAddMore) return;
    const next = [...internalValue, ""];
    setInternalValue(next);
    onChange(next);
  };

  const handleRemove = (index: number) => {
    const next = internalValue.slice();
    next.splice(index, 1);
    setInternalValue(next);
    onChange(next);
  };

  return (
    <div className="w-full min-w-0">
      <div className="flex flex-row items-center justify-between gap-2 mb-1">
        <div className="flex flex-row items-center gap-1.5 min-w-0">
          {label && (
            <label className="text-brand-light text-[10.5px] w-full font-medium text-start">
              {label}
            </label>
          )}
          {description && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="text-brand-light/70 hover:text-brand-light focus:outline-none shrink-0"
                  aria-label="Show description"
                >
                  <LuInfo className="w-3 h-3" />
                </button>
              </TooltipTrigger>
              <TooltipContent
                sideOffset={6}
                className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
              >
                {description}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        <div className="shrink-0 flex items-center gap-2">
          <span className="text-brand-light text-[10.5px] font-medium">
            {internalValue.length} / {maxItems ?? "∞"}
          </span>
          <button
            type="button"
            onClick={handleAdd}
            disabled={!canAddMore}
            title={canAddMore ? "Add" : "Reached max items"}
            className={cn(
              "h-7 px-2 rounded text-[10px] font-medium border border-brand-light/10 bg-brand text-brand-light/90 hover:bg-brand-light/10 transition-colors",
              !canAddMore && "opacity-40 cursor-not-allowed hover:bg-brand",
            )}
          >
            <div className="flex items-center gap-1">
              <LuPlus className="w-3 h-3" />
              <span>Add</span>
            </div>
          </button>
        </div>
      </div>
      {(() => {
        // Radix ScrollArea.Viewport is `h-full`, so the Root must have an explicit height
        // (maxHeight alone won't constrain the viewport). Only enable scrolling when needed
        // to avoid extra empty space for short lists.
        const shouldScroll = internalValue.length > 3;

        // Padding is important here because ScrollArea.Root is overflow-hidden,
        // and our delete button is slightly offset outside each card.
        const content = (
          <div className="flex flex-col gap-2 pr-3 pt-3 pb-2">
            {internalValue.length === 0 && (
              <div className="w-full flex flex-col items-start justify-center gap-y-1 py-3 opacity-70 bg-brand rounded-md border border-brand-light/5 px-3">
                <span className="text-brand-light text-[11px] font-medium">
                  No items added
                </span>
                <span className="text-brand-light/70 text-[9.5px]">
                  Click “Add” to create a new entry.
                </span>
              </div>
            )}

            {internalValue.map((item, index) => (
              <div key={`text-item-${index}`} className="relative w-full min-w-0">
                <button
                  type="button"
                  onClick={() => handleRemove(index)}
                  className="absolute shadow-lg -top-1 -right-1 z-50 rounded-full w-5 h-5 flex items-center justify-center bg-brand-background-dark border border-red-500 hover:bg-brand-background transition-all duration-200"
                  aria-label="Delete item"
                  title="Delete"
                >
                  <LuTrash className="w-2.5 h-2.5 text-red-400" />
                </button>

                <div className="flex flex-col items-start w-full gap-y-1 min-w-0 relative h-full px-3 py-3 pt-7 placeholder:text-brand-light/40 text-brand-light text-[11px] font-normal border border-brand-light/5 shadow bg-brand rounded-[7px]">
                  <label className="text-brand-light/70 text-[10px] w-full text-start font-medium absolute top-2 left-3">
                    {label ? `${label} ${index + 1}` : `Item ${index + 1}`}
                  </label>
                  <textarea
                    ref={(el) => {
                      textareaRefs.current[index] = el;
                    }}
                    value={item}
                    onChange={(e) => handleItemChange(index, e.target.value)}
                    placeholder={placeholder}
                    rows={2}
                    className="w-full h-full resize-none overflow-hidden dark focus-visible:outline-none focus-visible:ring-0 pb-1"
                  />
                </div>
              </div>
            ))}
          </div>
        );

        if (!shouldScroll) {
          return <div className="w-full mt-1">{content}</div>;
        }

        return (
          <ScrollArea
            className="w-full mt-1"
            style={{ height: maxScrollHeight }}
          >
            {content}
          </ScrollArea>
        );
      })()}
    </div>
  );
};

export default TextInputList;



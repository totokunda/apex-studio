import { cn } from "@/lib/utils";
import React, { useEffect, useState } from "react";
import { LuChevronDown, LuChevronUp, LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface NumberInputProps<T> {
  className?: string;
  label?: string;
  description?: string;
  value: T;
  onChange: (value: T) => void;
  startLogo?: string;
  canStep?: boolean;
  step?: number;
  max?: number;
  min?: number;
  emptyLabel?: boolean;
  toFixed?: number;
}

function isNumeric(value: unknown): boolean {
  return !isNaN(Number(value));
}

const NumberInput: React.FC<NumberInputProps<string>> = ({
  value,
  onChange,
  label,
  description,
  className,
  startLogo,
  step,
  max,
  min,
  emptyLabel,
  toFixed = 0,
}) => {
  const [tempValue, setTempValue] = useState(value);
  const lastValueRef = React.useRef(value);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    setTempValue(value);
    lastValueRef.current = value;
  }, [value]);

  const handleBlur = () => {
    setIsFocused(false);
    if (tempValue === value) return; // No change, skip
    const oldValue = lastValueRef.current;
    onChange(tempValue);

    // Check if change was accepted by seeing if value prop changed
    requestAnimationFrame(() => {
      // Schedule another check to see if parent updated the value
      requestAnimationFrame(() => {
        if (lastValueRef.current === oldValue) {
          // Value didn't change, reset to current value
          setTempValue(lastValueRef.current);
        }
      });
    });
  };

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      if (tempValue !== value) {
        const oldValue = lastValueRef.current;
        onChange(tempValue);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (lastValueRef.current === oldValue) {
              setTempValue(lastValueRef.current);
            }
          });
        });
      }
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setTempValue(value);
      e.currentTarget.blur();
    }
  };

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0">
      <div className="flex items-center gap-1.5">
        <label className="text-brand-light  text-[10px] font-medium text-start">
          {label}
        </label>
        {description && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="text-brand-light/70 hover:text-brand-light focus:outline-none"
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
      {emptyLabel && <span className="mb-3"></span>}

      <div className="relative w-full flex flex-row items-center min-w-0">
        {startLogo && (
          <span className="text-brand-light/50 font-medium text-[11px] absolute left-2 top-1/2 -translate-y-1/2">
            {startLogo}
          </span>
        )}
        <input
          className={cn(
            `w-full h-7 px-2 text-brand-light text-[11px] outline-none rounded-l font-normal items-center border border-brand-light/5  bg-brand ${className}`,
            {
              "pl-6": startLogo,
            },
          )}
          ref={inputRef}
          value={
            isNumeric(tempValue) && !isFocused
              ? Number(tempValue).toFixed(toFixed)
              : tempValue
          }
          onChange={(e) => setTempValue(e.target.value)}
          onBlur={handleBlur}
          onFocus={handleFocus}
          onKeyDown={handleKeyDown}
        />
        {
          <div className="flex flex-col items-center w-6 justify-center divide-y divide-brand-light/10 bg-brand  h-7 cursor-pointer rounded-r">
            <button
              className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center"
              onClick={() => {
                const numValue = Number(value);
                if (isNaN(numValue) || !isFinite(numValue)) return;
                const changedValue = Math.min(
                  numValue + (step ?? 0),
                  max ?? numValue + (step ?? 0),
                );
                onChange(changedValue.toString());
              }}
            >
              <LuChevronUp className="w-2 h-2 cursor-pointer text-brand-light" />
            </button>
            <button
              className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center"
              onClick={() => {
                const numValue = Number(value);
                if (isNaN(numValue) || !isFinite(numValue)) return;
                const changedValue = Math.max(
                  numValue - (step ?? 0),
                  min ?? numValue - (step ?? 0),
                );
                onChange(changedValue.toString());
              }}
            >
              <LuChevronDown className="w-2 h-2 cursor-pointer text-brand-light" />
            </button>
          </div>
        }
      </div>
      {/* description moved to tooltip */}
    </div>
  );
};

export default NumberInput;

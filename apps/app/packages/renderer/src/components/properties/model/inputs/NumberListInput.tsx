import { cn } from "@/lib/utils";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LuChevronDown,
  LuChevronLeft,
  LuChevronRight,
  LuChevronUp,
  LuInfo,
  LuPlus,
  LuX,
} from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface NumberListInputProps {
  className?: string;
  label?: string;
  description?: string;
  value: string; // comma-separated numbers
  onChange: (value: string) => void;
  startLogo?: string;
  step?: number;
  max?: number; // per-number max
  min?: number; // per-number min
  emptyLabel?: boolean;
  maxItems?: number; // optional max number of items in the list
  valueType?: "integer" | "float" | string;
  toFixed?: number;
}

const clampNumber = (num: number, min?: number, max?: number) => {
  if (typeof min === "number") num = Math.max(num, min);
  if (typeof max === "number") num = Math.min(num, max);
  return num;
};

const normalizeNumber = (raw: string, valueType?: string) => {
  if (!raw.trim()) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  if (valueType === "integer") return Math.trunc(n);
  return n;
};

const NumberListInput: React.FC<NumberListInputProps> = ({
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
  maxItems,
  valueType,
  toFixed = 2,
}) => {
  const [tempValue, setTempValue] = useState<string>("");
  const lastValueRef = useRef<string>(value);

  useEffect(() => {
    lastValueRef.current = value;
  }, [value]);

  const numbers = useMemo<number[]>(() => {
    if (!value) return [];
    return value
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n)) as number[];
  }, [value]);

  const canAddMore =
    typeof maxItems === "number" ? numbers.length < maxItems : true;

  const emit = (arr: number[]) => {
    const next = arr.join(",");
    onChange(next);
  };

  const tryAddNumber = () => {
    const parsed = normalizeNumber(tempValue, valueType);
    if (parsed == null) return;
    const clamped = clampNumber(parsed, min, max);
    if (!canAddMore) return;
    emit([...numbers, clamped]);
    setTempValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      tryAddNumber();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setTempValue("");
      (e.currentTarget as HTMLInputElement).blur();
    }
  };

  const incrementTemp = () => {
    const curr = normalizeNumber(
      tempValue || (numbers.length ? String(numbers[numbers.length - 1]) : "0"),
      valueType,
    );
    if (curr == null) return;
    const next = clampNumber(curr + (step ?? 0), min, max);
    setTempValue(
      valueType === "integer" ? String(Math.trunc(next)) : String(next),
    );
  };

  const decrementTemp = () => {
    const curr = normalizeNumber(
      tempValue || (numbers.length ? String(numbers[numbers.length - 1]) : "0"),
      valueType,
    );
    if (curr == null) return;
    const next = clampNumber(curr - (step ?? 0), min, max);
    setTempValue(
      valueType === "integer" ? String(Math.trunc(next)) : String(next),
    );
  };

  const removeAt = (idx: number) => {
    const arr = numbers.slice();
    arr.splice(idx, 1);
    emit(arr);
  };

  const moveLeft = (idx: number) => {
    if (idx <= 0) return;
    const arr = numbers.slice();
    const [it] = arr.splice(idx, 1);
    arr.splice(idx - 1, 0, it);
    emit(arr);
  };

  const moveRight = (idx: number) => {
    if (idx >= numbers.length - 1) return;
    const arr = numbers.slice();
    const [it] = arr.splice(idx, 1);
    arr.splice(idx + 1, 0, it);
    emit(arr);
  };

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0">
      <div className="flex items-center gap-1.5 flex-row justify-between w-full">
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
        {typeof maxItems === "number" && (
          <span className="text-[9px] text-brand-light/60 ml-auto">
            {numbers.length}/{maxItems}
          </span>
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
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={numbers.length === 0 ? "Add number and press Enter" : ""}
        />
        <div className="flex flex-col items-center w-6 justify-center divide-y divide-brand-light/10 bg-brand  h-7 cursor-pointer rounded-r">
          <button
            className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center"
            onClick={incrementTemp}
          >
            <LuChevronUp className="w-2 h-2 cursor-pointer text-brand-light" />
          </button>
          <button
            className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center"
            onClick={decrementTemp}
          >
            <LuChevronDown className="w-2 h-2 cursor-pointer text-brand-light" />
          </button>
        </div>
        <button
          className={cn(
            "ml-1 h-7 px-2 rounded text-[10px] font-medium border border-brand-light/10 bg-brand text-brand-light/90 hover:bg-brand-light/10 transition-colors",
            !canAddMore && "opacity-40 cursor-not-allowed hover:bg-brand",
          )}
          onClick={tryAddNumber}
          disabled={!canAddMore}
          title={canAddMore ? "Add" : "Reached max items"}
        >
          <div className="flex items-center gap-1">
            <LuPlus className="w-3 h-3" />
            <span>Add</span>
          </div>
        </button>
      </div>

      <div className="w-full flex items-center justify-between">
        <div className="flex items-center gap-1 flex-wrap">
          {numbers.map((num, idx) => (
            <span
              key={`${num}-${idx}-${numbers.length}`}
              className="text-[9.5px] text-brand-light bg-brand border shadow border-brand-light/10 rounded px-1.5 py-0.5 flex items-center gap-1"
            >
              <button
                className={cn(
                  "p-0.5 rounded hover:bg-brand/40 disabled:opacity-40 disabled:cursor-not-allowed",
                )}
                onClick={() => moveLeft(idx)}
                disabled={idx === 0}
                aria-label="Move left"
              >
                <LuChevronLeft className="w-3 h-3" />
              </button>
              <span className="px-0.5">
                {valueType === "integer"
                  ? Math.trunc(num)
                  : num.toFixed(toFixed)}
              </span>
              <button
                className={cn(
                  "p-0.5 rounded hover:bg-brand/40 disabled:opacity-40 disabled:cursor-not-allowed",
                )}
                onClick={() => moveRight(idx)}
                disabled={idx === numbers.length - 1}
                aria-label="Move right"
              >
                <LuChevronRight className="w-3 h-3" />
              </button>
              <button
                className="p-0.5 rounded hover:bg-brand/40"
                onClick={() => removeAt(idx)}
                aria-label="Remove"
              >
                <LuX className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default NumberListInput;

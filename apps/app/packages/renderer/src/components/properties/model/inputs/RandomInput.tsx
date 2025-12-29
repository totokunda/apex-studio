import { cn } from "@/lib/utils";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { LuDice5, LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface RandomInputProps<T> {
  className?: string;
  label?: string;
  description?: string;
  value: T; // string value; "-1" signifies auto-random on generate
  onChange: (value: T) => void;
  startLogo?: string;
  min?: number;
  max?: number;
  step?: number;
  emptyLabel?: boolean;
}

const RandomInput: React.FC<RandomInputProps<string>> = ({
  value,
  onChange,
  label,
  description,
  className,
  startLogo,
  min,
  max,
  step = 1,
  emptyLabel,
}) => {
  const isAuto = String(value) === "-1";
  const [tempValue, setTempValue] = useState(
    isAuto ? "Auto" : String(value ?? ""),
  );
  const lastValueRef = useRef(value);
  const lastManualValueRef = useRef<string>("");

  // remember last manual value so we can restore when leaving Auto
  useEffect(() => {
    const strVal = String(value);
    if (strVal !== "-1") {
      lastManualValueRef.current = strVal;
    }
  }, [value]);

  useEffect(() => {
    if (isAuto) {
      setTempValue("Auto");
    } else {
      setTempValue(String(value ?? ""));
    }
    lastValueRef.current = value;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, isAuto]);

  const minBound = useMemo(() => (typeof min === "number" ? min : 0), [min]);
  const maxBound = useMemo(() => (typeof max === "number" ? max : 100), [max]);
  const stepSize = useMemo(
    () => (typeof step === "number" && step > 0 ? step : 1),
    [step],
  );

  const clampToRange = (n: number) => Math.min(Math.max(n, minBound), maxBound);

  const quantizeToStep = (n: number) => {
    const steps = Math.round((n - minBound) / stepSize);
    return minBound + steps * stepSize;
  };

  const generateRandomValue = () => {
    const r = Math.random();
    const raw = minBound + r * (maxBound - minBound);
    const quantized = quantizeToStep(raw);
    const clamped = clampToRange(quantized);
    // avoid -0 for negative ranges
    const fixed = Math.abs(clamped) === 0 ? 0 : clamped;
    // stringify without trailing decimals if integer step
    const isIntegerStep =
      Number.isInteger(stepSize) &&
      Number.isInteger(minBound) &&
      Number.isInteger(maxBound);
    return isIntegerStep ? String(Math.round(fixed)) : String(fixed);
  };

  const handleBlur = () => {
    if (isAuto) return; // no editing in Auto
    if (tempValue === value) return; // No change, skip
    const oldValue = lastValueRef.current;
    onChange(tempValue);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (lastValueRef.current === oldValue) {
          setTempValue(String(lastValueRef.current));
        }
      });
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (isAuto) return;
    if (e.key === "Enter") {
      if (tempValue !== value) {
        const oldValue = lastValueRef.current;
        onChange(tempValue);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (lastValueRef.current === oldValue) {
              setTempValue(String(lastValueRef.current));
            }
          });
        });
      }
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setTempValue(String(value ?? ""));
      e.currentTarget.blur();
    }
  };

  const switchToAuto = () => {
    onChange("-1");
  };

  const switchToManual = () => {
    const fallback = String(typeof min === "number" ? min : 0);
    const next = lastManualValueRef.current || fallback;
    onChange(next);
  };

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0">
      <div className="flex items-center gap-1.5 w-full">
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

        <div className="ml-auto flex rounded-[4px] overflow-hidden border border-brand-light/10">
          <button
            type="button"
            onClick={switchToManual}
            className={cn(
              "px-2 py-0.5 text-[10px] font-medium transition-all duration-200",
              !isAuto
                ? "bg-brand-light/[0.075] text-brand-lighter"
                : "bg-brand text-brand-light/70 hover:bg-brand-light/5",
            )}
          >
            Manual
          </button>
          <button
            type="button"
            onClick={switchToAuto}
            className={cn(
              "px-2 py-0.5 text-[10px] font-medium transition-all duration-200 border-l border-brand-light/10",
              isAuto
                ? "bg-brand-light/[0.075] text-brand-lighter"
                : "bg-brand text-brand-light/70 hover:bg-brand-light/5",
            )}
          >
            Auto
          </button>
        </div>
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
            `w-full h-7 px-2 text-brand-light text-[11px] rounded-l font-normal items-center border border-brand-light/5  bg-brand ${className}`,
            {
              "pl-6": startLogo,
            },
          )}
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          readOnly={isAuto}
        />
        <div className="flex items-center w-6 justify-center bg-brand h-7 cursor-pointer rounded-r">
          <button
            type="button"
            className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center"
            onClick={() => {
              const randomValue = generateRandomValue();
              onChange(randomValue);
            }}
          >
            <LuDice5 className="w-3 h-3 text-brand-light" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default RandomInput;

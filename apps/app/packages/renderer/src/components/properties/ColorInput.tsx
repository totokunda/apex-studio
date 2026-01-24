import React, { useState, useEffect } from "react";
import {
  ColorPicker,
  ColorPickerSelection,
  ColorPickerEyeDropper,
  ColorPickerHue,
  ColorPickerAlpha,
  ColorPickerOutput,
  ColorPickerFormat,
} from "@/components/ui/shadcn-io/color-picker";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import Color from "color";
import { cn } from "@/lib/utils";

interface ColorInputProps {
  value: string;
  onChange: (value: string) => void;
  percentValue?: number;
  setPercentValue?: (value: number) => void;
  label?: string;
  labelClass?: string;
  size?: "small" | "medium";
}

const ColorInput: React.FC<ColorInputProps> = ({
  value,
  onChange,
  label,
  percentValue,
  setPercentValue,
  labelClass,
  size = "small",
}) => {
  const [tempColorValue, setTempColorValue] = useState(value);
  const [tempPercentValue, setTempPercentValue] = useState(
    percentValue?.toString() ?? "100",
  );
  const lastColorValueRef = React.useRef(value);
  const lastPercentValueRef = React.useRef(percentValue ?? 100);
  const isUserInteractingRef = React.useRef(false);

  useEffect(() => {
    setTempColorValue(value);
    lastColorValueRef.current = value;
  }, [value]);

  useEffect(() => {
    setTempPercentValue(percentValue?.toString() ?? "100");
    lastPercentValueRef.current = percentValue ?? 100;
  }, [percentValue]);

  const isValidHexColor = (color: string): boolean => {
    return /^#([0-9A-F]{3}){1,2}$/i.test(color);
  };

  const validateAndConvertColor = (color: string): string | null => {
    // If it's already a valid hex color, return it
    if (isValidHexColor(color)) {
      return color;
    }

    // Try to parse as CSS color (named colors, rgb, rgba, hsl, etc.)
    const ctx = document.createElement("canvas").getContext("2d");
    if (!ctx) return null;

    // Set the color and let the browser parse it
    ctx.fillStyle = color.trim();
    const parsedColor = ctx.fillStyle;

    // If it's a hex color, return it
    if (isValidHexColor(parsedColor)) {
      return parsedColor;
    }

    // If it's rgb/rgba, convert to hex
    const rgbMatch = parsedColor.match(
      /^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)$/,
    );
    if (rgbMatch) {
      const r = parseInt(rgbMatch[1]).toString(16).padStart(2, "0");
      const g = parseInt(rgbMatch[2]).toString(16).padStart(2, "0");
      const b = parseInt(rgbMatch[3]).toString(16).padStart(2, "0");
      return `#${r}${g}${b}`;
    }

    return null;
  };

  const handleColorBlur = () => {
    if (tempColorValue === value) return;

    const validColor = validateAndConvertColor(tempColorValue);
    if (validColor) {
      const oldValue = lastColorValueRef.current;
      onChange(validColor);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (lastColorValueRef.current === oldValue) {
            setTempColorValue(lastColorValueRef.current);
          }
        });
      });
    } else {
      setTempColorValue(value);
    }
  };

  const handleColorKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      const validColor = validateAndConvertColor(tempColorValue);
      if (tempColorValue !== value && validColor) {
        const oldValue = lastColorValueRef.current;
        onChange(validColor);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (lastColorValueRef.current === oldValue) {
              setTempColorValue(lastColorValueRef.current);
            }
          });
        });
      }
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setTempColorValue(value);
      e.currentTarget.blur();
    }
  };

  const handlePercentBlur = () => {
    const numericValue = tempPercentValue.replace(/[^0-9.]/g, "");
    const numValue = Number(numericValue);
    const currentDisplay = (percentValue ?? 100).toString();

    if (tempPercentValue === currentDisplay) return;

    if (
      !isNaN(numValue) &&
      isFinite(numValue) &&
      numValue >= 0 &&
      numValue <= 100
    ) {
      const oldValue = lastPercentValueRef.current;
      setPercentValue?.(numValue);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (lastPercentValueRef.current === oldValue) {
            setTempPercentValue(lastPercentValueRef.current.toString());
          }
        });
      });
    } else {
      setTempPercentValue(currentDisplay);
    }
  };

  const handlePercentKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      const numericValue = tempPercentValue.replace(/[^0-9.]/g, "");
      const numValue = Number(numericValue);
      const currentDisplay = (percentValue ?? 100).toString();

      if (
        tempPercentValue !== currentDisplay &&
        !isNaN(numValue) &&
        isFinite(numValue) &&
        numValue >= 0 &&
        numValue <= 100
      ) {
        const oldValue = lastPercentValueRef.current;
        setPercentValue?.(numValue);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (lastPercentValueRef.current === oldValue) {
              setTempPercentValue(lastPercentValueRef.current.toString());
            }
          });
        });
      }
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setTempPercentValue((percentValue ?? 100).toString());
      e.currentTarget.blur();
    }
  };

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0">
      {label && (
        <label
          className={cn(
            `text-brand-light text-[10.5px] font-medium w-full text-start mb-0.5`,
            labelClass,
          )}
        >
          {label}
        </label>
      )}
      <div className="flex flex-row relative w-full">
        <Popover>
          <PopoverTrigger>
            <span
              className="w-3 h-3 rounded absolute left-2 top-1/2 -translate-y-1/2"
              style={{ backgroundColor: tempColorValue }}
            />
          </PopoverTrigger>
          <input
            type="text"
            value={tempColorValue}
            onChange={(e) => setTempColorValue(e.target.value)}
            onBlur={handleColorBlur}
            onKeyDown={handleColorKeyDown}
            className={cn(
              "w-full px-1.5 text-brand-light text-[11px] font-normal items-center border border-brand-light/10 p-1 rounded-l bg-brand  pl-7",
              {
                "h-6": size === "small",
                "h-7": size === "medium",
              },
            )}
          />
          <input
            type="text"
            value={tempPercentValue}
            onChange={(e) => setTempPercentValue(e.target.value)}
            onBlur={handlePercentBlur}
            onKeyDown={handlePercentKeyDown}
            className={cn(
              "w-12 px-1.5 text-brand-light text-[11px]  font-normal items-center border border-brand-light/10 pl-2.5 rounded-r bg-brand border-l-0",
              {
                "h-6": size === "small",
                "h-7": size === "medium",
              },
            )}
          />
          <span className=" text-[11px] rounded absolute right-2 top-1/2 -translate-y-1/2 text-brand-light/50">
            %
          </span>
          <PopoverContent
            side="left"
            sideOffset={16}
            className="p-0 dark border-none w-60"
            onOpenAutoFocus={() => {
              isUserInteractingRef.current = true;
            }}
            onCloseAutoFocus={() => {
              isUserInteractingRef.current = false;
            }}
          >
            <ColorPicker
              value={Color(value)
                .alpha((percentValue ?? 100) / 100)
                .rgb()
                .array()}
              onChange={(value) => {
                if (!isUserInteractingRef.current) return;
                const color = Color(value);
                const newHex = color.hex();
                const newAlpha = Number((color.alpha() * 100).toFixed(0));

                if (
                  newHex !== lastColorValueRef.current ||
                  newAlpha !== lastPercentValueRef.current
                ) {
                  onChange(newHex);
                  setPercentValue?.(newAlpha);
                }
              }}
              className="w-60 rounded-md border p-3 text-brand-light bg-brand-background shadow-sm font-poppins"
            >
              <ColorPickerSelection className="h-48" />
              <div className="flex items-center gap-4">
                <ColorPickerEyeDropper />
                <div className="grid w-full gap-1">
                  <ColorPickerHue />
                  <ColorPickerAlpha />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <ColorPickerOutput />
                <ColorPickerFormat />
              </div>
            </ColorPicker>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  );
};

export default ColorInput;

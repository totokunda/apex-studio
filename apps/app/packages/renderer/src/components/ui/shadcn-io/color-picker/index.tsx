"use client";

import Color from "color";
import { PipetteIcon } from "lucide-react";
import { Slider } from "radix-ui";
import {
  type ComponentProps,
  createContext,
  type HTMLAttributes,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface ColorPickerContextValue {
  hue: number;
  saturation: number;
  lightness: number;
  alpha: number;
  mode: string;
  setHue: (hue: number) => void;
  setSaturation: (saturation: number) => void;
  setLightness: (lightness: number) => void;
  setAlpha: (alpha: number) => void;
  setMode: (mode: string) => void;
}

const ColorPickerContext = createContext<ColorPickerContextValue | undefined>(
  undefined,
);

export const useColorPicker = () => {
  const context = useContext(ColorPickerContext);

  if (!context) {
    throw new Error("useColorPicker must be used within a ColorPickerProvider");
  }

  return context;
};

export type ColorPickerProps = HTMLAttributes<HTMLDivElement> & {
  value?: Parameters<typeof Color>[0];
  defaultValue?: Parameters<typeof Color>[0];
  onChange?: (value: Parameters<typeof Color.rgb>[0]) => void;
};

export const ColorPicker = ({
  value,
  defaultValue = "#000000",
  onChange,
  className,
  ...props
}: ColorPickerProps) => {
  const selectedColor = value ? Color(value) : null;
  const defaultColor = Color(defaultValue);

  const [hue, setHue] = useState(
    selectedColor ? selectedColor.hue() : defaultColor.hue(),
  );
  const [saturation, setSaturation] = useState(
    selectedColor ? selectedColor.saturationl() : defaultColor.saturationl(),
  );
  const [lightness, setLightness] = useState(
    selectedColor ? selectedColor.lightness() : defaultColor.lightness(),
  );
  const [alpha, setAlpha] = useState(
    selectedColor ? selectedColor.alpha() * 100 : defaultColor.alpha() * 100,
  );
  const [mode, setMode] = useState("hex");

  // Update color when controlled value changes
  useEffect(() => {
    if (value) {
      const color = Color(value);
      const hslColor = color.hsl();
      setHue(hslColor.hue() || 0);
      setSaturation(hslColor.saturationl());
      setLightness(hslColor.lightness());
      setAlpha(color.alpha() * 100);
    }
  }, [value]);

  // Notify parent of changes
  useEffect(() => {
    if (onChange) {
      const color = Color.hsl(hue, saturation, lightness).alpha(alpha / 100);
      const rgba = color.rgb().array();

      onChange([rgba[0], rgba[1], rgba[2], alpha / 100]);
    }
  }, [hue, saturation, lightness, alpha, onChange]);

  return (
    <ColorPickerContext.Provider
      value={{
        hue,
        saturation,
        lightness,
        alpha,
        mode,
        setHue,
        setSaturation,
        setLightness,
        setAlpha,
        setMode,
      }}
    >
      <div
        className={cn("flex size-full flex-col gap-4", className)}
        {...props}
      />
    </ColorPickerContext.Provider>
  );
};

export type ColorPickerSelectionProps = HTMLAttributes<HTMLDivElement>;

export const ColorPickerSelection = memo(
  ({ className, ...props }: ColorPickerSelectionProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [positionX, setPositionX] = useState(0);
    const [positionY, setPositionY] = useState(0);
    const { hue, saturation, lightness, setSaturation, setLightness } =
      useColorPicker();

    // Initialize position from current saturation and lightness
    useEffect(() => {
      const x = saturation / 100;
      const topLightness = x < 0.01 ? 100 : 50 + 50 * (1 - x);
      const y = topLightness > 0 ? 1 - lightness / topLightness : 0;
      setPositionX(x);
      setPositionY(y);
    }, [saturation, lightness]);

    const backgroundGradient = useMemo(() => {
      return `linear-gradient(0deg, rgba(0,0,0,1), rgba(0,0,0,0)),
            linear-gradient(90deg, rgba(255,255,255,1), rgba(255,255,255,0)),
            hsl(${hue}, 100%, 50%)`;
    }, [hue]);

    const handlePointerMove = useCallback(
      (event: PointerEvent) => {
        if (!(isDragging && containerRef.current)) {
          return;
        }
        const rect = containerRef.current.getBoundingClientRect();
        const x = Math.max(
          0,
          Math.min(1, (event.clientX - rect.left) / rect.width),
        );
        const y = Math.max(
          0,
          Math.min(1, (event.clientY - rect.top) / rect.height),
        );
        setPositionX(x);
        setPositionY(y);
        setSaturation(x * 100);
        const topLightness = x < 0.01 ? 100 : 50 + 50 * (1 - x);
        const lightness = topLightness * (1 - y);

        setLightness(lightness);
      },
      [isDragging, setSaturation, setLightness],
    );

    useEffect(() => {
      const handlePointerUp = () => setIsDragging(false);

      if (isDragging) {
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerUp);
      }

      return () => {
        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", handlePointerUp);
      };
    }, [isDragging, handlePointerMove]);

    return (
      <div
        className={cn("relative size-full cursor-crosshair rounded", className)}
        onPointerDown={(e) => {
          e.preventDefault();
          setIsDragging(true);
          handlePointerMove(e.nativeEvent);
        }}
        ref={containerRef}
        style={{
          background: backgroundGradient,
        }}
        {...props}
      >
        <div
          className="-translate-x-1/2 -translate-y-1/2 pointer-events-none absolute h-4 w-4 rounded-full border-2 border-white"
          style={{
            left: `${positionX * 100}%`,
            top: `${positionY * 100}%`,
            boxShadow: "0 0 0 1px rgba(0,0,0,0.5)",
          }}
        />
      </div>
    );
  },
);

ColorPickerSelection.displayName = "ColorPickerSelection";

export type ColorPickerHueProps = ComponentProps<typeof Slider.Root>;

export const ColorPickerHue = ({
  className,
  ...props
}: ColorPickerHueProps) => {
  const { hue, setHue } = useColorPicker();

  return (
    <Slider.Root
      className={cn("relative flex h-4 w-full touch-none", className)}
      max={360}
      onValueChange={([hue]) => setHue(hue)}
      step={1}
      value={[hue]}
      {...props}
    >
      <Slider.Track className="relative my-1 h-2 w-full grow rounded-full bg-[linear-gradient(90deg,#FF0000,#FFFF00,#00FF00,#00FFFF,#0000FF,#FF00FF,#FF0000)]">
        <Slider.Range className="absolute h-full" />
      </Slider.Track>

      <Slider.Thumb className="block h-4 w-4 rounded-full border-4 border-brand-lighter bg-transparent shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-5" />
    </Slider.Root>
  );
};

export type ColorPickerAlphaProps = ComponentProps<typeof Slider.Root>;

export const ColorPickerAlpha = ({
  className,
  ...props
}: ColorPickerAlphaProps) => {
  const { alpha, setAlpha } = useColorPicker();

  return (
    <Slider.Root
      className={cn("relative flex h-4 w-full touch-none", className)}
      max={100}
      onValueChange={([alpha]) => setAlpha(alpha)}
      step={1}
      value={[alpha]}
      {...props}
    >
      <Slider.Track
        className="relative my-1 h-2 w-full grow rounded-full border border-brand-light/10"
        style={{
          background:
            'url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAMUlEQVQ4T2NkYGAQYcAP3uCTZhw1gGGYhAGBZIA/nYDCgBDAm9BGDWAAJyRCgLaBCAAgXwixzAS0pgAAAABJRU5ErkJggg==") left center',
        }}
      >
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-transparent to-black/50" />
        <Slider.Range className="absolute h-full rounded-full bg-transparent" />
      </Slider.Track>
      <Slider.Thumb className="block h-4 w-4 rounded-full border-4 border-brand-lighter bg-transparent shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50" />
    </Slider.Root>
  );
};

export type ColorPickerEyeDropperProps = ComponentProps<typeof Button>;

export const ColorPickerEyeDropper = ({
  className,
  ...props
}: ColorPickerEyeDropperProps) => {
  const { setHue, setSaturation, setLightness, setAlpha } = useColorPicker();

  const handleEyeDropper = async () => {
    try {
      // @ts-expect-error - EyeDropper API is experimental
      const eyeDropper = new EyeDropper();
      const result = await eyeDropper.open();
      const color = Color(result.sRGBHex);
      const [h, s, l] = color.hsl().array();

      setHue(h);
      setSaturation(s);
      setLightness(l);
      setAlpha(100);
    } catch (error) {
      console.error("EyeDropper failed:", error);
    }
  };

  return (
    <Button
      className={cn(
        "shrink-0 text-muted-foreground bg-brand h-8 w-8 border-brand-light/10 p-1",
        className,
      )}
      onClick={handleEyeDropper}
      size="icon"
      variant="outline"
      type="button"
      {...props}
    >
      <PipetteIcon size={16} />
    </Button>
  );
};

export type ColorPickerOutputProps = ComponentProps<typeof SelectTrigger>;

const formats = ["hex", "rgb", "hsl"];

export const ColorPickerOutput = ({
  className,
  ...props
}: ColorPickerOutputProps) => {
  const { mode, setMode } = useColorPicker();

  return (
    <Select onValueChange={setMode} value={mode}>
      <SelectTrigger
        className="!h-7 rounded w-16 shrink-0 text-[10px]"
        size="sm"
        {...props}
      >
        <SelectValue placeholder="Mode" />
      </SelectTrigger>
      <SelectContent className="dark">
        {formats.map((format) => (
          <SelectItem className="text-[10px]" key={format} value={format}>
            {format.toUpperCase()}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

type PercentageInputProps = Omit<
  ComponentProps<typeof Input>,
  "value" | "onChange"
> & {
  value: number;
  onValueChange?: (value: number) => void;
};

const PercentageInput = ({
  className,
  value,
  onValueChange,
  ...props
}: PercentageInputProps) => {
  const [tempValue, setTempValue] = useState(Math.round(value).toString());

  useEffect(() => {
    setTempValue(Math.round(value).toString());
  }, [value]);

  const validateAndApply = (inputValue: string) => {
    const numValue = parseInt(inputValue, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 100) {
      onValueChange?.(numValue);
    } else {
      setTempValue(Math.round(value).toString());
    }
  };

  return (
    <div className="relative">
      <Input
        type="text"
        value={tempValue}
        onChange={(e) => setTempValue(e.target.value)}
        onBlur={() => validateAndApply(tempValue)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            validateAndApply(tempValue);
            e.currentTarget.blur();
          } else if (e.key === "Escape") {
            setTempValue(Math.round(value).toString());
            e.currentTarget.blur();
          }
        }}
        {...props}
        className={cn(
          "h-7 w-[3.25rem] rounded-l-none bg-brand px-2 text-xs shadow-none text-brand-light border border-brand-light/10",
          className,
        )}
      />
      <span className="-translate-y-1/2 absolute top-1/2 right-2 text-muted-foreground text-xs pointer-events-none">
        %
      </span>
    </div>
  );
};

export type ColorPickerFormatProps = HTMLAttributes<HTMLDivElement>;

export const ColorPickerFormat = ({
  className,
  ...props
}: ColorPickerFormatProps) => {
  const {
    hue,
    saturation,
    lightness,
    alpha,
    mode,
    setHue,
    setSaturation,
    setLightness,
    setAlpha,
  } = useColorPicker();
  const color = Color.hsl(hue, saturation, lightness, alpha / 100);

  const [hexInput, setHexInput] = useState("");
  const [rgbInputs, setRgbInputs] = useState(["", "", ""]);
  const [hslInputs, setHslInputs] = useState(["", "", ""]);

  useEffect(() => {
    setHexInput(color.hex());
  }, [hue, saturation, lightness, alpha, mode]);

  useEffect(() => {
    const rgb = color
      .rgb()
      .array()
      .map((value) => Math.round(value))
      .slice(0, 3);
    setRgbInputs(rgb.map(String));
  }, [hue, saturation, lightness, alpha, mode]);

  useEffect(() => {
    const hsl = color
      .hsl()
      .array()
      .map((value) => Math.round(value))
      .slice(0, 3);
    setHslInputs(hsl.map(String));
  }, [hue, saturation, lightness, alpha, mode]);

  const validateAndApplyHex = (value: string) => {
    try {
      const trimmed = value.trim();
      const hexWithHash = trimmed.startsWith("#") ? trimmed : `#${trimmed}`;
      if (/^#([0-9A-F]{3}){1,2}$/i.test(hexWithHash)) {
        const newColor = Color(hexWithHash);
        setHue(newColor.hue() || 0);
        setSaturation(newColor.saturationl());
        setLightness(newColor.lightness());
      } else {
        setHexInput(color.hex());
      }
    } catch {
      setHexInput(color.hex());
    }
  };

  const validateAndApplyRgb = (values: string[]) => {
    try {
      const nums = values.map((v) => {
        const num = parseInt(v, 10);
        return isNaN(num) ? -1 : Math.max(0, Math.min(255, num));
      });

      if (nums.every((n) => n >= 0)) {
        const newColor = Color.rgb(nums[0], nums[1], nums[2]);
        setHue(newColor.hue() || 0);
        setSaturation(newColor.saturationl());
        setLightness(newColor.lightness());
      } else {
        const rgb = color
          .rgb()
          .array()
          .map((value) => Math.round(value))
          .slice(0, 3);
        setRgbInputs(rgb.map(String));
      }
    } catch {
      const rgb = color
        .rgb()
        .array()
        .map((value) => Math.round(value))
        .slice(0, 3);
      setRgbInputs(rgb.map(String));
    }
  };

  const validateAndApplyHsl = (values: string[]) => {
    try {
      const h = parseInt(values[0], 10);
      const s = parseInt(values[1], 10);
      const l = parseInt(values[2], 10);

      if (
        !isNaN(h) &&
        !isNaN(s) &&
        !isNaN(l) &&
        h >= 0 &&
        h <= 360 &&
        s >= 0 &&
        s <= 100 &&
        l >= 0 &&
        l <= 100
      ) {
        setHue(h);
        setSaturation(s);
        setLightness(l);
      } else {
        const hsl = color
          .hsl()
          .array()
          .map((value) => Math.round(value))
          .slice(0, 3);
        setHslInputs(hsl.map(String));
      }
    } catch {
      const hsl = color
        .hsl()
        .array()
        .map((value) => Math.round(value))
        .slice(0, 3);
      setHslInputs(hsl.map(String));
    }
  };

  if (mode === "hex") {
    return (
      <div
        className={cn(
          "-space-x-px relative flex w-full items-center rounded shadow-sm",
          className,
        )}
        {...props}
      >
        <Input
          className="h-7 rounded-r-none bg-brand px-2 !text-[10px] shadow-none text-brand-light border border-brand-light/10 rounded-l"
          type="text"
          value={hexInput}
          onChange={(e) => setHexInput(e.target.value)}
          onBlur={() => validateAndApplyHex(hexInput)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              validateAndApplyHex(hexInput);
              e.currentTarget.blur();
            } else if (e.key === "Escape") {
              setHexInput(color.hex());
              e.currentTarget.blur();
            }
          }}
        />
        <PercentageInput
          className="!text-[10px] rounded-r"
          value={alpha}
          onValueChange={setAlpha}
        />
      </div>
    );
  }

  if (mode === "rgb") {
    return (
      <div
        className={cn(
          "-space-x-px flex items-center rounded-md shadow-sm",
          className,
        )}
        {...props}
      >
        {rgbInputs.map((value, index) => (
          <Input
            className={cn(
              "h-7 rounded-r-none bg-secondary !text-[10px] px-1 text-center text-xs shadow-none rounded-l",
              index && "rounded-l-none",
              className,
            )}
            key={index}
            type="text"
            value={value}
            onChange={(e) => {
              const newInputs = [...rgbInputs];
              newInputs[index] = e.target.value;
              setRgbInputs(newInputs);
            }}
            onBlur={() => validateAndApplyRgb(rgbInputs)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                validateAndApplyRgb(rgbInputs);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                const rgb = color
                  .rgb()
                  .array()
                  .map((value) => Math.round(value))
                  .slice(0, 3);
                setRgbInputs(rgb.map(String));
                e.currentTarget.blur();
              }
            }}
          />
        ))}
        <PercentageInput
          className="!text-[10px] rounded-r"
          value={alpha}
          onValueChange={setAlpha}
        />
      </div>
    );
  }

  if (mode === "hsl") {
    return (
      <div
        className={cn(
          "-space-x-px flex items-center rounded shadow-sm",
          className,
        )}
        {...props}
      >
        {hslInputs.map((value, index) => (
          <Input
            className={cn(
              "h-7 rounded-r-none bg-secondary !text-[10px] px-1 text-center text-xs shadow-none rounded-l",
              index && "rounded-l-none",
              className,
            )}
            key={index}
            type="text"
            value={value}
            onChange={(e) => {
              const newInputs = [...hslInputs];
              newInputs[index] = e.target.value;
              setHslInputs(newInputs);
            }}
            onBlur={() => validateAndApplyHsl(hslInputs)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                validateAndApplyHsl(hslInputs);
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                const hsl = color
                  .hsl()
                  .array()
                  .map((value) => Math.round(value))
                  .slice(0, 3);
                setHslInputs(hsl.map(String));
                e.currentTarget.blur();
              }
            }}
          />
        ))}
        <PercentageInput
          className="!text-[10px] rounded-r"
          value={alpha}
          onValueChange={setAlpha}
        />
      </div>
    );
  }

  return null;
};

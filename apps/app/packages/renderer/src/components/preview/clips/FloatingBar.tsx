import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LuChevronDown,
  LuChevronUp,
  LuMousePointer2,
  LuHand,
  LuCheck,
  LuSquare,
  LuCircle,
  LuTriangle,
  LuMinus,
  LuStar,
  LuType,
  LuHighlighter,
  LuEraser,
  LuPenTool,
  LuLasso,
  LuBrush,
  LuWand,
  LuPlus,
  LuX,
  LuGripHorizontal,
} from "react-icons/lu";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { useViewportStore } from "@/lib/viewport";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useDrawingStore } from "@/lib/drawing";
import { useMaskStore } from "@/lib/mask";
import ColorInput from "@/components/properties/ColorInput";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface FloatingBarProps {}

const FLOATING_BAR_STORAGE_KEY = "preview-floating-bar-position-v1";
const FLOATING_BAR_MARGIN = 18;

const TextButton = ({
  active,
  onClick,
}: {
  active: boolean;
  onClick: () => void;
}) => {
  return (
    <button
      type="button"
      aria-label="Text tool"
      onClick={onClick}
      className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${active ? "text-brand bg-brand-light" : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"}`}
    >
      <LuType className="w-5 h-5" />
    </button>
  );
};

const SmallTooltip = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactElement;
}) => (
  <Tooltip>
    <TooltipTrigger asChild>{children}</TooltipTrigger>
    <TooltipContent side="bottom" sideOffset={6} className="px-2 py-1 text-[10px]">
      {label}
    </TooltipContent>
  </Tooltip>
);

// Removed individual ShapeButton in favor of a dropdown selector for shapes

const FloatingBar: React.FC<FloatingBarProps> = () => {
  const barRef = useRef<HTMLDivElement>(null);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const latestBarPositionRef = useRef({ x: 24, y: 24 });
  const scale = useViewportStore((s) => s.scale);
  const minScale = useViewportStore((s) => s.minScale);
  const maxScale = useViewportStore((s) => s.maxScale);
  const tool = useViewportStore((s) => s.tool);
  const shape = useViewportStore((s) => s.shape);
  const setTool = useViewportStore((s) => s.setTool);
  const setShape = useViewportStore((s) => s.setShape);
  const setScalePercent = useViewportStore((s) => s.setScalePercent);
  const zoomToFit = useViewportStore((s) => s.zoomToFit);

  const drawingTool = useDrawingStore((s) => s.tool);
  const setDrawingTool = useDrawingStore((s) => s.setTool);
  const brushSize = useDrawingStore((s) => s.brushSize);
  const highlighterSize = useDrawingStore((s) => s.highlighterSize);
  const eraserSize = useDrawingStore((s) => s.eraserSize);
  const smoothing = useDrawingStore((s) => s.smoothing);
  const setBrushSize = useDrawingStore((s) => s.setBrushSize);
  const setHighlighterSize = useDrawingStore((s) => s.setHighlighterSize);
  const setEraserSize = useDrawingStore((s) => s.setEraserSize);
  const setSmoothing = useDrawingStore((s) => s.setSmoothing);
  const drawingColor = useDrawingStore((s) => s.color);
  const drawingOpacity = useDrawingStore((s) => s.opacity);
  const setDrawingColor = useDrawingStore((s) => s.setColor);
  const setDrawingOpacity = useDrawingStore((s) => s.setOpacity);

  const getCurrentSize = () => {
    switch (drawingTool) {
      case "brush":
        return brushSize;
      case "highlighter":
        return highlighterSize;
      case "eraser":
        return eraserSize;
      default:
        return brushSize;
    }
  };

  const [zoomLevel, setZoomLevel] = useState<number>(Math.round(scale * 100));
  const [zoomOpen, setZoomOpen] = useState(false);

  const [shapeOpen, setShapeOpen] = useState(false);
  const [toolOpen, setToolOpen] = useState(false);
  const [drawOpen, setDrawOpen] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [tempSizeValue, setTempSizeValue] = useState<string>("3");
  const [isDragging, setIsDragging] = useState(false);
  const [barPosition, setBarPosition] = useState({ x: 24, y: 24 });

  const maskTool = useMaskStore((s) => s.tool);
  const setMaskTool = useMaskStore((s) => s.setTool);
  const maskShape = useMaskStore((s) => s.shape);
  const setMaskShape = useMaskStore((s) => s.setShape);
  const maskBrushSize = useMaskStore((s) => s.brushSize);
  const setMaskBrushSize = useMaskStore((s) => s.setBrushSize);
  const touchLabel = useMaskStore((s) => s.touchLabel);
  const setTouchLabel = useMaskStore((s) => s.setTouchLabel);

  useEffect(() => {
    setZoomLevel(Math.round(scale * 100));
  }, [scale]);

  useEffect(() => {
    setTempSizeValue(getCurrentSize().toString());
  }, [drawingTool, brushSize, highlighterSize, eraserSize, getCurrentSize]);
  const sliderMin = useMemo(() => Math.round(minScale * 100), [minScale]);
  const sliderMax = useMemo(() => Math.round(maxScale * 100), [maxScale]);

  const clampBarPosition = useCallback((x: number, y: number) => {
    const width = barRef.current?.offsetWidth ?? 320;
    const height = barRef.current?.offsetHeight ?? 64;
    const maxX = Math.max(FLOATING_BAR_MARGIN, window.innerWidth - width - FLOATING_BAR_MARGIN);
    const maxY = Math.max(FLOATING_BAR_MARGIN, window.innerHeight - height - FLOATING_BAR_MARGIN);
    return {
      x: Math.min(Math.max(x, FLOATING_BAR_MARGIN), maxX),
      y: Math.min(Math.max(y, FLOATING_BAR_MARGIN), maxY),
    };
  }, []);

  const getDockedPosition = useCallback(
    (dock: "top-right" | "top-left" | "bottom-center") => {
      const width = barRef.current?.offsetWidth ?? 320;
      const height = barRef.current?.offsetHeight ?? 64;
      switch (dock) {
        case "top-left":
          return clampBarPosition(FLOATING_BAR_MARGIN, 24);
        case "bottom-center":
          return clampBarPosition(
            Math.round((window.innerWidth - width) / 2),
            window.innerHeight - height - FLOATING_BAR_MARGIN,
          );
        case "top-right":
        default:
          return clampBarPosition(Math.round((window.innerWidth - width) / 2), 24);
      }
    },
    [clampBarPosition],
  );

  useEffect(() => {
    const applyStoredPosition = () => {
      const raw = window.localStorage.getItem(FLOATING_BAR_STORAGE_KEY);
      if (!raw) {
        setBarPosition(getDockedPosition("top-right"));
        return;
      }
      try {
        const parsed = JSON.parse(raw) as { x?: number; y?: number };
        if (typeof parsed.x === "number" && typeof parsed.y === "number") {
          setBarPosition(clampBarPosition(parsed.x, parsed.y));
          return;
        }
      } catch {
        // ignore malformed local storage and fall back to default dock
      }
      setBarPosition(getDockedPosition("top-right"));
    };

    applyStoredPosition();
    const onResize = () => setBarPosition((prev) => clampBarPosition(prev.x, prev.y));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clampBarPosition, getDockedPosition]);

  useEffect(() => {
    latestBarPositionRef.current = barPosition;
  }, [barPosition]);

  useEffect(() => {
    if (!isDragging) return;
    const onPointerMove = (event: PointerEvent) => {
      const next = clampBarPosition(
        event.clientX - dragOffsetRef.current.x,
        event.clientY - dragOffsetRef.current.y,
      );
      setBarPosition(next);
    };
    const onPointerUp = () => {
      setIsDragging(false);
      window.localStorage.setItem(
        FLOATING_BAR_STORAGE_KEY,
        JSON.stringify(latestBarPositionRef.current),
      );
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [isDragging, clampBarPosition]);

  const startDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) return;
      dragOffsetRef.current = {
        x: event.clientX - barPosition.x,
        y: event.clientY - barPosition.y,
      };
      setIsDragging(true);
      event.preventDefault();
    },
    [barPosition],
  );

  const dockFloatingBar = useCallback(
    (dock: "top-right" | "top-left" | "bottom-center") => {
      const next = getDockedPosition(dock);
      setBarPosition(next);
      window.localStorage.setItem(FLOATING_BAR_STORAGE_KEY, JSON.stringify(next));
    },
    [getDockedPosition],
  );

  return (
    <div
      ref={barRef}
      className="w-64 fixed rounded-lg px-6 z-[120]"
      style={{ left: `${barPosition.x}px`, top: `${barPosition.y}px` }}
    >
      <div className="w-full h-full flex items-center justify-between">
        <div className="relative flex flex-row-reverse justify-center items-center gap-x-1.5 p-2 bg-brand border border-brand-light/5 rounded-lg shadow-lg">
          <div className="absolute -left-7 flex flex-col gap-y-1">
            <button
              type="button"
              onPointerDown={startDrag}
              className={`h-6 w-6 rounded-md border border-brand-light/10 bg-brand text-brand-light/80 hover:bg-brand-light/15 hover:text-brand-light transition-all duration-200 cursor-grab active:cursor-grabbing flex items-center justify-center ${isDragging ? "ring-2 ring-brand-light/50" : ""}`}
              aria-label="Drag toolbar"
            >
              <LuGripHorizontal className="h-3.5 w-3.5" />
            </button>
            <div className="flex flex-col gap-y-1 rounded-md border border-brand-light/10 bg-brand p-1">
              <button
                type="button"
                aria-label="Dock toolbar top center"
                onClick={() => dockFloatingBar("top-right")}
                className="h-1.5 w-4 self-center rounded-full bg-brand-light/70 hover:bg-brand-light transition-colors"
              />
              <button
                type="button"
                aria-label="Dock toolbar top left"
                onClick={() => dockFloatingBar("top-left")}
                className="h-1.5 w-4 self-center rounded-full bg-brand-light/50 hover:bg-brand-light transition-colors"
              />
              <button
                type="button"
                aria-label="Dock toolbar bottom center"
                onClick={() => dockFloatingBar("bottom-center")}
                className="h-1.5 w-4 self-center rounded-full bg-brand-light/35 hover:bg-brand-light transition-colors"
              />
            </div>
          </div>
          <div className="flex items-center gap-x-1">
            <DropdownMenu open={zoomOpen} onOpenChange={setZoomOpen}>
              <SmallTooltip label="Zoom">
                <DropdownMenuTrigger className="text-brand-light/85 dark w-18  flex items-center font-medium justify-between px-2 text-xs border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/15 rounded-md py-[7px] transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60">
                  {zoomLevel}%
                  {zoomOpen ? (
                    <LuChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <LuChevronDown className="w-3.5 h-3.5" />
                  )}
                </DropdownMenuTrigger>
              </SmallTooltip>
              <DropdownMenuContent className="dark w-60 bg-brand-background font-inter z-[140]">
                <DropdownMenuLabel className="flex flex-col justify-center py-2 pb-0 px-1.5">
                  <span className="text-brand-light text-xs font-medium">
                    Size
                  </span>
                  <div className="flex flex-row items-center gap-x-2 w-full mb-1">
                    <Slider
                      className="w-full dark"
                      value={[zoomLevel]}
                      max={sliderMax}
                      min={sliderMin}
                      step={1}
                      onValueChange={(value) => {
                        setZoomLevel(value[0]);
                        setScalePercent(value[0]);
                      }}
                    />
                    <input
                      className="w-[42px] h-7 border border-brand-light/10 px-1 bg-brand  text-brand-light focus:outline-brand-background text-center ring-brand-background  text-[11px] font-light items-center justify-center rounded-sm "
                      value={`${zoomLevel}%`}
                      onChange={(e) => {
                        const raw = e.target.value.replace(/[^0-9]/g, "");
                        const num = Math.max(
                          sliderMin,
                          Math.min(sliderMax, Math.abs(parseInt(raw || "0"))),
                        );
                        if (!Number.isNaN(num)) setZoomLevel(num);
                      }}
                      onBlur={() => setScalePercent(zoomLevel)}
                    />
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  key="zoom-to-fit"
                  textValue="Zoom to fit"
                  className="dark text-[12px] font-medium"
                  onClick={() => {
                    zoomToFit();
                    setZoomOpen(false);
                  }}
                >
                  Zoom to Fit
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="zoom-to-50"
                  textValue="Zoom to 50%"
                  className="dark text-[12px] font-medium"
                  onClick={() => {
                    setScalePercent(50);
                    setZoomOpen(false);
                  }}
                >
                  Zoom to 50%
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="zoom-to-100"
                  textValue="Zoom to 100%"
                  className="dark text-[12px] font-medium"
                  onClick={() => {
                    setScalePercent(100);
                    setZoomOpen(false);
                  }}
                >
                  Zoom to 100%
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="zoom-to-200"
                  textValue="Zoom to 200%"
                  className="dark text-[12px] font-medium"
                  onClick={() => {
                    setScalePercent(200);
                    setZoomOpen(false);
                  }}
                >
                  Zoom to 200%
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div className="flex items-center gap-x-1">
            <SmallTooltip label="Draw tool">
              <button
                type="button"
                aria-label="Draw tool"
                onClick={() => setTool("draw")}
                className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${tool === "draw" ? "text-brand bg-brand-light" : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"}`}
              >
                {drawingTool === "brush" && <LuPenTool className="w-5 h-5" />}
                {drawingTool === "highlighter" && (
                  <LuHighlighter className="w-5 h-5" />
                )}
                {drawingTool === "eraser" && <LuEraser className="w-5 h-5" />}
              </button>
            </SmallTooltip>
            <Popover open={drawOpen} onOpenChange={setDrawOpen}>
              <SmallTooltip label="Draw options">
                <PopoverTrigger
                  className={`rounded h-8 px-0.5 -ml-0.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                    drawOpen
                      ? "text-brand-light bg-brand-light/15"
                      : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                  }`}
                >
                  {drawOpen ? (
                    <LuChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <LuChevronDown className="w-3.5 h-3.5" />
                  )}
                </PopoverTrigger>
              </SmallTooltip>
              <PopoverContent className="dark w-56 bg-brand-background font-inter p-3 z-[140]">
                <div className="flex flex-col gap-y-3">
                  {/* Drawing Tools */}
                  <div className="flex flex-col gap-y-0.5">
                    <label className="text-brand-light text-[11px] mb-1 font-medium">
                      Tool
                    </label>
                    <div className="flex flex-row border divide-x border-brand-light/10 rounded-md overflow-hidden bg-brand">
                      <div
                        onClick={() => {
                          setTool("draw");
                          setDrawingTool("brush");
                        }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 cursor-pointer justify-center transition-all duration-200 ${
                          drawingTool === "brush"
                            ? "bg-brand-light text-brand"
                            : "text-brand-light/80 hover:bg-brand-light/5"
                        }`}
                      >
                        <LuPenTool className="w-4 h-4" />
                      </div>
                      <div
                        onClick={() => {
                          setTool("draw");
                          setDrawingTool("highlighter");
                        }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 justify-center cursor-pointer transition-all duration-200 ${
                          drawingTool === "highlighter"
                            ? "bg-brand-light text-brand"
                            : "text-brand-light/80 hover:bg-brand-light/5"
                        }`}
                      >
                        <LuHighlighter className="w-4 h-4" />
                      </div>
                      <div
                        onClick={() => {
                          setTool("draw");
                          setDrawingTool("eraser");
                        }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 justify-center cursor-pointer transition-all duration-200 ${
                          drawingTool === "eraser"
                            ? "bg-brand-light text-brand"
                            : "text-brand-light/80 hover:bg-brand-light/5"
                        }`}
                      >
                        <LuEraser className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                  {/* Size Slider */}
                  <div className="flex flex-col gap-y-0.5">
                    <label className="text-brand-light text-[12px] font-medium">
                      Size
                    </label>
                    <div className="flex items-center gap-x-2">
                      <Slider
                        className="w-full dark"
                        value={[getCurrentSize()]}
                        max={
                          drawingTool === "highlighter"
                            ? 50
                            : drawingTool === "eraser"
                              ? 100
                              : 30
                        }
                        min={1}
                        step={1}
                        onValueChange={(value) => {
                          if (drawingTool === "brush") setBrushSize(value[0]);
                          else if (drawingTool === "highlighter")
                            setHighlighterSize(value[0]);
                          else if (drawingTool === "eraser")
                            setEraserSize(value[0]);
                        }}
                      />
                      <input
                        className="w-[40px] h-6 px-1 text-brand-light focus:outline-brand-background border border-brand-light/10 text-center bg-brand ring-brand-background text-[11px] font-normal items-center justify-center rounded"
                        value={tempSizeValue}
                        onChange={(e) => setTempSizeValue(e.target.value)}
                        onBlur={() => {
                          const raw = tempSizeValue.replace(/[^0-9]/g, "");
                          const maxVal =
                            drawingTool === "highlighter"
                              ? 50
                              : drawingTool === "eraser"
                                ? 100
                                : 20;
                          const num = Math.max(
                            1,
                            Math.min(maxVal, Math.abs(parseInt(raw || "1"))),
                          );
                          if (!Number.isNaN(num)) {
                            if (drawingTool === "brush") setBrushSize(num);
                            else if (drawingTool === "highlighter")
                              setHighlighterSize(num);
                            else if (drawingTool === "eraser")
                              setEraserSize(num);
                            setTempSizeValue(num.toString());
                          } else {
                            setTempSizeValue(getCurrentSize().toString());
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.currentTarget.blur();
                          } else if (e.key === "Escape") {
                            setTempSizeValue(getCurrentSize().toString());
                            e.currentTarget.blur();
                          }
                        }}
                      />
                    </div>
                  </div>
                  {/* Smoothing Slider (only for brush) */}
                  {drawingTool === "brush" && (
                    <div className="flex flex-col gap-y-0.5">
                      <label className="text-brand-light text-[11px] font-medium">
                        Smoothing
                      </label>
                      <div className="flex items-center gap-x-2">
                        <Slider
                          className="w-full dark"
                          value={[smoothing * 100]}
                          max={100}
                          min={0}
                          step={1}
                          onValueChange={(value) =>
                            setSmoothing(value[0] / 100)
                          }
                        />
                        <div className="w-[40px] h-6 px-1 text-brand-light border border-brand-light/10 text-center bg-brand text-[11px] font-normal flex items-center justify-center rounded">
                          {Math.round(smoothing * 100)}
                        </div>
                      </div>
                    </div>
                  )}
                  {/* Color Picker (hidden for eraser) */}
                  {drawingTool === "brush" && (
                    <ColorInput
                      size="medium"
                      label="Color"
                      labelClass="text-brand-light text-[11px] font-medium"
                      value={drawingColor}
                      onChange={setDrawingColor}
                      percentValue={drawingOpacity}
                      setPercentValue={setDrawingOpacity}
                    />
                  )}
                </div>
              </PopoverContent>
            </Popover>
            <SmallTooltip label="Text tool">
              <TextButton
                active={tool === "text"}
                onClick={() => setTool("text")}
              />
            </SmallTooltip>
            <SmallTooltip label="Mask tool">
              <button
                type="button"
                aria-label="Mask tool"
                onClick={() => setTool("mask")}
                className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${tool === "mask" ? "text-brand bg-brand-light" : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"}`}
              >
                {maskTool === "lasso" && <LuLasso className="w-5 h-5" />}
                {maskTool === "shape" && maskShape === "rectangle" && (
                  <LuSquare className="w-5 h-5" />
                )}
                {maskTool === "shape" && maskShape === "ellipse" && (
                  <LuCircle className="w-5 h-5" />
                )}
                {maskTool === "shape" && maskShape === "polygon" && (
                  <LuTriangle className="w-5 h-5" />
                )}
                {maskTool === "shape" && maskShape === "star" && (
                  <LuStar className="w-5 h-5" />
                )}
                {maskTool === "draw" && <LuBrush className="w-5 h-5" />}
                {maskTool === "touch" && <LuWand className="w-5 h-5" />}
              </button>
            </SmallTooltip>
            <Popover open={maskOpen} onOpenChange={setMaskOpen}>
              <SmallTooltip label="Mask options">
                <PopoverTrigger
                  className={`rounded h-8 px-0.5 -ml-0.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                    maskOpen
                      ? "text-brand-light bg-brand-light/15"
                      : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                  }`}
                >
                  {maskOpen ? (
                    <LuChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <LuChevronDown className="w-3.5 h-3.5" />
                  )}
                </PopoverTrigger>
              </SmallTooltip>
              <PopoverContent className="dark w-56 bg-brand-background font-inter p-3 pt-3 pb-5 z-[140]">
                <div className="mb-2.5 w-full flex ">
                  <label className="text-brand-light text-[11px] font-medium">
                    Masking Tool
                  </label>
                </div>
                <div className="flex flex-col gap-y-3">
                  {/* Mask Tools */}

                  <div className="flex flex-col gap-y-0.5">
                    <div className="grid grid-cols-3 gap-1.5">
                      <div
                        onClick={() => {
                          setTool("mask");
                          setMaskTool("touch");
                        }}
                        className={`flex flex-col relative items-center gap-y-0.5 px-1.5 py-1.5 cursor-pointer justify-center transition-all duration-200 rounded-sm border ${
                          maskTool === "touch" && tool === "mask"
                            ? "bg-brand-light border-brand-light text-brand"
                            : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                        }`}
                      >
                        <LuWand className="w-4 h-4" />
                        <div className="relative flex items-center gap-x-1">
                          <span className="text-[9px] font-medium">Touch</span>
                        </div>
                      </div>
                      <div
                        onClick={() => {
                          setTool("mask");
                          setMaskTool("lasso");
                        }}
                        className={`flex flex-col items-center gap-y-0.5 px-1.5 py-1.5 cursor-pointer justify-center transition-all duration-200 rounded-sm border ${
                          maskTool === "lasso" && tool === "mask"
                            ? "bg-brand-light border-brand-light text-brand"
                            : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                        }`}
                      >
                        <LuLasso className="w-4 h-4" />
                        <span className="text-[9px] font-medium">Lasso</span>
                      </div>
                      <div
                        onClick={() => {
                          setTool("mask");
                          setMaskTool("shape");
                        }}
                        className={`flex flex-col items-center gap-y-0.5 px-1.5 py-1.5 cursor-pointer justify-center transition-all duration-200 rounded-sm border ${
                          maskTool === "shape" && tool === "mask"
                            ? "bg-brand-light border-brand-light text-brand"
                            : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                        }`}
                      >
                        {maskShape === "rectangle" && (
                          <LuSquare className="w-4 h-4" />
                        )}
                        {maskShape === "ellipse" && (
                          <LuCircle className="w-4 h-4" />
                        )}
                        {maskShape === "polygon" && (
                          <LuTriangle className="w-4 h-4" />
                        )}
                        {maskShape === "star" && <LuStar className="w-4 h-4" />}
                        <span className="text-[9px] font-medium">Shape</span>
                      </div>
                    </div>
                  </div>

                  {/* Shape Selector (for shape tool) */}
                  {maskTool === "shape" && (
                    <div className="flex flex-col gap-y-2">
                      <label className="text-brand-light text-[11px] font-medium">
                        Mask Shape
                      </label>
                      <div className="grid grid-cols-4 gap-1">
                        <div
                          onClick={() => setMaskShape("rectangle")}
                          className={`flex items-center justify-center p-1.5 cursor-pointer transition-all duration-200 rounded-sm border ${
                            maskShape === "rectangle"
                              ? "bg-brand-light border-brand-light text-brand"
                              : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                          }`}
                        >
                          <LuSquare className="w-4 h-4" />
                        </div>
                        <div
                          onClick={() => setMaskShape("ellipse")}
                          className={`flex items-center justify-center p-1.5 cursor-pointer transition-all duration-200 rounded-sm border ${
                            maskShape === "ellipse"
                              ? "bg-brand-light border-brand-light text-brand"
                              : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                          }`}
                        >
                          <LuCircle className="w-4 h-4" />
                        </div>
                        <div
                          onClick={() => setMaskShape("polygon")}
                          className={`flex items-center justify-center p-1.5 cursor-pointer transition-all duration-200 rounded-sm border ${
                            maskShape === "polygon"
                              ? "bg-brand-light border-brand-light text-brand"
                              : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                          }`}
                        >
                          <LuTriangle className="w-4 h-4" />
                        </div>
                        <div
                          onClick={() => setMaskShape("star")}
                          className={`flex items-center justify-center p-1.5 cursor-pointer transition-all duration-200 rounded-sm border ${
                            maskShape === "star"
                              ? "bg-brand-light border-brand-light text-brand"
                              : "text-brand-light/80 bg-brand border-brand-light/10 hover:bg-brand-light/5"
                          }`}
                        >
                          <LuStar className="w-4 h-4" />
                        </div>
                      </div>
                    </div>
                  )}
                  {/* Brush Size (for draw tool) */}
                  {maskTool === "draw" && (
                    <div className="flex flex-col gap-y-0.5">
                      <label className="text-brand-light text-[12px] font-medium">
                        Brush Size
                      </label>
                      <div className="flex items-center gap-x-2">
                        <Slider
                          className="w-full dark"
                          value={[maskBrushSize]}
                          max={100}
                          min={1}
                          step={1}
                          onValueChange={(value) => setMaskBrushSize(value[0])}
                        />
                        <input
                          className="w-[40px] h-6 px-1 text-brand-light focus:outline-brand-background border border-brand-light/10 text-center bg-brand ring-brand-background text-[11px] font-normal items-center justify-center rounded"
                          value={maskBrushSize}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) || 1;
                            setMaskBrushSize(Math.max(1, Math.min(100, val)));
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Touch Tool Settings */}
                  {maskTool === "touch" && (
                    <>
                      <div className="flex flex-row items-center justify-between gap-x-2">
                        <label className="text-brand-light text-[11px] font-medium">
                          Point Type
                        </label>
                        <div className="flex flex-row gap-x-1 overflow-hidden w-22">
                          <div
                            onClick={() => setTouchLabel("positive")}
                            className={`flex items-center gap-x-2 px-3 py-1 w-1/2 rounded-md border border-brand-light/10 cursor-pointer justify-center transition-all duration-200 ${
                              touchLabel === "positive"
                                ? "bg-blue-500 text-brand-light"
                                : "text-brand-light/80 hover:bg-brand-light/5 bg-brand"
                            }`}
                          >
                            <div className="flex items-center justify-center w-4 h-4 rounded-md">
                              <LuPlus
                                className="w-2.5 h-2.5 text-white"
                                strokeWidth={3}
                              />
                            </div>
                          </div>
                          <div
                            onClick={() => setTouchLabel("negative")}
                            className={`flex items-center gap-x-2 px-3 py-1 w-1/2 rounded-md border border-brand-light/10 cursor-pointer justify-center transition-all duration-200 ${
                              touchLabel === "negative"
                                ? "bg-red-500 text-brand-light"
                                : "text-brand-light/80 hover:bg-brand-light/5 bg-brand"
                            }`}
                          >
                            <div className="flex items-center justify-center w-4 h-4 rounded-md">
                              <LuX
                                className="w-2.5 h-2.5 text-white"
                                strokeWidth={3}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </PopoverContent>
            </Popover>
            <SmallTooltip label="Shape tool">
              <button
                type="button"
                aria-label="Shape tool"
                onClick={() => setTool("shape")}
                className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                  tool === "shape"
                    ? "text-brand bg-brand-light"
                    : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                }`}
              >
                {shape === "rectangle" && <LuSquare className="w-5 h-5" />}
                {shape === "ellipse" && <LuCircle className="w-5 h-5" />}
                {shape === "polygon" && <LuTriangle className="w-5 h-5" />}
                {shape === "line" && <LuMinus className="w-5 h-5" />}
                {shape === "star" && <LuStar className="w-5 h-5" />}
              </button>
            </SmallTooltip>
            <DropdownMenu open={shapeOpen} onOpenChange={setShapeOpen}>
              <SmallTooltip label="Shape options">
                <DropdownMenuTrigger
                  className={`rounded h-8 px-0.5 -ml-0.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                    shapeOpen
                      ? "text-brand-light bg-brand-light/15"
                      : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                  }`}
                >
                  {shapeOpen ? (
                    <LuChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <LuChevronDown className="w-3.5 h-3.5" />
                  )}
                </DropdownMenuTrigger>
              </SmallTooltip>
              <DropdownMenuContent className="dark w-44 bg-brand-background font-inter z-[140]">
                <DropdownMenuItem
                  key="shape-rectangle"
                  textValue="Rectangle"
                  className="dark text-[11px] flex items-center gap-x-2 h-7 font-medium"
                  onClick={() => {
                    setShape("rectangle");
                    setTool("shape");
                    setShapeOpen(false);
                  }}
                >
                  <LuSquare className="w-4 h-4" />
                  <span>Rectangle</span>
                  {shape === "rectangle" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="shape-ellipse"
                  textValue="Ellipse"
                  className="dark text-[11px] flex items-center gap-x-2 h-7 font-medium"
                  onClick={() => {
                    setShape("ellipse");
                    setTool("shape");
                    setShapeOpen(false);
                  }}
                >
                  <LuCircle className="w-4 h-4" />
                  <span>Ellipse</span>
                  {shape === "ellipse" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="shape-polygon"
                  textValue="Polygon"
                  className="dark text-[11px] flex items-center gap-x-2 h-7 font-medium"
                  onClick={() => {
                    setShape("polygon");
                    setTool("shape");
                    setShapeOpen(false);
                  }}
                >
                  <LuTriangle className="w-4 h-4" />
                  <span>Polygon</span>
                  {shape === "polygon" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>
                <DropdownMenuItem
                  key="shape-star"
                  textValue="Star"
                  className="dark text-[11px] flex items-center gap-x-2 h-6 font-medium"
                  onClick={() => {
                    setShape("star");
                    setTool("shape");
                    setShapeOpen(false);
                  }}
                >
                  <LuStar className="w-4 h-4" />
                  <span>Star</span>
                  {shape === "star" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <SmallTooltip label="Select tool">
              <button
                type="button"
                aria-label="Select tool"
                onClick={() => setTool(tool === "hand" ? "hand" : "pointer")}
                className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                  tool === "pointer" || tool === "hand"
                    ? "text-brand bg-brand-light"
                    : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                }`}
              >
                {tool === "hand" ? (
                  <LuHand className="w-5 h-5" />
                ) : (
                  <LuMousePointer2 className="w-5 h-5" />
                )}
              </button>
            </SmallTooltip>
            <DropdownMenu open={toolOpen} onOpenChange={setToolOpen}>
              <SmallTooltip label="Select options">
                <DropdownMenuTrigger
                  className={`rounded py-2 px-0.5 -ml-0.5 h-8 transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-light/60 ${
                    toolOpen
                      ? "text-brand-light bg-brand-light/15"
                      : "text-brand-light/80 hover:text-brand-light hover:bg-brand-light/15"
                  }`}
                >
                  {toolOpen ? (
                    <LuChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <LuChevronDown className="w-3.5 h-3.5" />
                  )}
                </DropdownMenuTrigger>
              </SmallTooltip>
              <DropdownMenuContent className="dark w-36 bg-brand-background font-inter z-[140]">
                <DropdownMenuItem
                  key="tool-pointer"
                  textValue="Pointer"
                  className="dark text-[11px] flex items-center gap-x-2 h-7 font-medium"
                  onClick={() => {
                    setTool("pointer");
                    setToolOpen(false);
                  }}
                >
                  <LuMousePointer2 className="w-4 h-4" />
                  <span>Pointer</span>
                  {tool === "pointer" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>

                <DropdownMenuItem
                  key="tool-hand"
                  textValue="Hand"
                  className="dark text-[11px] flex items-center gap-x-2 h-7 font-medium"
                  onClick={() => {
                    setTool("hand");
                    setToolOpen(false);
                  }}
                >
                  <LuHand className="w-4 h-4" />
                  <span>Hand</span>
                  {tool === "hand" && (
                    <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                  )}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatingBar;

import { Layer } from "konva/lib/Layer.js";
import { Stage } from "konva/lib/Stage.js";
import { Group } from "konva/lib/Group.js";
import { Ellipse } from "konva/lib/shapes/Ellipse.js";
import { Line } from "konva/lib/shapes/Line.js";
import { Rect } from "konva/lib/shapes/Rect.js";
import { Star } from "konva/lib/shapes/Star.js";
import { Text } from "konva/lib/shapes/Text.js";
import { Shape } from "konva/lib/Shape.js";
import { applyWebGLFilters } from "./webgl-filters/apply";
import { applyMasksToCanvas } from "./masks/apply";
import type { WrappedCanvas } from "mediabunny";

import Konva from "konva";
//@ts-ignore
Konva._fixTextRendering = true;

type ShapeTool = "rectangle" | "ellipse" | "polygon" | "line" | "star";

export interface ClipTransform {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  cornerRadius: number;
  opacity: number;
  // Normalized crop in local content coordinates (0-1, relative to full media bounds)
  // Matches renderer `ClipTransform.crop` semantics used by `ImagePreview` / `VideoPreview`.
  crop?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

type NormalizedTransform = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

interface BaseClipProps {
  clipId: string;
  transform?: ClipTransform;
  /**
   * Optional normalized transform in [0, 1] space describing how this clip
   * should occupy the output canvas. When present, it will be mapped to
   * pixel coordinates using the target canvas size while leaving `transform`
   * as a pixel-space fallback.
   */
  normalizedTransform?: NormalizedTransform;
  type: "shape" | "text" | "draw";
}

export interface ShapeClipProps extends BaseClipProps {
  type: "shape";
  shapeType?: ShapeTool;
  fill?: string;
  fillOpacity?: number;
  stroke?: string;
  strokeOpacity?: number;
  strokeWidth?: number;
}

interface PolygonClipProps extends ShapeClipProps {
  shapeType: "polygon";
  sides?: number;
}

interface StarClipProps extends ShapeClipProps {
  shapeType: "star";
  points?: number;
}

export interface TextClipProps extends BaseClipProps {
  type: "text";
  text?: string;
  fontSize?: number;
  fontWeight?: number;
  fontStyle?: "normal" | "italic";
  fontFamily?: string;
  color?: string;
  colorOpacity?: number;
  textAlign?: "left" | "center" | "right";
  verticalAlign?: "top" | "middle" | "bottom";
  textTransform?: "none" | "uppercase" | "lowercase" | "capitalize";
  textDecoration?: "none" | "underline" | "overline" | "line-through";
  strokeEnabled?: boolean;
  stroke?: string;
  strokeWidth?: number;
  shadowEnabled?: boolean;
  shadowColor?: string;
  shadowOpacity?: number;
  shadowBlur?: number;
  shadowOffsetX?: number;
  shadowOffsetY?: number;
  backgroundEnabled?: boolean;
  backgroundColor?: string;
  backgroundOpacity?: number;
  backgroundCornerRadius?: number;
}
type MaskTransform = ClipTransform;

type MaskTool = "lasso" | "shape" | "draw" | "touch";
type MaskTrackingDirection = "forward" | "backward" | "both";

export type MaskClipProps = {
  id: string;
  clipId?: string;
  tool: MaskTool;
  featherAmount: number;
  brushSize?: number;
  // Mask data for the initial frame/keyframes
  keyframes: Map<number, any> | Record<number, any>;
  // Tracking settings
  isTracked: boolean;
  trackingDirection?: MaskTrackingDirection;
  confidenceThreshold?: number;
  // Transform applied to mask
  transform?: MaskTransform;
  // Metadata
  createdAt: number;
  lastModified: number;
  // Operation settings
  inverted?: boolean; // Invert the mask
  backgroundColor?: string;
  backgroundOpacity?: number;
  backgroundColorEnabled?: boolean;
  maskColor?: string;
  maskOpacity?: number;
  maskColorEnabled?: boolean;
  maxTrackingFrames?: number;
};

interface DrawingLineTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  opacity: number;
}

interface DrawingLine {
  lineId: string;
  tool: "brush" | "highlighter" | "eraser";
  points: number[];
  stroke: string;
  strokeWidth: number;
  opacity: number;
  smoothing: number;
  transform: DrawingLineTransform;
}

export interface DrawingClipProps extends BaseClipProps {
  type: "draw";
  lines: DrawingLine[];
}

type Canvas = HTMLCanvasElement;

function ensureTransform(
  transform?: ClipTransform,
  canvas?: Canvas,
): Required<Omit<ClipTransform, "crop">> & { crop?: ClipTransform["crop"] } {
  const defaultWidth = canvas?.width ?? 100;
  const defaultHeight = canvas?.height ?? 100;
  return {
    x: transform?.x ?? 0,
    y: transform?.y ?? 0,
    width: transform?.width ?? defaultWidth,
    height: transform?.height ?? defaultHeight,
    scaleX: transform?.scaleX ?? 1,
    scaleY: transform?.scaleY ?? 1,
    rotation: transform?.rotation ?? 0,
    cornerRadius: transform?.cornerRadius ?? 0,
    opacity: transform?.opacity ?? 100,
    crop: transform?.crop,
  };
}

function resolveTransformFromClip(
  transform: ClipTransform | undefined,
  normalizedTransform: NormalizedTransform | undefined,
  canvas?: Canvas,
): Required<Omit<ClipTransform, "crop">> & { crop?: ClipTransform["crop"] } {
  const base = ensureTransform(transform, canvas);
  if (!canvas || !normalizedTransform) return base;

  const cw = Math.max(1, canvas.width || 1);
  const ch = Math.max(1, canvas.height || 1);

  const toPx = (normVal: any, dim: number, fallback: number): number => {
    const n = Number(normVal);

    if (!Number.isFinite(n)) return fallback;
    // Allow values outside [0, 1] so off‑canvas normalized positions/sizes
    // are preserved when mapped back into pixel space.
    return n * dim;
  };

  const x = toPx(normalizedTransform.x, cw, base.x);
  const y = toPx(normalizedTransform.y, ch, base.y);
  const width = toPx(normalizedTransform.width, cw, base.width);
  const height = toPx(normalizedTransform.height, ch, base.height);

  return {
    ...base,
    x,
    y,
    width,
    height,
  };
}

function hexToRgba(hex: string, opacity: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, opacity / 100))})`;
}

function applyTextTransform(
  text: string,
  textTransform?: TextClipProps["textTransform"],
): string {
  if (!textTransform || textTransform === "none") return text;
  if (textTransform === "uppercase") return text.toUpperCase();
  if (textTransform === "lowercase") return text.toLowerCase();
  if (textTransform === "capitalize") {
    return text
      .split(" ")
      .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
      .join(" ");
  }
  return text;
}

function withStage<T>(canvas: Canvas, draw: (layer: Layer) => T): T {
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.left = "-10000px";
  container.style.top = "-10000px";
  container.style.width = `${canvas.width}px`;
  container.style.height = `${canvas.height}px`;
  document.body.appendChild(container);

  const stage = new Stage({
    container,
  });
  const layer = new Layer();
  stage.add(layer);

  try {
    const result = draw(layer);
    layer.draw();
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // Render using stage.toCanvas with pixelRatio 1 to avoid DPR-induced scaling
      const sceneCanvas = stage.toCanvas({
        pixelRatio: 1,
      }) as unknown as HTMLCanvasElement;
      ctx.drawImage(sceneCanvas, 0, 0, canvas.width, canvas.height);
    }
    return result;
  } finally {
    stage.destroy();
    if (container.parentNode) container.parentNode.removeChild(container);
  }
}

function createRoundedRegularPolygon(
  x: number,
  y: number,
  sides: number,
  radius: number,
  cornerRadius: number,
  attrs: Record<string, any>,
) {
  const getPoints = (r: number, s: number) => {
    const pts: Array<{ x: number; y: number }> = [];
    for (let n = 0; n < s; n++) {
      pts.push({
        x: r * Math.sin((n * 2 * Math.PI) / s),
        y: -1 * r * Math.cos((n * 2 * Math.PI) / s),
      });
    }
    return pts;
  };

  const shape = new Shape({
    x,
    y,
    ...attrs,
    sceneFunc: (context, konvaShape) => {
      const points = getPoints(radius, sides);
      if (cornerRadius <= 0 || points.length < 3) {
        context.beginPath();
        context.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          context.lineTo(points[i].x, points[i].y);
        }
        context.closePath();
        context.fillStrokeShape(konvaShape);
        return;
      }

      context.beginPath();
      for (let i = 0; i < points.length; i++) {
        const current = points[i];
        const next = points[(i + 1) % points.length];
        const prev = points[(i - 1 + points.length) % points.length];

        const v1x = current.x - prev.x;
        const v1y = current.y - prev.y;
        const v2x = next.x - current.x;
        const v2y = next.y - current.y;

        const len1 = Math.hypot(v1x, v1y);
        const len2 = Math.hypot(v2x, v2y);
        const n1x = v1x / len1;
        const n1y = v1y / len1;
        const n2x = v2x / len2;
        const n2y = v2y / len2;

        const maxR = Math.min(len1, len2) / 2;
        const effR = Math.min(cornerRadius, maxR);

        const startX = current.x - n1x * effR;
        const startY = current.y - n1y * effR;
        const endX = current.x + n2x * effR;
        const endY = current.y + n2y * effR;

        if (i === 0) context.moveTo(startX, startY);
        else context.lineTo(startX, startY);
        context.quadraticCurveTo(current.x, current.y, endX, endY);
      }
      context.closePath();
      context.fillStrokeShape(konvaShape);
    },
  });
  return shape;
}

function addDrawingLines(layer: Layer | Group, clip: DrawingClipProps) {
  const lines: DrawingLine[] = clip.lines ?? [];
  for (const l of lines) {
    const smoothing = l.smoothing ?? 0.5;
    let lineCap: "round" | "square" = "round";
    let lineJoin: "round" | "bevel" = "round";
    let tension = smoothing;
    let gco: GlobalCompositeOperation = "source-over";
    if (l.tool === "highlighter") {
      lineCap = "square";
      lineJoin = "bevel";
      tension = 0;
      gco = "multiply";
    } else if (l.tool === "eraser") {
      lineCap = "round";
      lineJoin = "round";
      tension = 0.5;
      gco = "destination-out";
    }

    const node = new Line({
      points: l.points,
      stroke: l.stroke,
      strokeWidth: l.strokeWidth,
      opacity:
        l.tool === "eraser"
          ? 1
          : Math.max(0, Math.min(1, (l.opacity ?? 100) / 100)),
      x: l.transform.x,
      y: l.transform.y,
      scaleX: l.transform.scaleX,
      scaleY: l.transform.scaleY,
      rotation: l.transform.rotation,
      lineCap,
      lineJoin,
      tension,
      globalCompositeOperation: gco,
      perfectDrawEnabled: false,
      shadowForStrokeEnabled: false,
    });
    layer.add(node);
  }
}

export async function blitDrawing(
  canvas: Canvas,
  clip: DrawingClipProps,
  applicators?: Array<{
    apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
    ensureResources?: () => Promise<void>;
  }>,
): Promise<void> {
  const hasApplicators = Array.isArray(applicators) && applicators.length > 0;

  if (!hasApplicators) {
    withStage(canvas, (layer) => {
      // Map editor rect (BASE_LONG_SIDE space) to the target canvas using cover scaling
      const BASE_LONG_SIDE = 600; // keep in sync with editor world units
      const ratio = canvas.width / canvas.height;
      const rectWidth =
        Number.isFinite(ratio) && ratio > 0
          ? BASE_LONG_SIDE * ratio
          : BASE_LONG_SIDE;
      const rectHeight = BASE_LONG_SIDE;
      const scaleX = canvas.width / rectWidth;
      const scaleY = canvas.height / rectHeight;
      // Use contain scaling so drawings fit the target canvas
      const scale = Math.min(scaleX, scaleY);
      const x = (canvas.width - rectWidth * scale) / 2;
      const y = (canvas.height - rectHeight * scale) / 2;

      const group = new Group({
        x,
        y,
        scaleX: scale,
        scaleY: scale,
        listening: false,
      });
      layer.add(group);
      addDrawingLines(group, clip);
    });
    return;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const BASE_LONG_SIDE = 600; // keep in sync with editor world units
  const ratio = canvas.width / canvas.height;
  const rectWidth =
    Number.isFinite(ratio) && ratio > 0
      ? BASE_LONG_SIDE * ratio
      : BASE_LONG_SIDE;
  const rectHeight = BASE_LONG_SIDE;
  const scaleX = canvas.width / rectWidth;
  const scaleY = canvas.height / rectHeight;
  const scale = Math.min(scaleX, scaleY);
  const x = (canvas.width - rectWidth * scale) / 2;
  const y = (canvas.height - rectHeight * scale) / 2;

  // Run applicators per-line asynchronously to keep same behavior as other blitters
  const run = async () => {
    // Create offscreen Konva stage sized to destination canvas
    const container = document.createElement("div");
    container.style.position = "fixed";
    container.style.left = "-10000px";
    container.style.top = "-10000px";
    container.style.width = `${canvas.width}px`;
    container.style.height = `${canvas.height}px`;
    document.body.appendChild(container);

    const stage = new Stage({
      container,
      width: canvas.width,
      height: canvas.height,
    });
    const layer = new Layer();
    stage.add(layer);

    // Accumulation canvas where we composite each applicator-processed line
    const accum = document.createElement("canvas");
    accum.width = canvas.width;
    accum.height = canvas.height;
    const accCtx = accum.getContext("2d");
    if (!accCtx) {
      stage.destroy();
      if (container.parentNode) container.parentNode.removeChild(container);
      return;
    }
    accCtx.clearRect(0, 0, accum.width, accum.height);

    try {
      const lines: DrawingLine[] = Array.isArray(clip.lines) ? clip.lines : [];
      for (const l of lines) {
        // Determine intended composite op for final accumulation
        let compositeOp: GlobalCompositeOperation = "source-over";
        let lineCap: "round" | "square" = "round";
        let lineJoin: "round" | "bevel" = "round";
        let tension = l.smoothing ?? 0.5;
        if (l.tool === "highlighter") {
          lineCap = "square";
          lineJoin = "bevel";
          tension = 0;
          compositeOp = "multiply";
        } else if (l.tool === "eraser") {
          lineCap = "round";
          lineJoin = "round";
          tension = 0.5;
          compositeOp = "destination-out";
        }

        // Build a scoped group to apply world->canvas mapping
        const group = new Group({
          x,
          y,
          scaleX: scale,
          scaleY: scale,
          listening: false,
        });
        layer.add(group);

        // Create the line node; force source-over here, apply composite on accumulation step
        const node = new Line({
          points: l.points,
          stroke: l.tool === "eraser" ? "#000000" : l.stroke,
          strokeWidth: l.strokeWidth,
          opacity:
            l.tool === "eraser"
              ? 1
              : Math.max(0, Math.min(1, (l.opacity ?? 100) / 100)),
          x: l.transform.x,
          y: l.transform.y,
          scaleX: l.transform.scaleX,
          scaleY: l.transform.scaleY,
          rotation: l.transform.rotation,
          lineCap,
          lineJoin,
          tension,
          globalCompositeOperation: "source-over",
          perfectDrawEnabled: false,
          shadowForStrokeEnabled: false,
        });
        group.add(node);

        layer.draw();

        // Rasterize just this line to a canvas the size of the destination
        let lineCanvas = stage.toCanvas({
          pixelRatio: 1,
        }) as unknown as HTMLCanvasElement;

        // Apply applicators to the line raster
        for (const app of applicators || []) {
          try {
            await app.ensureResources?.();
          } catch {}
          const out = app.apply(lineCanvas);
          if (out && out !== lineCanvas) {
            lineCanvas = out;
          }
        }

        // Composite onto the accumulation canvas using the intended blend op
        accCtx.save();
        accCtx.globalCompositeOperation = compositeOp;
        accCtx.drawImage(lineCanvas, 0, 0);
        accCtx.restore();

        // Prepare for next line
        layer.destroyChildren();
      }

      // Flush accumulation to destination canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(accum, 0, 0);
    } finally {
      stage.destroy();
      if (container.parentNode) container.parentNode.removeChild(container);
    }
  };

  await run();
}

export function blitShape(
  canvas: Canvas,
  clip: ShapeClipProps | PolygonClipProps | StarClipProps,
  applicators?: Array<{
    apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
    ensureResources?: () => Promise<void>;
  }>,
): void {
  withStage(canvas, (layer) => {
    const t = resolveTransformFromClip(
      clip.transform,
      (clip as any).normalizedTransform,
      canvas,
    );
    const shapeType: ShapeTool =
      (clip as ShapeClipProps).shapeType ?? "rectangle";
    const fillHex = (clip as ShapeClipProps).fill ?? "#3b82f6";
    const strokeHex = (clip as ShapeClipProps).stroke ?? "#1e40af";
    const strokeWidth = (clip as ShapeClipProps).strokeWidth ?? 2;
    const fillOpacity = (clip as ShapeClipProps).fillOpacity ?? 100;
    const strokeOpacity = (clip as ShapeClipProps).strokeOpacity ?? 100;

    const fill = hexToRgba(fillHex, fillOpacity);
    const stroke = hexToRgba(strokeHex, strokeOpacity);

    const common: Record<string, any> = {
      rotation: t.rotation,
      scaleX: t.scaleX,
      scaleY: t.scaleY,
      opacity: Math.max(0, Math.min(1, t.opacity / 100)),
      fill,
      stroke,
      strokeWidth,
      perfectDrawEnabled: false,
      shadowForStrokeEnabled: false,
    };

    const actualWidth = t.width * t.scaleX;
    const actualHeight = t.height * t.scaleY;

    switch (shapeType) {
      case "rectangle": {
        const rect = new Rect({
          x: t.x,
          y: t.y,
          width: t.width,
          height: t.height,
          cornerRadius: t.cornerRadius ?? 0,
          ...common,
        });
        layer.add(rect);
        break;
      }
      case "ellipse": {
        const ell = new Ellipse({
          x: t.x + actualWidth / 2,
          y: t.y + actualHeight / 2,
          radiusX: t.width / 2,
          radiusY: t.height / 2,
          ...common,
        });
        layer.add(ell);
        break;
      }
      case "polygon": {
        const sides = (clip as PolygonClipProps).sides ?? 3;
        const radius = Math.min(t.width / Math.sqrt(3), t.height / 1.5);
        const poly = createRoundedRegularPolygon(
          t.x + actualWidth / 2,
          t.y + actualHeight / 2,
          sides,
          radius,
          t.cornerRadius ?? 0,
          common,
        );
        layer.add(poly);
        break;
      }
      case "line": {
        const line = new Line({
          x: t.x,
          y: t.y,
          points: [0, 0, t.width, 0],
          ...common,
        });
        layer.add(line);
        break;
      }
      case "star": {
        const points = (clip as StarClipProps).points ?? 5;
        const outer = Math.min(t.width, t.height) / 2;
        const inner = outer / 2;
        const star = new Star({
          x: t.x + actualWidth / 2,
          y: t.y + actualHeight / 2,
          numPoints: points,
          innerRadius: inner,
          outerRadius: outer,
          ...common,
        });
        layer.add(star);
        break;
      }
      default: {
        const rect = new Rect({
          x: t.x,
          y: t.y,
          width: t.width,
          height: t.height,
          cornerRadius: t.cornerRadius ?? 0,
          ...common,
        });
        layer.add(rect);
      }
    }
  });

  // Apply applicators post-render (optional)
  if (applicators && applicators.length) {
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const run = async () => {
        for (const app of applicators) {
          try {
            await app.ensureResources?.();
          } catch {}
          const result = app.apply(canvas);
          if (result && result !== canvas) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(result, 0, 0);
          }
        }
      };
      void run();
    }
  }
}

export function blitText(
  canvas: Canvas,
  clip: TextClipProps,
  applicators?: Array<{
    apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
    ensureResources?: () => Promise<void>;
  }>,
): void {
  withStage(canvas, (layer) => {
    const t = resolveTransformFromClip(
      clip.transform,
      (clip as any).normalizedTransform,
      canvas,
    );
    // Editor world units use BASE_LONG_SIDE = 600 for the short side of the rect.
    // Scale font-related sizes by the ratio of export canvas height to this base
    // so text appears consistent across resolutions (e.g. 1080p vs 480p).
    const BASE_LONG_SIDE = 600;
    const canvasScale = canvas.height > 0 ? canvas.height / BASE_LONG_SIDE : 1;

    const x = t.x ?? 0;
    const y = t.y ?? 0;
    const width = t.width;
    const height = t.height;

    const backgroundEnabled = clip.backgroundEnabled ?? false;
    const backgroundColor = clip.backgroundColor ?? "#000000";
    const backgroundOpacity = clip.backgroundOpacity ?? 100;
    const backgroundCornerRadius = clip.backgroundCornerRadius ?? 0;

    if (backgroundEnabled) {
      const bg = new Rect({
        x,
        y,
        width,
        height,
        fill: backgroundColor,
        opacity: Math.max(
          0,
          Math.min(1, (backgroundOpacity / 100) * (t.opacity / 100)),
        ),
        cornerRadius: backgroundCornerRadius * canvasScale,
        rotation: t.rotation,
        listening: false,
      });
      layer.add(bg);
    }

    const text = clip.text ?? "";
    const textTransformed = applyTextTransform(text, clip.textTransform);

    const baseFontSize = clip.fontSize ?? 48;
    const scaledFontSize = baseFontSize * canvasScale;

    const baseStrokeWidth = clip.strokeEnabled
      ? (clip.strokeWidth ?? 2)
      : undefined;
    const scaledStrokeWidth =
      baseStrokeWidth !== undefined ? baseStrokeWidth * canvasScale : undefined;

    const baseShadowBlur = clip.shadowEnabled
      ? (clip.shadowBlur ?? 4)
      : undefined;
    const scaledShadowBlur =
      baseShadowBlur !== undefined ? baseShadowBlur * canvasScale : undefined;

    const baseShadowOffsetX = clip.shadowEnabled
      ? (clip.shadowOffsetX ?? 2)
      : undefined;
    const baseShadowOffsetY = clip.shadowEnabled
      ? (clip.shadowOffsetY ?? 2)
      : undefined;
    const scaledShadowOffsetX =
      baseShadowOffsetX !== undefined
        ? baseShadowOffsetX * canvasScale
        : undefined;
    const scaledShadowOffsetY =
      baseShadowOffsetY !== undefined
        ? baseShadowOffsetY * canvasScale
        : undefined;

    const txt = new Text({
      x,
      y,
      width,
      height,
      text: textTransformed,
      fontSize: scaledFontSize,
      fontFamily: clip.fontFamily ?? "Arial",
      fontStyle:
        `${clip.fontStyle ?? "normal"} ${(clip.fontWeight ?? 400) >= 700 ? "bold" : "normal"}`.trim(),
      textDecoration: clip.textDecoration ?? "none",
      align: clip.textAlign ?? "left",
      verticalAlign: clip.verticalAlign ?? "top",
      fill: clip.color ?? "#000000",
      fillOpacity: (clip.colorOpacity ?? 100) / 100,
      stroke: clip.strokeEnabled ? (clip.stroke ?? "#000000") : undefined,
      strokeWidth: scaledStrokeWidth,
      shadowColor: clip.shadowEnabled
        ? (clip.shadowColor ?? "#000000")
        : undefined,
      shadowBlur: scaledShadowBlur,
      shadowOpacity: clip.shadowEnabled
        ? (clip.shadowOpacity ?? 75) / 100
        : undefined,
      shadowOffsetX: scaledShadowOffsetX,
      shadowOffsetY: scaledShadowOffsetY,
      opacity: Math.max(0, Math.min(1, t.opacity / 100)),
      rotation: t.rotation,
      scaleX: t.scaleX,
      scaleY: t.scaleY,
      perfectDrawEnabled: false,
      shadowForStrokeEnabled: false,
    });
    layer.add(txt);
  });

  // Apply applicators post-render (optional)
  if (applicators && applicators.length) {
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const run = async () => {
        for (const app of applicators) {
          try {
            await app.ensureResources?.();
          } catch {}
          const result = app.apply(canvas);
          if (result && result !== canvas) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(result, 0, 0);
          }
        }
      };
      void run();
    }
  }
}

export type { Canvas };

// Minimal ImageClipProps type for export usage
export interface ImageClipProps {
  clipId: string;
  type: "image";
  src: string;
  // Optional preprocessor outputs that may override src for specific frames
  preprocessors?: Array<{
    src?: string;
    startFrame?: number;
    endFrame?: number;
    status?: "running" | "complete" | "failed";
  }>;
  transform?: ClipTransform;
  originalTransform?: ClipTransform;
  normalizedTransform?: NormalizedTransform;
  // adjustments
  brightness?: number;
  contrast?: number;
  hue?: number;
  saturation?: number;
  blur?: number;
  noise?: number;
  sharpness?: number;
  vignette?: number;
  // masks
  masks?: Array<MaskClipProps>;
  // timing
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
}

// Minimal VideoClipProps type for export usage
export interface VideoClipProps {
  clipId: string;
  type: "video";
  src: string;
  // preprocessors may swap source for specific frame ranges
  preprocessors?: Array<{
    src?: string;
    startFrame?: number;
    endFrame?: number;
    status?: "running" | "complete" | "failed";
  }>;
  transform?: ClipTransform;
  originalTransform?: ClipTransform;
  normalizedTransform?: NormalizedTransform;
  // adjustments same as image
  brightness?: number;
  contrast?: number;
  hue?: number;
  saturation?: number;
  blur?: number;
  noise?: number;
  sharpness?: number;
  vignette?: number;
  masks?: Array<MaskClipProps>;
  // timing
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
  speed?: number;
}

// Iterator cache keyed by clipId to drive frame-by-frame decoding via .next()
// Iterator and decoder are owned by caller; no internal cache here

export function resolveVideoSourceForFrame(
  c: VideoClipProps,
  projectFrame: number,
): { selectedSrc: string; frameOffset: number } {
  const preprocessors = Array.isArray(c?.preprocessors) ? c.preprocessors : [];
  const trimStart = Number(c?.trimStart || 0);
  if (!preprocessors.length) return { selectedSrc: c.src, frameOffset: 0 };
  const localFrame = Math.max(0, projectFrame - (Number(c?.startFrame) || 0));
  for (const p of preprocessors) {
    if (p?.status !== "complete" || !p?.src) continue;
    const s = (Number(p.startFrame) || 0) + trimStart;
    const e = (Number(p.endFrame) ?? Number(p.startFrame) ?? 0) + trimStart;
    if (localFrame >= s && localFrame <= e) {
      return { selectedSrc: p.src, frameOffset: Number(p.startFrame) || 0 };
    }
  }
  return { selectedSrc: c.src, frameOffset: 0 };
}

export async function blitVideo(
  canvas: Canvas,
  clip: VideoClipProps,
  applicators:
    | Array<{
        apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
        ensureResources?: () => Promise<void>;
      }>
    | undefined,
  iterator: AsyncIterator<WrappedCanvas | null>,
  projectFrame: number,
): Promise<void> {
  const content = document.createElement("canvas");
  content.width = canvas.width;
  content.height = canvas.height;
  const cctx = content.getContext("2d");
  if (!cctx) return;

  try {
    const { value } = (await iterator.next()) as {
      value: WrappedCanvas | null;
    };
    const wrapped = value;
    if (!wrapped) return;

    const t = resolveTransformFromClip(
      clip.transform,
      clip.normalizedTransform,
      canvas,
    );

    // Draw decoded frame into content, honoring crop (if any) in the same
    // normalized coordinates used by the renderer preview components.
    cctx.clearRect(0, 0, content.width, content.height);
    cctx.imageSmoothingEnabled = true;
    // @ts-ignore
    cctx.imageSmoothingQuality = "high";
    try {
      const sourceCanvas = wrapped.canvas as HTMLCanvasElement;
      const sw = Math.max(1, sourceCanvas.width);
      const sh = Math.max(1, sourceCanvas.height);
      const crop = t.crop;
      if (crop && crop.width > 0 && crop.height > 0) {
        const sx = Math.max(0, Math.min(sw, crop.x * sw));
        const sy = Math.max(0, Math.min(sh, crop.y * sh));
        const sWidth = Math.max(1, Math.min(sw - sx, crop.width * sw));
        const sHeight = Math.max(1, Math.min(sh - sy, crop.height * sh));
        const dx = 0;
        const dy = 0;
        const dWidth = content.width;
        const dHeight = content.height;
        console.log(sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
        cctx.drawImage(
          sourceCanvas,
          sx,
          sy,
          sWidth,
          sHeight,
          dx,
          dy,
          dWidth,
          dHeight,
        );
      } else {
        cctx.drawImage(sourceCanvas, 0, 0, content.width, content.height);
      }
    } catch {}

    // Apply masks
    if (Array.isArray(clip.masks) && clip.masks.length > 0) {
      // Compute mask frame honoring speed and preprocessor offset to match preview behavior
      const speedFactor = Math.max(0.1, Number(clip.speed ?? 1));
      const { selectedSrc, frameOffset } = resolveVideoSourceForFrame(
        clip,
        projectFrame,
      );
      const isUsingPreprocessorSrc = selectedSrc !== clip.src;
      const baseLocal = Math.max(
        0,
        projectFrame - (Number(clip.startFrame) || 0),
      );
      const derivedLocal = isUsingPreprocessorSrc
        ? Math.max(0, baseLocal - Math.max(0, frameOffset))
        : Math.max(0, baseLocal + (Number(clip.trimStart) || 0));
      const maskFrame = Math.max(0, Math.floor(derivedLocal * speedFactor));

      applyMasksToCanvas(content, {
        focusFrame: projectFrame,
        masks: clip.masks as MaskClipProps[],
        clip: clip,
        disabled: false,
        maskFrameOverride: maskFrame,
      });
    }

    // Apply WebGL filters
    applyWebGLFilters(content, {
      brightness: clip.brightness,
      contrast: clip.contrast,
      hue: clip.hue,
      saturation: clip.saturation,
      blur: clip.blur,
      noise: clip.noise,
      sharpness: clip.sharpness,
      vignette: clip.vignette,
    });

    // Apply applicators to content canvas before compositing
    if (applicators && applicators.length) {
      for (const app of applicators) {
        try {
          await app.ensureResources?.();
        } catch {}
        const out = app.apply(content);
        if (out && out !== content) {
          cctx.clearRect(0, 0, content.width, content.height);
          cctx.drawImage(out, 0, 0, content.width, content.height);
        }
      }
    }

    // Composite to destination canvas with transform/opacity/rotation
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawRoundedImage(
      ctx,
      content,
      t.x * (t.crop?.width ?? 1),
      t.y * (t.crop?.height ?? 1),
      t.width,
      t.height,
      t.cornerRadius ?? 0,
      t.rotation ?? 0,
      t.opacity ?? 100,
      t.scaleX ?? 1,
      t.scaleY ?? 1,
    );
  } catch {
    // swallow for robustness
  }
}

export function cleanupVideoDecoders(): void {}

function drawRoundedImage(
  ctx: CanvasRenderingContext2D,
  imageCanvas: HTMLCanvasElement,
  x: number,
  y: number,
  width: number,
  height: number,
  cornerRadius: number,
  rotationDeg: number,
  opacity: number,
  scaleX: number = 1,
  scaleY: number = 1,
) {
  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity / 100));

  // Mimic Konva Image: apply transform first, then draw at local 0,0
  ctx.translate(x, y);
  const rad = ((rotationDeg ?? 0) * Math.PI) / 180;
  if (rad) ctx.rotate(rad);
  if (scaleX !== 1 || scaleY !== 1) ctx.scale(scaleX, scaleY);

  if (cornerRadius && cornerRadius > 0) {
    const r = Math.min(cornerRadius, Math.min(width, height) / 2);
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(width - r, 0);
    ctx.quadraticCurveTo(width, 0, width, r);
    ctx.lineTo(width, height - r);
    ctx.quadraticCurveTo(width, height, width - r, height);
    ctx.lineTo(r, height);
    ctx.quadraticCurveTo(0, height, 0, height - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.clip();
  }

  ctx.drawImage(imageCanvas, 0, 0, width, height);
  ctx.restore();
}

export async function blitImage(
  canvas: Canvas,
  clip: ImageClipProps,
  applicators?: Array<{
    apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
    ensureResources?: () => Promise<void>;
  }>,
  focusFrame?: number,
): Promise<void> {
  const t = resolveTransformFromClip(
    clip.transform,
    clip.normalizedTransform,
    canvas,
  );

  const content = document.createElement("canvas");
  content.width = Math.max(1, Math.floor(t.width));
  content.height = Math.max(1, Math.floor(t.height));
  const cctx = content.getContext("2d");
  if (!cctx) return;

  const loadAndDraw = async () => {
    try {
      // Resolve effective src based on preprocessors and current frame (if provided)
      let effectiveSrc = clip.src;
      if (
        clip.preprocessors &&
        Array.isArray(clip.preprocessors) &&
        clip.preprocessors.length > 0
      ) {
        const f = typeof focusFrame === "number" ? focusFrame : undefined;
        const trimStart = isFinite(clip.trimStart ?? 0)
          ? (clip.trimStart ?? 0)
          : 0;
        if (typeof f === "number") {
          const matched = clip.preprocessors.find((p) => {
            if (p?.status !== "complete" || !p?.src) return false;
            const s =
              typeof p.startFrame === "number"
                ? p.startFrame + trimStart
                : undefined;
            const e =
              typeof p.endFrame === "number" ? p.endFrame + trimStart : s;
            if (s === undefined && e === undefined) return false;
            if (s !== undefined && e !== undefined) return f >= s && f <= e;
            if (s !== undefined) return f === s;
            return false;
          });
          if (matched?.src) {
            effectiveSrc = matched.src;
          }
        } else {
          // No frame provided: fall back to first completed preprocessor src
          const first = clip.preprocessors.find(
            (p) => p?.status === "complete" && !!p?.src,
          );
          if (first?.src) effectiveSrc = first.src;
        }
      }
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = (e) => reject(e);
        im.src = effectiveSrc;
      });
      cctx.clearRect(0, 0, content.width, content.height);
      cctx.imageSmoothingEnabled = true;
      // @ts-ignore
      cctx.imageSmoothingQuality = "high";
      // Draw image into content, honoring normalized crop (if any) to match
      // the behavior of `ImagePreview` in the renderer.
      const nativeW = img.naturalWidth || img.width || content.width;
      const nativeH = img.naturalHeight || img.height || content.height;
      const crop = t.crop;
      if (crop && crop.width > 0 && crop.height > 0) {
        // Map the normalized crop rect both in source-space (native media
        // pixels) and destination-space (content canvas pixels) so that we
        // *crop* without introducing an extra zoom when both width and height
        // are cropped.
        const sx = Math.max(0, Math.min(nativeW, crop.x * nativeW));
        const sy = Math.max(0, Math.min(nativeH, crop.y * nativeH));
        const sWidth = Math.max(
          1,
          Math.min(nativeW - sx, crop.width * nativeW),
        );
        const sHeight = Math.max(
          1,
          Math.min(nativeH - sy, crop.height * nativeH),
        );
        const dx = 0;
        const dy = 0;
        cctx.drawImage(
          img,
          sx,
          sy,
          sWidth,
          sHeight,
          dx,
          dy,
          content.width,
          content.height,
        );
      } else {
        cctx.drawImage(img, 0, 0, content.width, content.height);
      }

      // Apply masks
      if (Array.isArray(clip.masks) && clip.masks.length > 0) {
        applyMasksToCanvas(content, {
          focusFrame: 0,
          masks: clip.masks as MaskClipProps[],
          clip: clip,
          disabled: false,
        });
      }

      // Apply WebGL filters
      applyWebGLFilters(content, {
        brightness: clip.brightness,
        contrast: clip.contrast,
        hue: clip.hue,
        saturation: clip.saturation,
        blur: clip.blur,
        noise: clip.noise,
        sharpness: clip.sharpness,
        vignette: clip.vignette,
      });

      // Apply applicators to content canvas before compositing
      if (applicators && applicators.length) {
        for (const app of applicators) {
          try {
            await app.ensureResources?.();
          } catch {}
          const out = app.apply(content);
          if (out && out !== content) {
            cctx.clearRect(0, 0, content.width, content.height);
            cctx.drawImage(out, 0, 0, content.width, content.height);
          }
        }
      }

      // Composite to destination canvas at transform with rotation, corner radius, opacity
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      drawRoundedImage(
        ctx,
        content,
        t.x,
        t.y,
        t.width,
        t.height,
        t.cornerRadius ?? 0,
        t.rotation ?? 0,
        t.opacity ?? 100,
        t.scaleX ?? 1,
        t.scaleY ?? 1,
      );
    } catch {
      // swallow errors for robustness
    }
  };

  await loadAndDraw();
}

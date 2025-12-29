import { Stage } from "konva/lib/Stage.js";
import { Layer } from "konva/lib/Layer.js";
import { Group } from "konva/lib/Group.js";
import { Line } from "konva/lib/shapes/Line.js";
import { Rect } from "konva/lib/shapes/Rect.js";
import { Ellipse } from "konva/lib/shapes/Ellipse.js";
import { Star } from "konva/lib/shapes/Star.js";
import { Text } from "konva/lib/shapes/Text.js";
import { Shape } from "konva/lib/Shape.js";
import { Image as KonvaImage } from "konva/lib/shapes/Image.js";
import { applyWebGLFilters } from "./webgl-filters/apply";
import { applyMasksToCanvas } from "./masks/apply";
import { resolveVideoSourceForFrame } from "./blit";
import type {
  DrawingClipProps,
  ShapeClipProps,
  ClipTransform,
  ImageClipProps,
  MaskClipProps,
  VideoClipProps,
} from "./blit";
import type { WrappedCanvas } from "mediabunny";
import { getMediaInfo } from "../../renderer/src/lib/media/utils";

// Simple in-memory cache so repeated requests for the same image src
// reuse the same HTMLImageElement (and underlying load work).
const imageElementCache = new Map<string, Promise<HTMLImageElement>>();

async function loadCachedImage(effectiveSrc: string): Promise<HTMLImageElement> {
  const cached = imageElementCache.get(effectiveSrc);
  if (cached) {
    return cached;
  }

  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const im = new Image();
    im.crossOrigin = "anonymous";

    im.onload = () => {
      resolve(im);
    };

    im.onerror = (e) => {
      // If loading fails, remove from cache so future attempts can retry.
      imageElementCache.delete(effectiveSrc);
      reject(e);
    };

    getMediaInfo(effectiveSrc)
      .then((info) => {
        if (info) {
          im.width = info.image?.width ?? 0;
          im.height = info.image?.height ?? 0;
          if (info.image?.input && typeof info.image.input === "string") {
            im.src = info.image.input;
            return;
          }
        }

        // Fallback to the effectiveSrc if there is no info
        // or if it does not contain a usable image input.
        im.src = effectiveSrc;
      })
      .catch((e) => {
        console.error("Error loading image info", e);
        // Even if media info fetching fails, still attempt to load the image.
        im.src = effectiveSrc;
      });

  });

  imageElementCache.set(effectiveSrc, promise);
  return promise;
}

// Re-defining missing local types from blit.ts to fix linter errors
// These are not exported from blit.ts, so we must define them locally or export them from blit.ts.
// Defining locally to avoid modifying blit.ts structure which might be used elsewhere.

export type ShapeTool = "rectangle" | "ellipse" | "polygon" | "line" | "star";

export type NormalizedTransform = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

interface PolygonClipProps extends ShapeClipProps {
  shapeType: "polygon";
  sides?: number;
}

interface StarClipProps extends ShapeClipProps {
  shapeType: "star";
  points?: number;
}

export interface TextClipProps {
  clipId: string;
  transform?: ClipTransform;
  normalizedTransform?: NormalizedTransform;
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

export interface KonvaExportRendererOptions {
  width: number;
  height: number;
  /**
   * Optional pixel ratio for high-resolution exports.
   * Defaults to 1.
   * For 4K export from a 1080p stage, use 2.
   */
  pixelRatio?: number;
}

export interface ToBlobOptions {
  callback?: (blob: Blob | null) => void;
  pixelRatio?: number;
  mimeType?: string;
  quality?: number;
}

export class KonvaExportRenderer {
  private stage: Stage;
  private layer: Layer;
  private container: HTMLDivElement;
  private width: number;
  private height: number;
  private defaultPixelRatio: number;

  constructor({ width, height, pixelRatio = 1 }: KonvaExportRendererOptions) {
    this.width = width;
    this.height = height;
    this.defaultPixelRatio = pixelRatio;

    this.container = document.createElement("div");
    this.container.style.position = "fixed";
    // Move off-screen
    this.container.style.left = "-99999px";
    this.container.style.top = "-99999px";
    this.container.style.width = `${width}px`;
    this.container.style.height = `${height}px`;
    // Ensure container is in DOM so Konva can attach event listeners/calculate styles correctly
    document.body.appendChild(this.container);

    this.stage = new Stage({
      container: this.container,
      width: this.width,
      height: this.height,
    });

    this.layer = new Layer();
    this.stage.add(this.layer);
  }

  /**
   * Exports the stage to a Blob.
   * Supports high resolution exports via pixelRatio.
   */
  public toBlob(
    optionsOrCallback?: ToBlobOptions | ((blob: Blob | null) => void),
  ): Promise<Blob | null> {
    let options: ToBlobOptions = {};

    if (typeof optionsOrCallback === "function") {
      options = { callback: optionsOrCallback };
    } else if (optionsOrCallback) {
      options = optionsOrCallback;
    }

    const {
      callback,
      pixelRatio = this.defaultPixelRatio,
      mimeType = "image/png",
      quality,
    } = options;

    return new Promise((resolve, reject) => {
      try {
        this.stage.toBlob({
          pixelRatio,
          mimeType,
          quality,
          callback: (blob) => {
            if (callback) {
              callback(blob);
            }
            resolve(blob);
          },
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  public getStage(): Stage {
    return this.stage;
  }

  public getLayer(): Layer {
    return this.layer;
  }

  public destroy(): void {
    this.stage.destroy();
    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }

  public clearStage(): void {
    this.stage.getLayers().forEach((layer) => {
      layer.destroyChildren();
    });
    this.stage.draw();
  }

  public async blitDrawing(
    clip: DrawingClipProps,
    applicators?: Array<{
      apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
      ensureResources?: () => Promise<void>;
    }>,
  ): Promise<void> {
    const hasApplicators = Array.isArray(applicators) && applicators.length > 0;

    // Common setup for coordinate mapping
    const BASE_LONG_SIDE = 600; // keep in sync with editor world units
    const width = this.width;
    const height = this.height;

    const ratio = width / height;
    const rectWidth =
      Number.isFinite(ratio) && ratio > 0
        ? BASE_LONG_SIDE * ratio
        : BASE_LONG_SIDE;
    const rectHeight = BASE_LONG_SIDE;
    const scaleX = width / rectWidth;
    const scaleY = height / rectHeight;
    // Use contain scaling so drawings fit the target canvas
    const scale = Math.min(scaleX, scaleY);
    const x = (width - rectWidth * scale) / 2;
    const y = (height - rectHeight * scale) / 2;

    if (!hasApplicators) {
      const group = new Group({
        x,
        y,
        scaleX: scale,
        scaleY: scale,
        listening: false,
      });
      this.layer.add(group);
      this.addDrawingLines(group, clip);
      return;
    }

    // Accumulation canvas where we composite each applicator-processed line
    const accum = document.createElement("canvas");
    accum.width = width;
    accum.height = height;
    const accCtx = accum.getContext("2d");
    if (!accCtx) return;

    // Use a temporary layer on the current stage to render lines one by one.
    // We avoid creating a new Stage to keep things lightweight, as we are already off-screen.
    const tempLayer = new Layer();
    this.stage.add(tempLayer);
    // Move temp layer to bottom so it doesn't obscure anything (though we clean it up anyway)
    tempLayer.moveToBottom();

    try {
      const lines = clip.lines ?? [];
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
        tempLayer.add(group);

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

        // Draw just this layer
        tempLayer.draw();

        // Rasterize just this line to a canvas the size of the destination
        // We use the tempLayer's toCanvas which captures just this line
        let lineCanvas = tempLayer.toCanvas({
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
        tempLayer.destroyChildren();
      }

      // Add the accumulated bitmap to the main layer
      const image = new KonvaImage({
        image: accum,
        x: 0,
        y: 0,
        width: width,
        height: height,
        listening: false,
      });
      this.layer.add(image);
    } finally {
      tempLayer.destroy();
    }
  }

  public async blitImage(
    clip: ImageClipProps,
    applicators?: Array<{
      apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
      ensureResources?: () => Promise<void>;
    }>,
    focusFrame?: number,
  ): Promise<void> {
    const t = this.resolveTransformFromClip(
      clip.transform,
      clip.normalizedTransform,
    );

    // 1. Resolve effective source
    let effectiveSrc = clip.src;
    if (
      clip.preprocessors &&
      Array.isArray(clip.preprocessors) &&
      clip.preprocessors.length > 0
    ) {
      const f = typeof focusFrame === "number" ? focusFrame : undefined;
      const trimStart = Number.isFinite(clip.trimStart ?? 0)
        ? (clip.trimStart ?? 0)
        : 0;
      if (typeof f === "number") {
        const matched = clip.preprocessors.find((p) => {
          if (p?.status !== "complete" || !p?.src) return false;
          const s =
            typeof p.startFrame === "number"
              ? p.startFrame + trimStart
              : undefined;
          const e = typeof p.endFrame === "number" ? p.endFrame + trimStart : s;
          if (s === undefined && e === undefined) return false;
          if (s !== undefined && e !== undefined) return f >= s && f <= e;
          if (s !== undefined) return f === s;
          return false;
        });
        if (matched?.src) {
          effectiveSrc = matched.src;
        }
      } else {
        const first = clip.preprocessors.find(
          (p) => p?.status === "complete" && !!p?.src,
        );
        if (first?.src) effectiveSrc = first.src;
      }
    }

    // 2. Load Image (cached by effectiveSrc)
    const img = await loadCachedImage(effectiveSrc);


    // 3. Prepare high-quality texture
    // Use natural dimensions to preserve maximum quality.
    const texWidth = Math.max(1, img.naturalWidth);
    const texHeight = Math.max(1, img.naturalHeight);

    const content = document.createElement("canvas");
    content.width = texWidth;
    content.height = texHeight;
    const ctx = content.getContext("2d");
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    // @ts-ignore
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, texWidth, texHeight);
    const data = ctx.getImageData(0, 0, texWidth, texHeight);

    // 4. Apply Masks
    if (Array.isArray(clip.masks) && clip.masks.length > 0) {
      // map masks to the content canvas
      applyMasksToCanvas(content, {
        focusFrame: typeof focusFrame === "number" ? focusFrame : 0,
        masks: clip.masks,
        clip: clip,
        disabled: false,
      });
    }

    // 5. Apply Filters
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

    // 6. Apply Applicators
    if (applicators && applicators.length) {
      for (const app of applicators) {
        try {
          await app.ensureResources?.();
        } catch {}
        const out = app.apply(content);
        if (out && out !== content) {
          // If size changed, resize content canvas
          if (out.width !== content.width || out.height !== content.height) {
            content.width = out.width;
            content.height = out.height;
          }
          const ctx2 = content.getContext("2d");
          if (ctx2) {
            ctx2.clearRect(0, 0, content.width, content.height);
            ctx2.drawImage(out, 0, 0);
          }
        }
      }
    }

    // 7. Calculate Crop (mapped to texture dimensions)
    let pixelCrop = undefined;
    if (t.crop) {
      pixelCrop = {
        x: t.crop.x * content.width,
        y: t.crop.y * content.height,
        width: t.crop.width * content.width,
        height: t.crop.height * content.height,
      };
    }

    // 8. Create Konva Node
    const node = new KonvaImage({
      image: content,
      x: t.x,
      y: t.y,
      width: t.width,
      height: t.height,
      scaleX: t.scaleX,
      scaleY: t.scaleY,
      rotation: t.rotation,
      cornerRadius: t.cornerRadius,
      opacity: t.opacity / 100,
      crop: pixelCrop,
      listening: false,
      perfectDrawEnabled: true,
      shadowForStrokeEnabled: false,
    });

    this.layer.add(node);
  }

  public async blitVideo(
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
    const t = this.resolveTransformFromClip(
      clip.transform,
      clip.normalizedTransform,
    );

    // 1. Get Frame from Iterator
    const { value } = (await iterator.next()) as {
      value: WrappedCanvas | null;
    };
    const wrapped = value;
    if (!wrapped) return;
    const sourceCanvas = wrapped.canvas as HTMLCanvasElement;

    // 2. Prepare content canvas (High Quality Texture)
    // Use source dimensions to preserve quality, matching blitImage approach
    const texWidth = Math.max(1, sourceCanvas.width);
    const texHeight = Math.max(1, sourceCanvas.height);

    const content = document.createElement("canvas");
    content.width = texWidth;
    content.height = texHeight;
    const ctx = content.getContext("2d");
    if (!ctx) return;

    ctx.imageSmoothingEnabled = true;
    // @ts-ignore
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(sourceCanvas, 0, 0, texWidth, texHeight);

    // 3. Apply Masks
    if (Array.isArray(clip.masks) && clip.masks.length > 0) {
      // Calculate mask frame - Logic from VideoPreview.tsx / blit.ts
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
        masks: clip.masks,
        clip: clip,
        disabled: false,
        maskFrameOverride: maskFrame,
      });
    }

    // 4. Apply Filters
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

    // 5. Apply Applicators
    if (applicators && applicators.length) {
      for (const app of applicators) {
        try {
          await app.ensureResources?.();
        } catch {}
        const out = app.apply(content);
        if (out && out !== content) {
          if (out.width !== content.width || out.height !== content.height) {
            content.width = out.width;
            content.height = out.height;
          }
          const ctx2 = content.getContext("2d");
          if (ctx2) {
            ctx2.clearRect(0, 0, content.width, content.height);
            ctx2.drawImage(out, 0, 0);
          }
        }
      }
    }

    // 6. Calculate Crop (mapped to texture dimensions)
    let pixelCrop = undefined;
    if (t.crop) {
      pixelCrop = {
        x: t.crop.x * content.width,
        y: t.crop.y * content.height,
        width: t.crop.width * content.width,
        height: t.crop.height * content.height,
      };
    }

    // 7. Create Konva Node
    const node = new KonvaImage({
      image: content,
      x: t.x,
      y: t.y,
      width: t.width,
      height: t.height,
      scaleX: t.scaleX,
      scaleY: t.scaleY,
      rotation: t.rotation,
      cornerRadius: t.cornerRadius,
      opacity: t.opacity / 100,
      crop: pixelCrop,
      listening: false,
      perfectDrawEnabled: true,
      shadowForStrokeEnabled: false,
    });

    this.layer.add(node);
  }

  public async blitShape(
    clip: ShapeClipProps | PolygonClipProps | StarClipProps,
    applicators?: Array<{
      apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
      ensureResources?: () => Promise<void>;
    }>,
  ): Promise<void> {
    const hasApplicators = Array.isArray(applicators) && applicators.length > 0;

    // Resolve transform logic locally to avoid dependency on non-exported blit functions
    const t = this.resolveTransformFromClip(
      clip.transform,
      (clip as any).normalizedTransform,
    );
    const shapeType: ShapeTool =
      (clip as ShapeClipProps).shapeType ?? "rectangle";
    const fillHex = (clip as ShapeClipProps).fill ?? "#3b82f6";
    const strokeHex = (clip as ShapeClipProps).stroke ?? "#1e40af";
    const strokeWidth = (clip as ShapeClipProps).strokeWidth ?? 2;
    const fillOpacity = (clip as ShapeClipProps).fillOpacity ?? 100;
    const strokeOpacity = (clip as ShapeClipProps).strokeOpacity ?? 100;

    const fill = this.hexToRgba(fillHex, fillOpacity);
    const stroke = this.hexToRgba(strokeHex, strokeOpacity);

    // Common properties matching ShapePreview.tsx logic
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
      draggable: false,
      listening: false,
    };

    const actualWidth = t.width * t.scaleX;
    const actualHeight = t.height * t.scaleY;

    // Helper to create the Konva node
    const createNode = () => {
      switch (shapeType) {
        case "rectangle": {
          return new Rect({
            x: t.x,
            y: t.y,
            width: t.width,
            height: t.height,
            cornerRadius: t.cornerRadius ?? 0,
            ...common,
          });
        }
        case "ellipse": {
          return new Ellipse({
            x: t.x + actualWidth / 2,
            y: t.y + actualHeight / 2,
            radiusX: t.width / 2,
            radiusY: t.height / 2,
            ...common,
          });
        }
        case "polygon": {
          const sides = (clip as PolygonClipProps).sides ?? 3;
          const radius = Math.min(t.width / Math.sqrt(3), t.height / 1.5);
          return this.createRoundedRegularPolygon(
            t.x + actualWidth / 2,
            t.y + actualHeight / 2,
            sides,
            radius,
            t.cornerRadius ?? 0,
            common,
          );
        }
        case "line": {
          return new Line({
            x: t.x,
            y: t.y,
            points: [0, 0, t.width, 0],
            ...common,
          });
        }
        case "star": {
          const points = (clip as StarClipProps).points ?? 5;
          const outer = Math.min(t.width, t.height) / 2;
          const inner = outer / 2;
          return new Star({
            x: t.x + actualWidth / 2,
            y: t.y + actualHeight / 2,
            numPoints: points,
            innerRadius: inner,
            outerRadius: outer,
            ...common,
          });
        }
        default: {
          return new Rect({
            x: t.x,
            y: t.y,
            width: t.width,
            height: t.height,
            cornerRadius: t.cornerRadius ?? 0,
            ...common,
          });
        }
      }
    };

    if (!hasApplicators) {
      const node = createNode();
      this.layer.add(node);
      return;
    }

    // Handle applicators: render shape to temp layer, rasterize, apply filters, composite back
    const tempLayer = new Layer();
    this.stage.add(tempLayer);
    tempLayer.moveToBottom();

    try {
      const node = createNode();
      tempLayer.add(node);
      tempLayer.draw();

      let shapeCanvas = tempLayer.toCanvas({
        pixelRatio: 1,
      }) as unknown as HTMLCanvasElement;

      for (const app of applicators || []) {
        try {
          await app.ensureResources?.();
        } catch {}
        const out = app.apply(shapeCanvas);
        if (out && out !== shapeCanvas) {
          shapeCanvas = out;
        }
      }

      const image = new KonvaImage({
        image: shapeCanvas,
        x: 0,
        y: 0,
        width: this.width,
        height: this.height,
        listening: false,
      });
      this.layer.add(image);
    } finally {
      tempLayer.destroy();
    }
  }

  public async blitText(
    clip: TextClipProps,
    applicators?: Array<{
      apply: (c: HTMLCanvasElement) => HTMLCanvasElement;
      ensureResources?: () => Promise<void>;
    }>,
  ): Promise<void> {
    const hasApplicators = Array.isArray(applicators) && applicators.length > 0;

    const t = this.resolveTransformFromClip(
      clip.transform,
      clip.normalizedTransform,
    );
    // Editor world units use BASE_LONG_SIDE = 600 for the short side of the rect.
    // Scale font-related sizes by the ratio of export canvas height to this base
    // so text appears consistent across resolutions (e.g. 1080p vs 480p).
    const BASE_LONG_SIDE = 600;
    const canvasScale = this.height > 0 ? this.height / BASE_LONG_SIDE : 1;

    const x = t.x ?? 0;
    const y = t.y ?? 0;
    const width = t.width;
    const height = t.height;

    const backgroundEnabled = clip.backgroundEnabled ?? false;
    const backgroundColor = clip.backgroundColor ?? "#000000";
    const backgroundOpacity = clip.backgroundOpacity ?? 100;
    const backgroundCornerRadius = clip.backgroundCornerRadius ?? 0;

    const text = clip.text ?? "";
    const textTransformed = this.applyTextTransform(text, clip.textTransform);

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

    // Helper to create nodes
    const createNodes = () => {
      const nodes: (Rect | Text)[] = [];

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
        nodes.push(bg);
      }

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
      nodes.push(txt);

      return nodes;
    };

    if (!hasApplicators) {
      const nodes = createNodes();
      nodes.forEach((node) => this.layer.add(node));
      return;
    }

    // Handle applicators
    const tempLayer = new Layer();
    this.stage.add(tempLayer);
    tempLayer.moveToBottom();

    try {
      const nodes = createNodes();
      nodes.forEach((node) => tempLayer.add(node));
      tempLayer.draw();

      let textCanvas = tempLayer.toCanvas({
        pixelRatio: 1,
      }) as unknown as HTMLCanvasElement;

      for (const app of applicators || []) {
        try {
          await app.ensureResources?.();
        } catch {}
        const out = app.apply(textCanvas);
        if (out && out !== textCanvas) {
          textCanvas = out;
        }
      }

      const image = new KonvaImage({
        image: textCanvas,
        x: 0,
        y: 0,
        width: this.width,
        height: this.height,
        listening: false,
      });
      this.layer.add(image);
    } finally {
      tempLayer.destroy();
    }
  }

  private addDrawingLines(container: Group | Layer, clip: DrawingClipProps) {
    const lines = clip.lines ?? [];
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
      container.add(node);
    }
  }

  private applyTextTransform(text: string, textTransform?: string): string {
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

  // --- Helper methods duplicated from blit.ts to ensure self-contained class ---

  private resolveTransformFromClip(
    transform: ClipTransform | undefined,
    normalizedTransform: NormalizedTransform | undefined,
  ): Required<Omit<ClipTransform, "crop">> & { crop?: ClipTransform["crop"] } {
    const cw = Math.max(1, this.width || 1);
    const ch = Math.max(1, this.height || 1);

    const ensureTransform = (t?: ClipTransform) => ({
      x: t?.x ?? 0,
      y: t?.y ?? 0,
      width: t?.width ?? cw,
      height: t?.height ?? ch,
      scaleX: t?.scaleX ?? 1,
      scaleY: t?.scaleY ?? 1,
      rotation: t?.rotation ?? 0,
      cornerRadius: t?.cornerRadius ?? 0,
      opacity: t?.opacity ?? 100,
      crop: t?.crop,
    });

    const base = ensureTransform(transform);
    if (!normalizedTransform) return base;

    const toPx = (normVal: any, dim: number, fallback: number): number => {
      const n = Number(normVal);
      if (!Number.isFinite(n)) return fallback;
      return n * dim;
    };

    return {
      ...base,
      x: toPx(normalizedTransform.x, cw, base.x),
      y: toPx(normalizedTransform.y, ch, base.y),
      width: toPx(normalizedTransform.width, cw, base.width),
      height: toPx(normalizedTransform.height, ch, base.height),
    };
  }

  private hexToRgba(hex: string, opacity: number): string {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, opacity / 100))})`;
  }

  private createRoundedRegularPolygon(
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

    return new Shape({
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
  }
}

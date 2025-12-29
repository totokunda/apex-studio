import { ClipTransform, MaskClipProps } from "@/lib/types";
import { WebGLMaskBase } from "./base";

// Reuse the same fullscreen quad shaders as lasso, but support multiple contours
const vertexShader = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;

  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// WebGL1-friendly fragment shader with packed point/contour data
const fragmentShader = `
  precision mediump float;

  uniform sampler2D u_image;
  uniform sampler2D u_pointsTex;   // Packed all points across all contours
  uniform sampler2D u_contoursTex; // Packed (start,count) pairs per contour

  // Colors are 0..1
  uniform vec3 u_backgroundColor;
  uniform float u_backgroundOpacity;
  uniform vec3 u_maskColor;
  uniform float u_maskOpacity;

  uniform int u_backgroundEnabled; // 0 or 1
  uniform int u_maskEnabled;       // 0 or 1

  // Canvas size in pixels
  uniform vec2 u_canvasSize;

  // Points texture size (width = number of points, height = 1)
  uniform float u_pointsTexWidth;
  // Contours texture size (width = number of contours, height = 1)
  uniform float u_contoursTexWidth;
  uniform int u_numPoints;
  uniform int u_numContours;

  varying vec2 v_texCoord;

  // Decode a point from a packed RGBA texel where:
  // R = x high byte, G = x low byte, B = y high byte, A = y low byte
  // Coordinates are normalized to 0..1 relative to canvas size before packing
  vec2 getPoint(int idx) {
    float fx = (float(idx) + 0.5) / max(u_pointsTexWidth, 1.0);
    vec4 t = texture2D(u_pointsTex, vec2(fx, 0.5));
    float xh = t.r * 255.0;
    float xl = t.g * 255.0;
    float yh = t.b * 255.0;
    float yl = t.a * 255.0;
    float xNorm = (xh * 256.0 + xl) / 65535.0;
    float yNorm = (yh * 256.0 + yl) / 65535.0;
    return vec2(xNorm * u_canvasSize.x, yNorm * u_canvasSize.y);
  }

  // Decode (start,count) for a contour index from a packed RGBA texel where:
  // R = start high, G = start low, B = count high, A = count low
  void getContourInfo(int contourIdx, out int startIdx, out int count) {
    float fx = (float(contourIdx) + 0.5) / max(u_contoursTexWidth, 1.0);
    vec4 t = texture2D(u_contoursTex, vec2(fx, 0.5));
    float startF = floor(t.r * 255.0 + 0.5) * 256.0 + floor(t.g * 255.0 + 0.5);
    float countF = floor(t.b * 255.0 + 0.5) * 256.0 + floor(t.a * 255.0 + 0.5);
    startIdx = int(startF);
    count = int(countF);
  }

  bool pointInPolygonRange(vec2 p, int startIdx, int count) {
    if (count < 3) return false;

    // Initialize with last vertex so edges wrap around
    int lastIndex = startIdx + count - 1;
    vec2 prev = getPoint(lastIndex);
    bool inside = false;

    const int MAX_POINTS = 4096; // same cap as lasso
    for (int i = 0; i < MAX_POINTS; i++) {
      if (i >= count) break;
      int currentIndex = startIdx + i;
      vec2 cur = getPoint(currentIndex);
      bool cond = ((cur.y > p.y) != (prev.y > p.y));
      if (cond) {
        float xIntersect = (prev.x - cur.x) * (p.y - cur.y) / ((prev.y - cur.y) + 0.000001) + cur.x;
        if (p.x < xIntersect) {
          inside = !inside;
        }
      }
      prev = cur;
    }
    return inside;
  }

  void main() {
    // Derive pixel coords from fragment position; map to top-left origin
    vec2 pixelCoord = vec2(gl_FragCoord.x, u_canvasSize.y - gl_FragCoord.y);

    bool insideAny = false;

    const int MAX_CONTOURS = 1024; // conservative cap
    for (int ci = 0; ci < MAX_CONTOURS; ci++) {
      if (ci >= u_numContours) break;
      int startIdx; int count;
      getContourInfo(ci, startIdx, count);
      if (count >= 3) {
        if (pointInPolygonRange(pixelCoord, startIdx, count)) {
          insideAny = true;
          break; // union semantics: inside if inside any contour
        }
      }
    }

    vec4 src = texture2D(u_image, v_texCoord);
    vec4 outColor = src;

    if (insideAny) {
      if (u_maskEnabled == 1) {
        outColor = vec4(u_maskColor, clamp(u_maskOpacity, 0.0, 1.0));
      } else {
        outColor = src;
      }
    } else {
      if (u_backgroundEnabled == 1) {
        outColor = vec4(u_backgroundColor, clamp(u_backgroundOpacity, 0.0, 1.0));
      } else {
        outColor = src;
      }
    }

    gl_FragColor = outColor;
  }
`;

export const getCropOffset = (
  transform?: ClipTransform | null,
): { offsetX: number; offsetY: number } => {
  if (!transform) {
    return { offsetX: 0, offsetY: 0 };
  }

  const anyTransform = transform as any;
  const crop = anyTransform?.crop as
    | { x?: number; y?: number; width?: number; height?: number }
    | undefined;

  const hasValidCrop =
    crop &&
    Number.isFinite(crop.x) &&
    Number.isFinite(crop.y) &&
    Number.isFinite(transform.width) &&
    Number.isFinite(transform.height);

  if (!hasValidCrop) {
    return { offsetX: 0, offsetY: 0 };
  }

  const cropW = crop!.width && crop!.width > 0 ? crop!.width : 1;
  const cropH = crop!.height && crop!.height > 0 ? crop!.height : 1;

  const baseWidth = (transform.width as number) || 0;
  const baseHeight = (transform.height as number) || 0;
  const offsetX = (baseWidth / cropW) * (crop!.x as number);
  const offsetY = (baseHeight / cropH) * (crop!.y as number);

  return {
    offsetX,
    offsetY,
  };
};

const applyClipTransformContours = (
  contours: number[][],
  originalClipTransform?: ClipTransform,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): number[][] => {
  if (!clipTransform) return contours;
  if (!maskTransform) return contours;

  // Mirror ShapeMask semantics: compute a scale ratio and a delta
  // between mask and original clip, then subtract any mask crop from that delta.
  const baseScaleX = maskTransform.scaleX || 1;
  const baseScaleY = maskTransform.scaleY || 1;
  const scaleRatioX = (clipTransform.scaleX || 1) / baseScaleX;
  const scaleRatioY = (clipTransform.scaleY || 1) / baseScaleY;

  let deltaX = maskTransform.x - (originalClipTransform?.x ?? 0);
  let deltaY = maskTransform.y - (originalClipTransform?.y ?? 0);

  if (maskTransform.crop) {
    const fullWidth =
      (maskTransform.width || 0) / (maskTransform.crop.width || 1);
    const fullHeight =
      (maskTransform.height || 0) / (maskTransform.crop.height || 1);
    const cropX = fullWidth * (maskTransform.crop.x || 0);
    const cropY = fullHeight * (maskTransform.crop.y || 0);
    deltaX -= cropX;
    deltaY -= cropY;
  }

  const transformed: number[][] = [];

  // Clip crop (same style as ShapeMask.applyClipTransform)
  let hasCrop = false;
  let cropX = 0;
  let cropY = 0;
  let cropW = 1;
  let cropH = 1;
  let displayWidth = 0;
  let displayHeight = 0;

  if (clipTransform.crop) {
    hasCrop = true;
    cropX = clipTransform.crop.x;
    cropY = clipTransform.crop.y;
    cropW = clipTransform.crop.width;
    cropH = clipTransform.crop.height;

    displayWidth = Math.abs(
      (clipTransform.width || 0) * (clipTransform.scaleX || 1),
    );
    displayHeight = Math.abs(
      (clipTransform.height || 0) * (clipTransform.scaleY || 1),
    );
  }

  for (let c = 0; c < contours.length; c++) {
    const contour = contours[c];
    const newPts: number[] = [];
    for (let i = 0; i < contour.length; i += 2) {
      let x: number;
      let y: number;

      // Local to mask, then shifted by delta into clip space and scaled
      x = (contour[i] - maskTransform.x + deltaX) * scaleRatioX;
      y = (contour[i + 1] - maskTransform.y + deltaY) * scaleRatioY;

      if (hasCrop) {
        x = cropX * displayWidth + x * cropW;
        y = cropY * displayHeight + y * cropH;
      }

      newPts.push(x, y);
    }
    transformed.push(newPts);
  }
  return transformed;
};

const computeCanvasScale = (
  canvas: HTMLCanvasElement,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): { scaleX: number; scaleY: number } => {
  const baseScaleX = clipTransform?.scaleX ?? maskTransform?.scaleX ?? 1;
  const baseScaleY = clipTransform?.scaleY ?? maskTransform?.scaleY ?? 1;
  const baseWidth =
    (clipTransform?.width ?? maskTransform?.width ?? canvas.width) * baseScaleX;
  const baseHeight =
    (clipTransform?.height ?? maskTransform?.height ?? canvas.height) *
    baseScaleY;

  const scaleX = baseWidth !== 0 ? canvas.width / baseWidth : 1;
  const scaleY = baseHeight !== 0 ? canvas.height / baseHeight : 1;

  return { scaleX, scaleY };
};

export class TouchMask extends WebGLMaskBase {
  private program: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private texcoordBuffer: WebGLBuffer | null = null;
  private aPositionLoc = -1;
  private aTexCoordLoc = -1;

  constructor(contextKey?: string) {
    super(contextKey);
    this.initResources();
  }

  private initResources() {
    const gl = this.ensureContext();
    if (!gl || (typeof gl.isContextLost === "function" && gl.isContextLost())) {
      console.error("No WebGL context in TouchMask or context is lost");
      return;
    }
    this.program = this.createProgram(vertexShader, fragmentShader);
    if (!this.program) {
      console.error("Failed to create shader program");
    }
    this.initQuadBuffers();
  }

  protected onContextLost(): void {
    super.onContextLost();
    this.program = null;
    this.positionBuffer = null;
    this.texcoordBuffer = null;
  }

  protected onContextRestored(): void {
    this.initResources();
  }

  private initQuadBuffers() {
    const gl = this.ensureContext();
    if (!gl) return;

    // Full-screen quad in clip space for TRIANGLE_STRIP
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const texcoords = new Float32Array([
      // Flip v in the buffer so the texture renders upright without shader/image flips
      0,
      1, 1, 1, 0, 0, 1, 0,
    ]);

    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    this.texcoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texcoords, gl.STATIC_DRAW);

    // leave ARRAY_BUFFER unbound
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private hexToRgb(hex: string): { r: number; g: number; b: number } {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m
      ? {
          r: parseInt(m[1], 16) / 255,
          g: parseInt(m[2], 16) / 255,
          b: parseInt(m[3], 16) / 255,
        }
      : { r: 0, g: 0, b: 0 };
  }

  public dispose() {
    const gl = this.gl;
    if (gl) {
      if (this.program) gl.deleteProgram(this.program);
      if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
      if (this.texcoordBuffer) gl.deleteBuffer(this.texcoordBuffer);
    }
    this.program = null;
    this.positionBuffer = null;
    this.texcoordBuffer = null;
    super.dispose();
  }
  

  public apply(
    sourceCanvas: HTMLCanvasElement,
    mask: MaskClipProps,
    frame: number,
    clipTransform?: ClipTransform,
    originalClipTransform?: ClipTransform,
    maskTransform?: ClipTransform,
    debug?: { download?: boolean; annotateBounds?: boolean; filename?: string },
  ): HTMLCanvasElement {
    if (!this.gl || !this.program) {
      console.error("No GL context or program in apply");
      return sourceCanvas;
    }

    if (this.gl.isContextLost()) {
      console.error("WebGL context is lost!");
      return sourceCanvas;
    }


    // Resolve keyframe
    let keyFrame = frame;
    if (Object.keys(mask.keyframes).length === 1) {
      keyFrame = Number(Object.keys(mask.keyframes)[0]);
    } else {
      const ks =
        mask.keyframes instanceof Map
          ? Array.from(mask.keyframes.keys()).sort((a, b) => a - b)
          : Object.keys(mask.keyframes)
              .map(Number)
              .sort((a, b) => a - b);
      if (ks.length) {
        keyFrame =
          frame < ks[0]
            ? ks[0]
            : (ks.filter((k) => k <= frame).pop() ?? ks[ks.length - 1]);
      }
    }

    const keyFrameData =
      mask.keyframes instanceof Map
        ? mask.keyframes.get(keyFrame)
        : (mask.keyframes as any)[keyFrame];
    if (
      !keyFrameData ||
      !keyFrameData.contours ||
      keyFrameData.contours.length === 0
    )
      return sourceCanvas;

    const rawContours: number[][] = keyFrameData.contours as number[][];
    const transformedContours = applyClipTransformContours(
      rawContours,
      originalClipTransform,
      clipTransform,
      maskTransform,
    );

    const { scaleX: canvasScaleX, scaleY: canvasScaleY } = computeCanvasScale(
      sourceCanvas,
      clipTransform,
      maskTransform,
    );

    // Flatten points and compute per-contour metadata
    const contours: number[][] = transformedContours
      .filter((c) => Array.isArray(c) && c.length >= 6)
      .map((contour) => {
        const scaled: number[] = [];
        for (let i = 0; i < contour.length; i += 2) {
          scaled.push(contour[i] * canvasScaleX, contour[i + 1] * canvasScaleY);
        }
        return scaled;
      });
    const numContours = contours.length;
    if (numContours === 0) return sourceCanvas;

    const starts: number[] = [];
    const counts: number[] = [];
    const allPoints: number[] = [];
    let running = 0;
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (let ci = 0; ci < numContours; ci++) {
      const contour = contours[ci];
      const count = Math.floor(contour.length / 2);
      if (count < 3) continue;
      starts.push(running);
      counts.push(count);
      for (let i = 0; i < contour.length; i += 2) {
        const x = contour[i];
        const y = contour[i + 1];
        allPoints.push(x, y);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      running += count;
    }

    const numPoints = Math.floor(allPoints.length / 2);
    if (numPoints < 3) return sourceCanvas;

    // Normalize inputs
    const backgroundColorHex = mask.backgroundColor || "#000000";
    const maskColorHex = mask.maskColor || "#ffffff";
    const backgroundOpacity = Math.max(
      0,
      Math.min(1, (mask.backgroundOpacity ?? 100) / 100),
    );
    const maskOpacity = Math.max(
      0,
      Math.min(1, (mask.maskOpacity ?? 100) / 100),
    );
    const backgroundEnabled = !!(mask.backgroundColorEnabled ?? false);
    const maskEnabled = !!(mask.maskColorEnabled ?? false);

    // Resize and setup
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, sourceCanvas.width, sourceCanvas.height);

    const texture = this.createTextureFromCanvas(sourceCanvas);
    if (!texture) {
      console.error("Failed to create texture from source canvas");
      return sourceCanvas;
    }

    // Build packed points texture (RGBA8, width=numPoints, height=1)
    const pointsTex = this.gl.createTexture();
    if (!pointsTex) {
      console.error("Failed to create points texture");
      this.gl.deleteTexture(texture);
      return sourceCanvas;
    }
    const pointsTexWidth = Math.max(1, numPoints);
    const pointsData = new Uint8Array(pointsTexWidth * 4);
    const W = sourceCanvas.width;
    const H = sourceCanvas.height;
    for (let i = 0; i < numPoints; i++) {
      const x = Math.max(0, Math.min(W, allPoints[i * 2]));
      const y = Math.max(0, Math.min(H, allPoints[i * 2 + 1]));
      const xn = x / (W || 1);
      const yn = y / (H || 1);
      const xi = Math.max(0, Math.min(65535, Math.round(xn * 65535)));
      const yi = Math.max(0, Math.min(65535, Math.round(yn * 65535)));
      const o = i * 4;
      pointsData[o + 0] = (xi >> 8) & 0xff;
      pointsData[o + 1] = xi & 0xff;
      pointsData[o + 2] = (yi >> 8) & 0xff;
      pointsData[o + 3] = yi & 0xff;
    }
    this.gl.bindTexture(this.gl.TEXTURE_2D, pointsTex);
    this.gl.pixelStorei(this.gl.UNPACK_FLIP_Y_WEBGL, false);
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_S,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_T,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MIN_FILTER,
      this.gl.NEAREST,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MAG_FILTER,
      this.gl.NEAREST,
    );
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      pointsTexWidth,
      1,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      pointsData,
    );

    // Build contours texture (RGBA8, width=numContours, height=1) with (start,count)
    const contoursTex = this.gl.createTexture();
    if (!contoursTex) {
      console.error("Failed to create contours texture");
      this.gl.deleteTexture(texture);
      this.gl.deleteTexture(pointsTex);
      return sourceCanvas;
    }
    const contoursTexWidth = Math.max(1, numContours);
    const contoursData = new Uint8Array(contoursTexWidth * 4);
    for (let ci = 0; ci < numContours; ci++) {
      const start = Math.max(0, Math.min(65535, starts[ci] ?? 0));
      const count = Math.max(0, Math.min(65535, counts[ci] ?? 0));
      const o = ci * 4;
      contoursData[o + 0] = (start >> 8) & 0xff; // R high
      contoursData[o + 1] = start & 0xff; // G low
      contoursData[o + 2] = (count >> 8) & 0xff; // B high
      contoursData[o + 3] = count & 0xff; // A low
    }
    this.gl.bindTexture(this.gl.TEXTURE_2D, contoursTex);
    this.gl.pixelStorei(this.gl.UNPACK_FLIP_Y_WEBGL, false);
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_S,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_WRAP_T,
      this.gl.CLAMP_TO_EDGE,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MIN_FILTER,
      this.gl.NEAREST,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MAG_FILTER,
      this.gl.NEAREST,
    );
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      contoursTexWidth,
      1,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      contoursData,
    );

    this.gl.useProgram(this.program);

    // attributes
    this.aPositionLoc = this.gl.getAttribLocation(this.program, "a_position");
    this.aTexCoordLoc = this.gl.getAttribLocation(this.program, "a_texCoord");
    if (this.aPositionLoc === -1 || this.aTexCoordLoc === -1) {
      console.error("Failed to get attribute locations");
      this.gl.deleteTexture(texture);
      this.gl.deleteTexture(pointsTex);
      this.gl.deleteTexture(contoursTex);
      return sourceCanvas;
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
    this.gl.enableVertexAttribArray(this.aPositionLoc);
    this.gl.vertexAttribPointer(
      this.aPositionLoc,
      2,
      this.gl.FLOAT,
      false,
      0,
      0,
    );
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);
    this.gl.enableVertexAttribArray(this.aTexCoordLoc);
    this.gl.vertexAttribPointer(
      this.aTexCoordLoc,
      2,
      this.gl.FLOAT,
      false,
      0,
      0,
    );

    const bg = this.hexToRgb(backgroundColorHex);
    const mk = this.hexToRgb(maskColorHex);

    // Bind samplers: 0=image, 1=points, 2=contours
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.uniform1i(this.gl.getUniformLocation(this.program, "u_image"), 0);

    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, pointsTex);
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_pointsTex"),
      1,
    );

    this.gl.activeTexture(this.gl.TEXTURE2);
    this.gl.bindTexture(this.gl.TEXTURE_2D, contoursTex);
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_contoursTex"),
      2,
    );

    this.gl.uniform3f(
      this.gl.getUniformLocation(this.program, "u_backgroundColor"),
      bg.r,
      bg.g,
      bg.b,
    );
    this.gl.uniform1f(
      this.gl.getUniformLocation(this.program, "u_backgroundOpacity"),
      backgroundOpacity,
    );
    this.gl.uniform3f(
      this.gl.getUniformLocation(this.program, "u_maskColor"),
      mk.r,
      mk.g,
      mk.b,
    );
    this.gl.uniform1f(
      this.gl.getUniformLocation(this.program, "u_maskOpacity"),
      maskOpacity,
    );
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_backgroundEnabled"),
      backgroundEnabled ? 1 : 0,
    );
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_maskEnabled"),
      maskEnabled ? 1 : 0,
    );
    this.gl.uniform2f(
      this.gl.getUniformLocation(this.program, "u_canvasSize"),
      sourceCanvas.width,
      sourceCanvas.height,
    );
    this.gl.uniform1f(
      this.gl.getUniformLocation(this.program, "u_pointsTexWidth"),
      pointsTexWidth,
    );
    this.gl.uniform1f(
      this.gl.getUniformLocation(this.program, "u_contoursTexWidth"),
      contoursTexWidth,
    );
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_numPoints"),
      numPoints,
    );
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_numContours"),
      numContours,
    );

    // Blend as in lasso
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    if (debug?.annotateBounds) {
      this.debugAnnotateBoundsOnTop(sourceCanvas, {
        x: minX,
        y: minY,
        width: Math.max(0, maxX - minX),
        height: Math.max(0, maxY - minY),
      });
    }

    if (debug?.download) {
      const safeName =
        debug.filename ??
        `touch-mask-debug-f${frame}-nC${numContours}-nP${numPoints}.png`;
      this.debugDownloadCanvas(safeName);
    }

    // cleanup
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    this.gl.activeTexture(this.gl.TEXTURE2);
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    this.gl.deleteTexture(texture);
    this.gl.deleteTexture(pointsTex);
    this.gl.deleteTexture(contoursTex);

    return this.canvas;
  }
}

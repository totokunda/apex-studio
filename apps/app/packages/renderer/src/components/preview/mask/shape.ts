import { WebGLMaskBase } from "./base";
import { ClipTransform, MaskClipProps, MaskShapeTool } from "@/lib/types";

const vertexShader = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;

  void main() {
    // a_position is already in clip space (-1..1)
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Note: use int flags (0/1) instead of bool uniforms for WebGL1
const fragmentShader = `
  precision mediump float;

  uniform sampler2D u_image;

  // x, y, width, height in pixels (Canvas top-left origin)
  uniform vec4 u_shapeBounds;

  // Colors are 0..1
  uniform vec3 u_backgroundColor;
  uniform float u_backgroundOpacity;

  uniform vec3 u_maskColor;
  uniform float u_maskOpacity;

  uniform int u_backgroundEnabled; // 0 or 1
  uniform int u_maskEnabled;       // 0 or 1

  // 0 = rectangle, 1 = ellipse, 2 = triangle (upward), 3 = star (5-point)
  uniform int u_shapeKind;

  // Canvas size in pixels
  uniform vec2 u_canvasSize;
  // Canvas-to-mask scale conversion factors (applied before rotation)
  uniform vec2 u_canvasScale;
  // Per-shape non-uniform scale factors (1 = no scaling)
  uniform vec2 u_shapeScale;
  // Rotation in radians (clockwise around shape center)
  uniform float u_rotation;

  varying vec2 v_texCoord;

  // Point-in-triangle via barycentric technique
  bool pointInTriangle(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = c - a;
    vec2 v1 = b - a;
    vec2 v2 = p - a;
    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);
    float denom = dot00 * dot11 - dot01 * dot01;
    if (denom == 0.0) {
      return false;
    }
    float invDenom = 1.0 / denom;
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0);
  }

  void main() {
    // Derive pixel coords from fragment position; map to top-left origin
    vec2 pixelCoord = vec2(gl_FragCoord.x, u_canvasSize.y - gl_FragCoord.y);

    // Precompute scale and rotation helpers
    float shapeScaleX = max(u_shapeScale.x, 0.0001);
    float shapeScaleY = max(u_shapeScale.y, 0.0001);
    
    // We work in screen pixels to avoid rotation distortion from non-uniform canvas scaling.
    // u_shapeBounds are already in screen pixels.

    float rectWidthMask = max(u_shapeBounds.z * shapeScaleX, 0.0001);
    float rectHeightMask = max(u_shapeBounds.w * shapeScaleY, 0.0001);
    float halfW = rectWidthMask * 0.5;
    float halfH = rectHeightMask * 0.5;

    float scaledWCanvas = u_shapeBounds.z * shapeScaleX;
    float scaledHCanvas = u_shapeBounds.w * shapeScaleY;

    float c = cos(-u_rotation);
    float s = sin(-u_rotation);

    bool insideRect = false;
    bool insideEllipse = false;
    bool insideTriangle = false;
    bool insideStar = false;

    if (u_shapeKind == 0) {
      // Rectangle rotates around its top-left corner to match editor behavior
      vec2 pivot = vec2(u_shapeBounds.x, u_shapeBounds.y);
      vec2 localCanvas = pixelCoord - pivot;
      
      // Rotate in pixel space
      vec2 rotRect = vec2(c * localCanvas.x - s * localCanvas.y,
                          s * localCanvas.x + c * localCanvas.y);
                          
      insideRect = (rotRect.x >= 0.0) && (rotRect.x <= rectWidthMask) &&
                   (rotRect.y >= 0.0) && (rotRect.y <= rectHeightMask);
    } else {
      // Center-anchored shapes rotate around their midpoint
      vec2 center = vec2(u_shapeBounds.x + 0.5 * scaledWCanvas,
                         u_shapeBounds.y + 0.5 * scaledHCanvas);
      vec2 local = pixelCoord - center;
      
      // Rotate in pixel space
      vec2 rotLocal = vec2(c * local.x - s * local.y,
                           s * local.x + c * local.y);

      insideRect = abs(rotLocal.x) <= halfW &&
                   abs(rotLocal.y) <= halfH;

      // Ellipse inclusion in rotated local space: ((x)/rx)^2 + ((y)/ry)^2 <= 1
      vec2 radii = vec2(max(halfW, 0.0001), max(halfH, 0.0001));
      vec2 d = rotLocal / radii;
      insideEllipse = dot(d, d) <= 1.0;

      // Triangle (upward) inclusion: base along bottom edge of bounds, apex at top-center
      // Define triangle vertices in rotated local space with origin at shape center
      // Shift Y by -h/6 to align centroid with bounding box center (matching Konva)
      float yOffset = -rectHeightMask / 6.0;
      vec2 tApex = vec2(0.0, -halfH + yOffset);
      vec2 tLeft = vec2(-halfW, halfH + yOffset);
      vec2 tRight = vec2(halfW, halfH + yOffset);
      insideTriangle = pointInTriangle(rotLocal, tApex, tLeft, tRight);

      // Star (5-point) inclusion using even-odd point-in-polygon computed on-the-fly
      const int MAX_VERTS = 10; // 5-point star => 10 vertices (outer/inner alternating)
      
      // Use halfW/halfH directly for radii to allow stretching (anisotropic scaling)
      float rOuterX = max(halfW, 0.0001);
      float rOuterY = max(halfH, 0.0001);
      float rInnerX = 0.5 * rOuterX;
      float rInnerY = 0.5 * rOuterY;

      // Initialize previous vertex to the last star vertex (index 9, which is inner)
      float prevAngle = float(MAX_VERTS - 1) * 3.14159265358979323846 / 5.0;
      // Check if MAX_VERTS-1 is even or odd. 9 is odd -> inner radius.
      // But loop logic below: i=0 (even) uses Outer. i=9 (odd) uses Inner.
      // So init with Inner.
      vec2 prevVertex = vec2(rInnerX * sin(prevAngle), -rInnerY * cos(prevAngle));
      
      // Even-odd ray casting without array indexing
      bool useOuter = true; // i=0 uses outer radius
      for (int i = 0; i < MAX_VERTS; i++) {
        float rx = useOuter ? rOuterX : rInnerX;
        float ry = useOuter ? rOuterY : rInnerY;
        
        float angle = float(i) * 3.14159265358979323846 / 5.0;
        vec2 vi = vec2(rx * sin(angle), -ry * cos(angle));
        
        bool intersect = ((vi.y > rotLocal.y) != (prevVertex.y > rotLocal.y)) &&
          (rotLocal.x < (prevVertex.x - vi.x) * (rotLocal.y - vi.y) / ((prevVertex.y - vi.y) + 0.000001) + vi.x);
        if (intersect) {
          insideStar = !insideStar;
        }
        prevVertex = vi;
        useOuter = !useOuter;
      }
    }

    bool inside = (u_shapeKind == 0) ? insideRect : ((u_shapeKind == 1) ? insideEllipse : ((u_shapeKind == 2) ? insideTriangle : insideStar));

    vec4 src = texture2D(u_image, v_texCoord);
    vec4 outColor = src;

    if (inside) {
      if (u_maskEnabled == 1) {
        // Foreground mask: render color with given opacity; alpha 0 creates holes
        outColor = vec4(u_maskColor, clamp(u_maskOpacity, 0.0, 1.0));
      } else {
        outColor = src;
      }
    } else {
      if (u_backgroundEnabled == 1) {
        // Background outside mask: render color with given opacity; alpha 0 creates holes
        outColor = vec4(u_backgroundColor, clamp(u_backgroundOpacity, 0.0, 1.0));
      } else {
        outColor = src;
      }
    }

    gl_FragColor = outColor;
  }
`;

interface ShapeBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX?: number;
  scaleY?: number;
  rotation?: number;
  shapeType?: MaskShapeTool;
}

const applyClipTransform = (
  shapeBounds: ShapeBounds,
  originalClipTransform?: ClipTransform,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): ShapeBounds => {
  if (!clipTransform) return shapeBounds;

  let localX: number, localY: number, scaledWidth: number, scaledHeight: number;


  if (maskTransform) {
    const baseScaleX = maskTransform.scaleX || 1;
    const baseScaleY = maskTransform.scaleY || 1;
    const scaleRatioX = (clipTransform.scaleX || 1) / baseScaleX;
    const scaleRatioY = (clipTransform.scaleY || 1) / baseScaleY;
    let deltaX = maskTransform.x - (originalClipTransform?.x ?? 0)
    let deltaY = maskTransform.y - (originalClipTransform?.y ?? 0);
    if (maskTransform.crop) {
      let fullWidth = maskTransform.width / maskTransform.crop.width;
      let fullHeight = maskTransform.height / maskTransform.crop.height;
      let cropX = fullWidth * maskTransform.crop.x;
      let cropY = fullHeight * maskTransform.crop.y;
      deltaX -= cropX;
      deltaY -= cropY;
    }
    if (originalClipTransform) {
       localX = (shapeBounds.x - maskTransform.x + deltaX) * scaleRatioX;
       localY = (shapeBounds.y - maskTransform.y + deltaY) * scaleRatioY;
    } else {
      localX = (shapeBounds.x) * scaleRatioX;
      localY = (shapeBounds.y) * scaleRatioY;
    }


    scaledWidth = shapeBounds.width * scaleRatioX;
    scaledHeight = shapeBounds.height * scaleRatioY;
  } else {
    localX = shapeBounds.x;
    localY = shapeBounds.y;
    scaledWidth = shapeBounds.width;
    scaledHeight = shapeBounds.height;
  }

  if (clipTransform && clipTransform.crop) {
    const cropX = clipTransform.crop.x;
    const cropY = clipTransform.crop.y;
    const cropW = clipTransform.crop.width;
    const cropH = clipTransform.crop.height;

    const displayWidth = Math.abs(
      (clipTransform.width || 0) * (clipTransform.scaleX || 1),
    );
    const displayHeight = Math.abs(
      (clipTransform.height || 0) * (clipTransform.scaleY || 1),
    );

    localX = cropX * displayWidth + localX * cropW;
    localY = cropY * displayHeight + localY * cropH;
    scaledWidth *= cropW;
    scaledHeight *= cropH;
  }
  const newBounds: ShapeBounds = {
    x:localX,
    y:localY,
    width: scaledWidth,
    height: scaledHeight,
    scaleX: shapeBounds.scaleX,
    scaleY: shapeBounds.scaleY,
    rotation: shapeBounds.rotation,
    shapeType: shapeBounds.shapeType,
  };

  return newBounds;
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

export class ShapeMask extends WebGLMaskBase {
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
      console.error("No WebGL context in ShapeMask or context is lost");
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


    // Check if context is lost
    if (this.gl.isContextLost()) {
      console.error("WebGL context is lost!");
      return sourceCanvas;
    }

    // Resolve keyframe to use
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
        : mask.keyframes[keyFrame];
    if (!keyFrameData || !keyFrameData.shapeBounds) return sourceCanvas;

    //const { x, y, width, height } = keyFrameData.shapeBounds;
    const transformedBounds = applyClipTransform(
      keyFrameData.shapeBounds,
      originalClipTransform,
      clipTransform,
      maskTransform,
    );
    const { scaleX: canvasScaleX, scaleY: canvasScaleY } = computeCanvasScale(
      sourceCanvas,
      clipTransform,
      maskTransform,
    );
    const x = transformedBounds.x * canvasScaleX;
    const y = transformedBounds.y * canvasScaleY;
    const width = transformedBounds.width * canvasScaleX;
    const height = transformedBounds.height * canvasScaleY;

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
    // Resize the destination GL canvas and viewport
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Ensure we're rendering to the default framebuffer (the canvas)
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, sourceCanvas.width, sourceCanvas.height);

    // Source texture
    const texture = this.createTextureFromCanvas(sourceCanvas);
    if (!texture) {
      console.error("Failed to create texture from source canvas");
      return sourceCanvas;
    }

    this.gl.useProgram(this.program);

    // ---- attributes ----
    this.aPositionLoc = this.gl.getAttribLocation(this.program, "a_position");
    this.aTexCoordLoc = this.gl.getAttribLocation(this.program, "a_texCoord");

    if (this.aPositionLoc === -1 || this.aTexCoordLoc === -1) {
      console.error("Failed to get attribute locations");
      return sourceCanvas;
    }

    // position
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

    // texcoord
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

    // ---- uniforms ----
    const bg = this.hexToRgb(backgroundColorHex);
    const mk = this.hexToRgb(maskColorHex);

    // texture sampler
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.uniform1i(this.gl.getUniformLocation(this.program, "u_image"), 0);

    // bounds (pixels)
    this.gl.uniform4f(
      this.gl.getUniformLocation(this.program, "u_shapeBounds"),
      x,
      y,
      width,
      height,
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

    // flags
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_backgroundEnabled"),
      backgroundEnabled ? 1 : 0,
    );
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_maskEnabled"),
      maskEnabled ? 1 : 0,
    );

    // shape kind: 0 rectangle, 1 ellipse, 2 triangle (polygon), 3 star
    let shapeKind = 0;
    if (keyFrameData.shapeBounds.shapeType === "ellipse") {
      shapeKind = 1;
    } else if (keyFrameData.shapeBounds.shapeType === "polygon") {
      shapeKind = 2;
    } else if (keyFrameData.shapeBounds.shapeType === "star") {
      shapeKind = 3;
    }
    this.gl.uniform1i(
      this.gl.getUniformLocation(this.program, "u_shapeKind"),
      shapeKind,
    );

    // rotation in radians (default 0). shapeBounds.rotation is in degrees? Assuming degrees, convert.
    const rotDeg = keyFrameData.shapeBounds.rotation ?? 0;

    const rotRad = (rotDeg * Math.PI) / 180;

    this.gl.uniform1f(
      this.gl.getUniformLocation(this.program, "u_rotation"),
      rotRad,
    );

    // per-shape scale (default 1). When provided, scales bounds about center in shader
    const sx = keyFrameData.shapeBounds.scaleX ?? 1;
    const sy = keyFrameData.shapeBounds.scaleY ?? 1;
    this.gl.uniform2f(
      this.gl.getUniformLocation(this.program, "u_shapeScale"),
      sx,
      sy,
    );

    // canvas scaling (used to undo non-uniform scaling before rotation tests)
    this.gl.uniform2f(
      this.gl.getUniformLocation(this.program, "u_canvasScale"),
      canvasScaleX,
      canvasScaleY,
    );

    // canvas size (pixels)
    this.gl.uniform2f(
      this.gl.getUniformLocation(this.program, "u_canvasSize"),
      sourceCanvas.width,
      sourceCanvas.height,
    );

    // No origin flip needed anymore; pixel coords are derived from gl_FragCoord
    // Leave any previous uniform untouched (shader no longer declares it)

    // Enable blending so alpha from mask/background produces holes (transparent)
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    // Clear to transparent so holes reveal nothing behind
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // Read pixel after clear
    const clearPixels = new Uint8Array(4);
    this.gl.readPixels(
      this.canvas.width / 2,
      this.canvas.height / 2,
      1,
      1,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      clearPixels,
    );

    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    // Read a pixel from the center to verify there's content
    const pixels = new Uint8Array(4);
    this.gl.readPixels(
      this.canvas.width / 2,
      this.canvas.height / 2,
      1,
      1,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      pixels,
    );

    if (debug?.annotateBounds) {
      // overlay a dashed red rectangle where the shader thinks the bounds are
      this.debugAnnotateBoundsOnTop(sourceCanvas, { x, y, width, height });
    }

    if (debug?.download) {
      const safeName =
        debug.filename ??
        `shape-mask-debug-f${frame}-x${x}-y${y}-w${width}-h${height}.png`;
      this.debugDownloadCanvas(safeName);
    }

    // cleanup bindings
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    this.gl.deleteTexture(texture);

    return this.canvas;
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
}

/**
 * Base class for WebGL filters and masks
 * Provides common functionality for shader compilation, texture management, and rendering
 */

import {
  WebGLContextListener,
  WebGLContextManager,
  WebGLSharedContextHandle,
} from "../webgl/WebGLContextManager";

export abstract class WebGLMaskBase {
  protected gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  protected canvas: HTMLCanvasElement;
  protected vertexBuffer: WebGLBuffer | null = null;
  protected texCoordBuffer: WebGLBuffer | null = null;

  private readonly contextHandle: WebGLSharedContextHandle;
  private unsubscribeFromContext?: () => void;
  private bufferContext: WebGLRenderingContext | WebGL2RenderingContext | null =
    null;

  constructor(contextKey?: string, contextType: "webgl" | "webgl2" = "webgl") {
    // Use a shared default key when not provided, or an instance-specific key for isolation
    this.contextHandle = WebGLContextManager.acquire(
      contextKey || "preview-webgl-mask",
      {
        contextType,
        attributes: {
          premultipliedAlpha: false,
          preserveDrawingBuffer: true,
        },
      },
    );

    this.canvas = this.contextHandle.canvas;
    this.gl = this.contextHandle.ensureContext();

    if (!this.gl) {
      console.error("Failed to initialize WebGL context");
    } else {
      this.initBuffers();
    }

    const listener: WebGLContextListener = {
      onContextLost: () => {
        this.bufferContext = null;
        this.vertexBuffer = null;
        this.texCoordBuffer = null;
        this.onContextLost();
      },
      onContextRestored: (gl) => {
        this.gl = gl;
        this.bufferContext = null;
        this.vertexBuffer = null;
        this.texCoordBuffer = null;
        this.initBuffers();
        this.onContextRestored();
      },
    };

    this.unsubscribeFromContext = this.contextHandle.subscribe(listener);
  }

  protected ensureContext():
    | WebGLRenderingContext
    | WebGL2RenderingContext
    | null {
    const gl = this.contextHandle.ensureContext();
    if (gl && gl !== this.gl) {
      this.gl = gl;
      this.bufferContext = null;
      this.vertexBuffer = null;
      this.texCoordBuffer = null;
    }
    return this.gl;
  }

  // Subclasses can override these to handle context loss/restoration
  protected onContextLost() {
    // Default: do nothing
  }

  protected onContextRestored() {
    // Default: do nothing
  }

  protected initBuffers() {
    const gl = this.ensureContext();
    if (!gl) return;

    // Full-screen quad vertices
    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    // Texture coordinates without Y-flip to prevent upside-down rendering
    const texCoords = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);

    this.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    this.texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.bufferContext = gl;
  }

  protected createShader(type: number, source: string): WebGLShader | null {
    const gl = this.ensureContext();
    if (!gl) return null;

    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  protected createProgram(
    vertexSource: string,
    fragmentSource: string,
  ): WebGLProgram | null {
    const gl = this.ensureContext();
    if (!gl) return null;

    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(
      gl.FRAGMENT_SHADER,
      fragmentSource,
    );

    if (!vertexShader || !fragmentShader) return null;

    const program = gl.createProgram();
    if (!program) return null;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program link error:", gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }

    gl.detachShader(program, vertexShader);
    gl.detachShader(program, fragmentShader);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    return program;
  }

  protected createTextureFromCanvas(
    canvas: HTMLCanvasElement,
  ): WebGLTexture | null {
    const gl = this.ensureContext();
    if (!gl) return null;

    const texture = gl.createTexture();
    if (!texture) return null;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Use shader-based flip (u_originTopLeft) instead of flipping uploads
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);

    return texture;
  }

  protected setupAttributes(program: WebGLProgram) {
    const gl = this.ensureContext();
    if (!gl) return;

    if (!this.vertexBuffer || !this.texCoordBuffer) {
      this.initBuffers();
    }
    if (!this.vertexBuffer || !this.texCoordBuffer) {
      return;
    }

    const positionLocation = gl.getAttribLocation(program, "a_position");
    const texCoordLocation = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);
  }

  protected resizeCanvas(width: number, height: number) {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }

    const gl = this.ensureContext();
    if (gl) {
      gl.viewport(0, 0, width, height);
    }
  }

  public dispose() {
    const gl = this.gl;
    if (gl && this.vertexBuffer && this.bufferContext === gl) {
      gl.deleteBuffer(this.vertexBuffer);
    }
    if (gl && this.texCoordBuffer && this.bufferContext === gl) {
      gl.deleteBuffer(this.texCoordBuffer);
    }

    this.vertexBuffer = null;
    this.texCoordBuffer = null;
    this.bufferContext = null;

    if (this.unsubscribeFromContext) {
      this.unsubscribeFromContext();
      this.unsubscribeFromContext = undefined;
    }

    this.contextHandle.release();
    this.gl = null;
  }

  // Abstract method that subclasses must implement
  public abstract apply(
    sourceCanvas: HTMLCanvasElement,
    ...params: any[]
  ): HTMLCanvasElement;

  protected debugDownloadCanvas(filename = "shape-mask-debug.png") {
    if (!this.canvas) return;
    // Prefer toBlob (async, crisp); fallback to dataURL if not supported.
    if ("toBlob" in this.canvas) {
      (this.canvas as HTMLCanvasElement).toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }, "image/png");
    } else {
      const url = (this.canvas as HTMLCanvasElement).toDataURL("image/png");
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  }

  /**
   * (Optional) Draw a debug rectangle over the GL output so we can
   * visually confirm the interpreted bounds. This uses a 2D temp canvas
   * and then writes back into our GL canvas.
   */
  protected debugAnnotateBoundsOnTop(
    srcCanvas: HTMLCanvasElement,
    bounds: { x: number; y: number; width: number; height: number },
  ) {
    const { width: W, height: H } = srcCanvas;
    const tmp = document.createElement("canvas");
    tmp.width = W;
    tmp.height = H;
    const ctx = tmp.getContext("2d");
    if (!ctx) return;

    // draw GL result first
    ctx.drawImage(this.canvas, 0, 0);

    // draw bounds (top-left origin)
    ctx.save();
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = "rgba(255,0,0,0.9)";
    ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
    ctx.restore();

    // copy annotated back into our GL canvas
    const gl2d = this.canvas.getContext("2d");
    if (gl2d) {
      gl2d.clearRect(0, 0, W, H);
      gl2d.drawImage(tmp, 0, 0);
    }
  }
}

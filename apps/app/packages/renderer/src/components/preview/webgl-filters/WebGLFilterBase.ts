/**
 * Base class for WebGL filters
 * Provides common functionality for shader compilation, texture management, and rendering
 */

import {
  WebGLContextListener,
  WebGLContextManager,
  WebGLSharedContextHandle,
} from "../webgl/WebGLContextManager";

export abstract class WebGLFilterBase {
  protected gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  protected canvas: HTMLCanvasElement;
  protected vertexBuffer: WebGLBuffer | null = null;
  protected texCoordBuffer: WebGLBuffer | null = null;

  private readonly contextHandle: WebGLSharedContextHandle;
  private unsubscribeFromContext?: () => void;
  private bufferContext: WebGLRenderingContext | WebGL2RenderingContext | null =
    null;

  constructor(
    contextType: "webgl" | "webgl2" = "webgl",
    contextKey = "preview-webgl-filter",
  ) {
    this.contextHandle = WebGLContextManager.acquire(contextKey, {
      contextType,
      attributes: {
        premultipliedAlpha: false,
        preserveDrawingBuffer: true,
      },
    });

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

  protected onContextLost(): void {
    // Subclasses can override to tear down resources.
  }

  protected onContextRestored(): void {
    // Subclasses can override to recreate resources.
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
}

/**
 * NativeFrameUploader
 *
 * Efficiently uploads raw RGBA pixel buffers (from SharedArrayBuffers)
 * to the GPU via WebGL texImage2D and renders them to an HTMLCanvasElement.
 *
 * This avoids the overhead of constructing VideoFrame objects and leverages
 * direct GPU texture upload for maximum throughput.
 */

// Vertex shader: full-screen quad
const VERTEX_SHADER_SRC = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Fragment shader: sample texture and flip Y (WebGL Y-axis is inverted)
const FRAGMENT_SHADER_SRC = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  void main() {
    // Flip Y: WebGL textures have origin at bottom-left,
    // but our RGBA buffer has origin at top-left
    vec2 flippedCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
    gl_FragColor = texture2D(u_texture, flippedCoord);
  }
`;

export class NativeFrameUploader {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private texture: WebGLTexture;
  private currentWidth = 0;
  private currentHeight = 0;
  private disposed = false;

  constructor() {
    this.canvas = document.createElement("canvas");

    const gl = this.canvas.getContext("webgl", {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
    });

    if (!gl) {
      throw new Error("Failed to create WebGL context for frame uploader");
    }

    this.gl = gl;
    this.program = this.createProgram();
    this.texture = this.createTexture();
    this.setupGeometry();
  }

  /**
   * Upload an RGBA buffer to the GPU and render it to the internal canvas.
   * Returns the canvas (same reference each call, content updated).
   */
  upload(
    rgbaBuffer: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
  ): HTMLCanvasElement {
    if (this.disposed) {
      throw new Error("NativeFrameUploader has been disposed");
    }

    const gl = this.gl;

    // Resize canvas if dimensions changed
    if (this.currentWidth !== width || this.currentHeight !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.currentWidth = width;
      this.currentHeight = height;
      gl.viewport(0, 0, width, height);
    }

    // Upload texture data
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      rgbaBuffer,
    );

    // Draw full-screen quad
    gl.useProgram(this.program);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    return this.canvas;
  }

  /**
   * Release WebGL resources.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    const gl = this.gl;
    gl.deleteTexture(this.texture);
    gl.deleteProgram(this.program);

    // Force context loss to free GPU memory
    const ext = gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
  }

  // -------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------

  private createShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Failed to create shader");

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${log}`);
    }

    return shader;
  }

  private createProgram(): WebGLProgram {
    const gl = this.gl;
    const vs = this.createShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    const fs = this.createShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);

    const program = gl.createProgram();
    if (!program) throw new Error("Failed to create program");

    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program link error: ${log}`);
    }

    // Shaders can be deleted after linking
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return program;
  }

  private createTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");

    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set texture parameters for non-power-of-2 textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return texture;
  }

  private setupGeometry(): void {
    const gl = this.gl;
    gl.useProgram(this.program);

    // Full-screen quad (triangle strip)
    const positions = new Float32Array([
      -1, -1,  // bottom-left
       1, -1,  // bottom-right
      -1,  1,  // top-left
       1,  1,  // top-right
    ]);

    const texCoords = new Float32Array([
      0, 0,
      1, 0,
      0, 1,
      1, 1,
    ]);

    // Position buffer
    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Tex coord buffer
    const texBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texBuf);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    const texLoc = gl.getAttribLocation(this.program, "a_texCoord");
    gl.enableVertexAttribArray(texLoc);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

    // Set texture uniform
    const texUniform = gl.getUniformLocation(this.program, "u_texture");
    gl.uniform1i(texUniform, 0);
  }
}

/**
 * Shared singleton uploader instance.
 * Reused across all decoder manager assets to minimize WebGL context creation.
 */
let sharedUploader: NativeFrameUploader | null = null;

export function getSharedFrameUploader(): NativeFrameUploader {
  if (!sharedUploader) {
    sharedUploader = new NativeFrameUploader();
  }
  return sharedUploader;
}

export function disposeSharedFrameUploader(): void {
  if (sharedUploader) {
    sharedUploader.dispose();
    sharedUploader = null;
  }
}

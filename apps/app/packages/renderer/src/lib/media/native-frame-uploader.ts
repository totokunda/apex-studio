/**
 * NativeFrameUploader
 *
 * Efficiently uploads raw RGBA/NV12 pixel buffers (from SharedArrayBuffers)
 * to the GPU via WebGL textures and renders them to an HTMLCanvasElement.
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

// Fragment shader: sample RGBA texture and flip Y (WebGL Y-axis is inverted)
const RGBA_FRAGMENT_SHADER_SRC = `
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

// Fragment shader: convert NV12 (Y + interleaved UV) to RGB.
const NV12_FRAGMENT_SHADER_SRC = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_textureY;
  uniform sampler2D u_textureUV;
  void main() {
    vec2 flippedCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
    float ySample = texture2D(u_textureY, flippedCoord).r;
    vec2 uvSample = texture2D(u_textureUV, flippedCoord).ra - vec2(0.5, 0.5);

    // BT.709 limited-range conversion.
    float y = max(0.0, 1.16438356 * (ySample - 0.0625));
    float r = y + 1.79274107 * uvSample.y;
    float g = y - 0.21324861 * uvSample.x - 0.53290933 * uvSample.y;
    float b = y + 2.11240179 * uvSample.x;

    gl_FragColor = vec4(clamp(vec3(r, g, b), 0.0, 1.0), 1.0);
  }
`;

export class NativeFrameUploader {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private rgbaProgram: WebGLProgram;
  private nv12Program: WebGLProgram;
  private rgbaTexture: WebGLTexture;
  private yTexture: WebGLTexture;
  private uvTexture: WebGLTexture;
  private positionBuffer: WebGLBuffer;
  private texCoordBuffer: WebGLBuffer;
  private rgbaTextureUniform: WebGLUniformLocation | null;
  private nv12YUniform: WebGLUniformLocation | null;
  private nv12UVUniform: WebGLUniformLocation | null;
  private currentWidth = 0;
  private currentHeight = 0;
  private rgbaTextureAllocated = false;
  private yTextureAllocated = false;
  private uvTextureAllocated = false;
  private disposed = false;

  constructor(targetCanvas?: HTMLCanvasElement) {
    this.canvas = targetCanvas ?? document.createElement("canvas");

    const gl = this.canvas.getContext("webgl", {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
    });

    if (!gl) {
      throw new Error("Failed to create WebGL context for frame uploader");
    }

    this.gl = gl;
    this.rgbaProgram = this.createProgram(RGBA_FRAGMENT_SHADER_SRC);
    this.nv12Program = this.createProgram(NV12_FRAGMENT_SHADER_SRC);
    this.rgbaTexture = this.createTexture();
    this.yTexture = this.createTexture();
    this.uvTexture = this.createTexture();
    this.positionBuffer = this.createStaticBuffer(
      new Float32Array([
        -1, -1, // bottom-left
         1, -1, // bottom-right
        -1,  1, // top-left
         1,  1, // top-right
      ]),
    );
    this.texCoordBuffer = this.createStaticBuffer(
      new Float32Array([
        0, 0,
        1, 0,
        0, 1,
        1, 1,
      ]),
    );

    this.rgbaTextureUniform = gl.getUniformLocation(this.rgbaProgram, "u_texture");
    this.nv12YUniform = gl.getUniformLocation(this.nv12Program, "u_textureY");
    this.nv12UVUniform = gl.getUniformLocation(this.nv12Program, "u_textureUV");
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
    this.ensureCanvasSize(width, height);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rgbaTexture);
    if (!this.rgbaTextureAllocated) {
      // Allocate storage once when dimensions change; update contents via texSubImage2D.
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        width,
        height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
      this.rgbaTextureAllocated = true;
    }
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      width,
      height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      rgbaBuffer,
    );

    gl.useProgram(this.rgbaProgram);
    this.bindGeometry(this.rgbaProgram);
    if (this.rgbaTextureUniform) {
      gl.uniform1i(this.rgbaTextureUniform, 0);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    return this.canvas;
  }

  /**
   * Upload an NV12 buffer to the GPU and render it to the internal canvas.
   * Buffer layout: [Y plane][UV plane interleaved].
   */
  uploadNV12(
    nv12Buffer: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
  ): HTMLCanvasElement {
    if (this.disposed) {
      throw new Error("NativeFrameUploader has been disposed");
    }

    const ySize = width * height;
    const uvWidth = Math.floor((width + 1) / 2);
    const uvHeight = Math.floor((height + 1) / 2);
    const uvSize = uvWidth * uvHeight * 2;
    if (nv12Buffer.byteLength < ySize + uvSize) {
      throw new Error("NV12 buffer is too small for frame dimensions");
    }

    const gl = this.gl;
    this.ensureCanvasSize(width, height);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const src = nv12Buffer as Uint8Array;
    const yPlane = src.subarray(0, ySize);
    const uvPlane = src.subarray(ySize, ySize + uvSize);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.yTexture);
    if (!this.yTextureAllocated) {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.LUMINANCE,
        width,
        height,
        0,
        gl.LUMINANCE,
        gl.UNSIGNED_BYTE,
        null,
      );
      this.yTextureAllocated = true;
    }
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      width,
      height,
      gl.LUMINANCE,
      gl.UNSIGNED_BYTE,
      yPlane,
    );

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.uvTexture);
    if (!this.uvTextureAllocated) {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.LUMINANCE_ALPHA,
        uvWidth,
        uvHeight,
        0,
        gl.LUMINANCE_ALPHA,
        gl.UNSIGNED_BYTE,
        null,
      );
      this.uvTextureAllocated = true;
    }
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      uvWidth,
      uvHeight,
      gl.LUMINANCE_ALPHA,
      gl.UNSIGNED_BYTE,
      uvPlane,
    );

    gl.useProgram(this.nv12Program);
    this.bindGeometry(this.nv12Program);
    if (this.nv12YUniform) {
      gl.uniform1i(this.nv12YUniform, 0);
    }
    if (this.nv12UVUniform) {
      gl.uniform1i(this.nv12UVUniform, 1);
    }
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
    gl.deleteTexture(this.rgbaTexture);
    gl.deleteTexture(this.yTexture);
    gl.deleteTexture(this.uvTexture);
    gl.deleteProgram(this.rgbaProgram);
    gl.deleteProgram(this.nv12Program);
    gl.deleteBuffer(this.positionBuffer);
    gl.deleteBuffer(this.texCoordBuffer);

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

  private createProgram(fragmentShaderSource: string): WebGLProgram {
    const gl = this.gl;
    const vs = this.createShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
    const fs = this.createShader(gl.FRAGMENT_SHADER, fragmentShaderSource);

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

  private createStaticBuffer(data: Float32Array): WebGLBuffer {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error("Failed to create WebGL buffer");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buffer;
  }

  private createTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    // Set texture parameters for non-power-of-2 textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return texture;
  }

  private ensureCanvasSize(width: number, height: number): void {
    const gl = this.gl;
    if (this.currentWidth !== width || this.currentHeight !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.currentWidth = width;
      this.currentHeight = height;
      this.rgbaTextureAllocated = false;
      this.yTextureAllocated = false;
      this.uvTextureAllocated = false;
      gl.viewport(0, 0, width, height);
    }
  }

  private bindGeometry(program: WebGLProgram): void {
    const gl = this.gl;
    const posLoc = gl.getAttribLocation(program, "a_position");
    if (posLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    }

    const texLoc = gl.getAttribLocation(program, "a_texCoord");
    if (texLoc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
      gl.enableVertexAttribArray(texLoc);
      gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
    }
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

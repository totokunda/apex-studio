/**
 * WebGL Filter Engine
 * Manages WebGL context and applies filters efficiently using GPU
 */

export class WebGLFilterEngine {
  private gl: WebGLRenderingContext | null = null;
  private canvas: HTMLCanvasElement;
  private program: WebGLProgram | null = null;
  private frameBuffer: WebGLFramebuffer | null = null;
  private tempTextures: WebGLTexture[] = [];
  private vertexBuffer: WebGLBuffer | null = null;
  private texCoordBuffer: WebGLBuffer | null = null;

  constructor(width: number, height: number) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl = this.canvas.getContext("webgl", {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });

    if (!this.gl) {
      throw new Error("WebGL not supported");
    }

    this.initBuffers();
  }

  private initBuffers() {
    if (!this.gl) return;

    // Vertex positions (full screen quad)
    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

    // Texture coordinates
    const texCoords = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);

    this.vertexBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

    this.texCoordBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
  }

  private createShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;

    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  private createProgram(
    vertexSource: string,
    fragmentSource: string,
  ): WebGLProgram | null {
    if (!this.gl) return null;

    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(
      this.gl.FRAGMENT_SHADER,
      fragmentSource,
    );

    if (!vertexShader || !fragmentShader) return null;

    const program = this.gl.createProgram();
    if (!program) return null;

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error("Program link error:", this.gl.getProgramInfoLog(program));
      this.gl.deleteProgram(program);
      return null;
    }

    return program;
  }

  private createTexture(
    image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  ): WebGLTexture | null {
    if (!this.gl) return null;

    const texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
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
      this.gl.LINEAR,
    );
    this.gl.texParameteri(
      this.gl.TEXTURE_2D,
      this.gl.TEXTURE_MAG_FILTER,
      this.gl.LINEAR,
    );
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      image,
    );

    return texture;
  }

  public applyFilter(
    sourceCanvas: HTMLCanvasElement,
    vertexShader: string,
    fragmentShader: string,
    uniforms: { [key: string]: any } = {},
  ): HTMLCanvasElement {
    if (!this.gl) return sourceCanvas;

    // Resize if needed
    if (
      this.canvas.width !== sourceCanvas.width ||
      this.canvas.height !== sourceCanvas.height
    ) {
      this.canvas.width = sourceCanvas.width;
      this.canvas.height = sourceCanvas.height;
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    // Create program
    const program = this.createProgram(vertexShader, fragmentShader);
    if (!program) return sourceCanvas;

    this.gl.useProgram(program);

    // Create texture from source
    const texture = this.createTexture(sourceCanvas);
    if (!texture) return sourceCanvas;

    // Set up vertex attributes
    const positionLocation = this.gl.getAttribLocation(program, "a_position");
    const texCoordLocation = this.gl.getAttribLocation(program, "a_texCoord");

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.vertexAttribPointer(
      positionLocation,
      2,
      this.gl.FLOAT,
      false,
      0,
      0,
    );

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
    this.gl.enableVertexAttribArray(texCoordLocation);
    this.gl.vertexAttribPointer(
      texCoordLocation,
      2,
      this.gl.FLOAT,
      false,
      0,
      0,
    );

    // Set uniforms
    const textureLoc = this.gl.getUniformLocation(program, "u_image");
    this.gl.uniform1i(textureLoc, 0);

    const resolutionLoc = this.gl.getUniformLocation(program, "u_resolution");
    this.gl.uniform2f(resolutionLoc, this.canvas.width, this.canvas.height);

    // Set custom uniforms
    for (const [name, value] of Object.entries(uniforms)) {
      const location = this.gl.getUniformLocation(program, name);
      if (location) {
        if (typeof value === "number") {
          this.gl.uniform1f(location, value);
        } else if (Array.isArray(value)) {
          if (value.length === 2)
            this.gl.uniform2f(location, value[0], value[1]);
          else if (value.length === 3)
            this.gl.uniform3f(location, value[0], value[1], value[2]);
          else if (value.length === 4)
            this.gl.uniform4f(location, value[0], value[1], value[2], value[3]);
        }
      }
    }

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    // Cleanup
    this.gl.deleteTexture(texture);
    this.gl.deleteProgram(program);

    return this.canvas;
  }

  public resize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;
    if (this.gl) {
      this.gl.viewport(0, 0, width, height);
    }
  }

  public dispose() {
    if (this.gl) {
      if (this.vertexBuffer) this.gl.deleteBuffer(this.vertexBuffer);
      if (this.texCoordBuffer) this.gl.deleteBuffer(this.texCoordBuffer);
      if (this.frameBuffer) this.gl.deleteFramebuffer(this.frameBuffer);
      this.tempTextures.forEach((tex) => this.gl!.deleteTexture(tex));
    }
  }
}

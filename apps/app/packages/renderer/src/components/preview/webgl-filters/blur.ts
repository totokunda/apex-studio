/**
 * WebGL Gaussian Blur Filter
 * Uses a two-pass approach (horizontal then vertical) for optimal performance
 */

import { WebGLFilterBase } from "./WebGLFilterBase";

const vertexShader = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Horizontal blur pass
const horizontalBlurShader = `
  precision mediump float;
  
  uniform sampler2D u_image;
  uniform vec2 u_resolution;
  uniform float u_radius;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec2 texelSize = 1.0 / u_resolution;
    vec4 color = vec4(0.0);
    float total = 0.0;
    
    // Gaussian weights for radius
    float radius = u_radius;
    
    for (float x = -10.0; x <= 10.0; x += 1.0) {
      if (abs(x) > radius) continue;
      
      float weight = exp(-0.5 * (x * x) / (radius * radius * 0.15));
      vec2 offset = vec2(x * texelSize.x, 0.0);
      color += texture2D(u_image, v_texCoord + offset) * weight;
      total += weight;
    }
    
    gl_FragColor = color / total;
  }
`;

// Vertical blur pass
const verticalBlurShader = `
  precision mediump float;
  
  uniform sampler2D u_image;
  uniform vec2 u_resolution;
  uniform float u_radius;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec2 texelSize = 1.0 / u_resolution;
    vec4 color = vec4(0.0);
    float total = 0.0;
    
    // Gaussian weights for radius
    float radius = u_radius;
    
    for (float y = -10.0; y <= 10.0; y += 1.0) {
      if (abs(y) > radius) continue;
      
      float weight = exp(-0.5 * (y * y) / (radius * radius * 0.15));
      vec2 offset = vec2(0.0, y * texelSize.y);
      color += texture2D(u_image, v_texCoord + offset) * weight;
      total += weight;
    }
    
    gl_FragColor = color / total;
  }
`;

export class WebGLBlur extends WebGLFilterBase {
  private horizontalProgram: WebGLProgram | null = null;
  private verticalProgram: WebGLProgram | null = null;
  private tempTexture: WebGLTexture | null = null;
  private frameBuffer: WebGLFramebuffer | null = null;

  constructor() {
    super();

    this.initResources();
  }

  private initResources() {
    const gl = this.ensureContext();
    if (!gl) {
      this.horizontalProgram = null;
      this.verticalProgram = null;
      this.frameBuffer = null;
      this.tempTexture = null;
      return;
    }

    this.horizontalProgram = this.createProgram(
      vertexShader,
      horizontalBlurShader,
    );
    this.verticalProgram = this.createProgram(vertexShader, verticalBlurShader);
    this.frameBuffer = gl.createFramebuffer();
    if (!this.frameBuffer) {
      console.error("Failed to create framebuffer for blur filter");
    }
    this.tempTexture = null;
  }

  protected onContextLost(): void {
    super.onContextLost();
    this.horizontalProgram = null;
    this.verticalProgram = null;
    this.tempTexture = null;
    this.frameBuffer = null;
  }

  protected onContextRestored(): void {
    super.onContextRestored();
    this.initResources();
  }

  private applyPass(
    program: WebGLProgram,
    texture: WebGLTexture,
    radius: number,
    outputToScreen: boolean,
  ) {
    const gl = this.ensureContext();
    if (!gl) return;

    gl.useProgram(program);

    // Bind framebuffer for intermediate pass, null for final pass
    if (!outputToScreen && !this.frameBuffer) {
      return;
    }
    gl.bindFramebuffer(
      gl.FRAMEBUFFER,
      outputToScreen ? null : this.frameBuffer,
    );

    // Set up attributes using base class method
    this.setupAttributes(program);

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);
    gl.uniform2f(
      gl.getUniformLocation(program, "u_resolution"),
      this.canvas.width,
      this.canvas.height,
    );
    gl.uniform1f(gl.getUniformLocation(program, "u_radius"), radius);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  public apply(
    sourceCanvas: HTMLCanvasElement,
    radius: number,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (
      !gl ||
      !this.horizontalProgram ||
      !this.verticalProgram ||
      radius <= 0
    ) {
      return sourceCanvas;
    }

    // Resize canvas if needed using base class method
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Resize temp texture if needed
    if (!this.frameBuffer) {
      return sourceCanvas;
    }

    if (
      !this.tempTexture ||
      this.canvas.width !== sourceCanvas.width ||
      this.canvas.height !== sourceCanvas.height
    ) {
      if (this.tempTexture) {
        gl.deleteTexture(this.tempTexture);
      }
      this.tempTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.tempTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        this.canvas.width,
        this.canvas.height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );

      // Attach to framebuffer
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.tempTexture,
        0,
      );
    }

    // Create texture from source
    const sourceTexture = this.createTextureFromCanvas(sourceCanvas);
    if (!sourceTexture) return sourceCanvas;

    // Pass 1: Horizontal blur (render to temp texture)
    this.applyPass(this.horizontalProgram, sourceTexture, radius, false);

    // Pass 2: Vertical blur (render to screen)
    if (this.tempTexture) {
      this.applyPass(this.verticalProgram, this.tempTexture, radius, true);
    }

    // Cleanup source texture
    gl.deleteTexture(sourceTexture);

    return this.canvas;
  }

  public dispose() {
    const gl = this.gl;
    if (gl) {
      if (this.horizontalProgram) gl.deleteProgram(this.horizontalProgram);
      if (this.verticalProgram) gl.deleteProgram(this.verticalProgram);
      if (this.tempTexture) gl.deleteTexture(this.tempTexture);
      if (this.frameBuffer) gl.deleteFramebuffer(this.frameBuffer);
    }
    this.horizontalProgram = null;
    this.verticalProgram = null;
    this.tempTexture = null;
    this.frameBuffer = null;
    super.dispose();
  }
}

/**
 * WebGL Gaussian Blur Filter
 * Uses a two-pass approach (horizontal then vertical) for optimal performance
 */

import { WebGLFilterBase } from './WebGLFilterBase';

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

    if (this.gl) {
      this.horizontalProgram = this.createProgram(vertexShader, horizontalBlurShader);
      this.verticalProgram = this.createProgram(vertexShader, verticalBlurShader);
      this.frameBuffer = this.gl.createFramebuffer();
    }
  }

  private applyPass(program: WebGLProgram, texture: WebGLTexture, radius: number, outputToScreen: boolean) {
    if (!this.gl) return;

    this.gl.useProgram(program);

    // Bind framebuffer for intermediate pass, null for final pass
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, outputToScreen ? null : this.frameBuffer);

    // Set up attributes using base class method
    this.setupAttributes(program);

    // Set uniforms
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_image'), 0);
    this.gl.uniform2f(this.gl.getUniformLocation(program, 'u_resolution'), this.canvas.width, this.canvas.height);
    this.gl.uniform1f(this.gl.getUniformLocation(program, 'u_radius'), radius);

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  public apply(sourceCanvas: HTMLCanvasElement, radius: number): HTMLCanvasElement {
    if (!this.gl || !this.horizontalProgram || !this.verticalProgram || radius <= 0) {
      return sourceCanvas;
    }

    // Resize canvas if needed using base class method
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Resize temp texture if needed
    if (!this.tempTexture || this.canvas.width !== sourceCanvas.width || this.canvas.height !== sourceCanvas.height) {
      if (this.tempTexture) {
        this.gl.deleteTexture(this.tempTexture);
      }
      this.tempTexture = this.gl.createTexture();
      this.gl.bindTexture(this.gl.TEXTURE_2D, this.tempTexture);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);

      // Attach to framebuffer
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.frameBuffer);
      this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, this.tempTexture, 0);
    }

    // Create texture from source
    const sourceTexture = this.createTextureFromCanvas(sourceCanvas);
    if (!sourceTexture) return sourceCanvas;

    // Pass 1: Horizontal blur (render to temp texture)
    this.applyPass(this.horizontalProgram, sourceTexture, radius, false);

    // Pass 2: Vertical blur (render to screen)
    this.applyPass(this.verticalProgram, this.tempTexture!, radius, true);

    // Cleanup source texture
    this.gl.deleteTexture(sourceTexture);

    return this.canvas;
  }

  public dispose() {
    super.dispose();
    if (this.gl) {
      if (this.horizontalProgram) this.gl.deleteProgram(this.horizontalProgram);
      if (this.verticalProgram) this.gl.deleteProgram(this.verticalProgram);
      if (this.tempTexture) this.gl.deleteTexture(this.tempTexture);
      if (this.frameBuffer) this.gl.deleteFramebuffer(this.frameBuffer);
    }
  }
}

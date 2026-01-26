/**
 * WebGL Noise Filter
 * Adds random noise to the image using GPU acceleration
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

const fragmentShader = `
  precision mediump float;
  
  uniform sampler2D u_image;
  uniform float u_noise;
  uniform float u_seed;
  
  varying vec2 v_texCoord;
  
  // Simple pseudo-random function
  float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233)) + u_seed) * 43758.5453);
  }
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Generate random noise for this pixel
    float noise = (random(v_texCoord) - 0.5) * u_noise;
    
    // Apply noise to RGB channels
    color.rgb += noise;
    
    gl_FragColor = color;
  }
`;

export class WebGLNoise extends WebGLFilterBase {
  private program: WebGLProgram | null = null;
  private seed: number = 0;

  constructor() {
    super();
    this.initProgram();
  }

  private initProgram() {
    this.program = this.createProgram(vertexShader, fragmentShader);
  }

  protected onContextLost(): void {
    super.onContextLost();
    this.program = null;
  }

  protected onContextRestored(): void {
    super.onContextRestored();
    this.initProgram();
  }

  public apply(
    sourceCanvas: HTMLCanvasElement,
    amount: number,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (!gl || !this.program || amount <= 0) {
      return sourceCanvas;
    }

    // Resize canvas if needed
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Create texture from source
    const texture = this.createTextureFromCanvas(sourceCanvas);
    if (!texture) return sourceCanvas;

    // Use program
    gl.useProgram(this.program);

    // Set up attributes
    this.setupAttributes(this.program);

    // Update seed for random variation
    this.seed = Math.random() * 1000;

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_noise"), amount / 100); // Normalize to 0..1
    gl.uniform1f(gl.getUniformLocation(this.program, "u_seed"), this.seed);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Cleanup
    gl.deleteTexture(texture);

    return this.canvas;
  }

  public dispose() {
    const gl = this.gl;
    if (gl && this.program) {
      gl.deleteProgram(this.program);
    }
    this.program = null;
    super.dispose();
  }
}

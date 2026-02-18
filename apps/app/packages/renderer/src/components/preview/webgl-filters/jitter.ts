/**
 * WebGL Jitter Filter
 * Simulates camera shake / frame instability by applying random per-frame
 * UV offsets, creating the look of handheld found-footage camcorder recordings.
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
  uniform float u_intensity;
  uniform vec2 u_offset;
  
  varying vec2 v_texCoord;
  
  void main() {
    // Apply the random offset to simulate camera shake
    vec2 uv = v_texCoord + u_offset * u_intensity;
    
    // Clamp UV to prevent sampling outside the texture
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    
    gl_FragColor = texture2D(u_image, uv);
  }
`;

export class WebGLJitter extends WebGLFilterBase {
  private program: WebGLProgram | null = null;

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

    // Generate random offset per frame
    // Amplitude is scaled by amount (0-100 → 0-0.02 in UV space)
    const maxShift = 0.02;
    const offsetX = (Math.random() - 0.5) * 2.0 * maxShift;
    const offsetY = (Math.random() - 0.5) * 2.0 * maxShift;

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_intensity"),
      amount / 100,
    ); // Normalize to 0..1
    gl.uniform2f(
      gl.getUniformLocation(this.program, "u_offset"),
      offsetX,
      offsetY,
    );

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

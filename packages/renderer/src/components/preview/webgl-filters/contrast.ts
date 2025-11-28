/**
 * WebGL Contrast Filter
 * Adjusts image contrast using GPU acceleration
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
  uniform float u_contrast;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Apply contrast adjustment
    // Formula: (color - 0.5) * (1 + contrast) + 0.5
    // This pivots around middle gray (0.5)
    float factor = (1.0 + u_contrast);
    color.rgb = (color.rgb - 0.5) * factor + 0.5;
    
    gl_FragColor = vec4(color.rgb, color.a);
  }
`;

export class WebGLContrast extends WebGLFilterBase {
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
    contrast: number,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (!gl || !this.program || contrast === 0) {
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

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    // Normalize contrast from -100..100 to -1..1
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_contrast"),
      contrast / 100,
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

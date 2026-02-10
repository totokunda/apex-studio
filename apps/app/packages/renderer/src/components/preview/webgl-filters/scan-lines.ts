/**
 * WebGL Scan Lines Filter
 * Adds horizontal scan lines to simulate CRT/VHS/camcorder display artifacts
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
  uniform vec2 u_resolution;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Create scan lines based on pixel y position
    // Line spacing is ~2-3 pixels for a classic CRT look
    float lineSpacing = 3.0;
    float y = gl_FragCoord.y;
    
    // Create alternating bright/dark bands with smooth transitions
    float scanLine = 0.5 + 0.5 * sin(y * 3.14159265 / (lineSpacing * 0.5));
    
    // Mix between full brightness and darkened based on intensity
    float dimFactor = mix(1.0, scanLine, u_intensity * 0.6);
    
    // Add subtle brightness variation for more authenticity
    float largeBand = 0.5 + 0.5 * sin(y * 3.14159265 / (u_resolution.y * 0.25));
    float bandFactor = mix(1.0, 0.92 + 0.08 * largeBand, u_intensity);
    
    gl_FragColor = vec4(color.rgb * dimFactor * bandFactor, color.a);
  }
`;

export class WebGLScanLines extends WebGLFilterBase {
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

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_intensity"),
      amount / 100,
    ); // Normalize to 0..1
    gl.uniform2f(
      gl.getUniformLocation(this.program, "u_resolution"),
      this.canvas.width,
      this.canvas.height,
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

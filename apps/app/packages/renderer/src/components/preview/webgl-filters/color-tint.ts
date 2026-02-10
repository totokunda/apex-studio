/**
 * WebGL Color Tint Filter
 * Maps luminance to a specified tint color using GPU acceleration.
 * Used for effects like night vision green, security cam blue, etc.
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
  uniform vec3 u_tintColor;
  uniform float u_intensity;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Calculate luminance using standard coefficients
    float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Map luminance to tint color
    vec3 tinted = luminance * u_tintColor;
    
    // Mix between original and tinted based on intensity
    vec3 result = mix(color.rgb, tinted, u_intensity);
    
    gl_FragColor = vec4(result, color.a);
  }
`;

export class WebGLColorTint extends WebGLFilterBase {
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

  /**
   * Apply color tint to the source canvas.
   * @param sourceCanvas - Source canvas to apply filter to
   * @param r - Red component of tint color (0-1)
   * @param g - Green component of tint color (0-1)
   * @param b - Blue component of tint color (0-1)
   * @param intensity - Intensity of the tint effect (0-100)
   */
  public apply(
    sourceCanvas: HTMLCanvasElement,
    r: number,
    g: number,
    b: number,
    intensity: number,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (!gl || !this.program || intensity <= 0) {
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
    gl.uniform3f(
      gl.getUniformLocation(this.program, "u_tintColor"),
      r,
      g,
      b,
    );
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_intensity"),
      intensity / 100,
    ); // Normalize to 0..1

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

/**
 * WebGL Pixelate Filter
 * Simulates low-resolution camera footage by downsampling and re-upsampling.
 * Creates a blocky, degraded quality look typical of old camcorders.
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
  uniform float u_pixelSize;
  uniform vec2 u_resolution;
  
  varying vec2 v_texCoord;
  
  void main() {
    // Calculate the size of each "pixel block" in texture coordinates
    vec2 blockSize = u_pixelSize / u_resolution;
    
    // Snap the texture coordinate to the nearest block center
    vec2 blockCoord = floor(v_texCoord / blockSize) * blockSize + blockSize * 0.5;
    
    // Clamp to valid range
    blockCoord = clamp(blockCoord, vec2(0.0), vec2(1.0));
    
    // Sample the color from the block center
    gl_FragColor = texture2D(u_image, blockCoord);
  }
`;

export class WebGLPixelate extends WebGLFilterBase {
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

    // Map 0-100 amount to pixel block size (1 = no effect, up to 20 = very blocky)
    const pixelSize = 1.0 + (amount / 100) * 19.0;

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_pixelSize"),
      pixelSize,
    );
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

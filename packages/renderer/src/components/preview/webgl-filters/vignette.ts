/**
 * WebGL Vignette Filter
 * Darkens the edges of the image using GPU acceleration
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

const fragmentShader = `
  precision mediump float;
  
  uniform sampler2D u_image;
  uniform float u_vignette;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Calculate distance from center
    vec2 center = vec2(0.5, 0.5);
    vec2 diff = v_texCoord - center;
    float dist = length(diff);
    
    // Normalize distance (0 at center, ~0.707 at corners)
    float maxDist = length(vec2(0.5, 0.5));
    float normDist = dist / maxDist;
    
    // Create stronger falloff using cubic easing and increased strength
    // Using cubic (power of 3) instead of quadratic for more aggressive darkening
    float falloff = 1.0 - (pow(normDist, 3.0) * u_vignette * 2.0);
    falloff = clamp(falloff, 0.0, 1.0);
    
    // Apply vignette to RGB channels only
    gl_FragColor = vec4(color.rgb * falloff, color.a);
  }
`;

export class WebGLVignette extends WebGLFilterBase {
  private program: WebGLProgram | null = null;

  constructor() {
    super();
    if (this.gl) {
      this.program = this.createProgram(vertexShader, fragmentShader);
    }
  }

  public apply(sourceCanvas: HTMLCanvasElement, amount: number): HTMLCanvasElement {
    if (!this.gl || !this.program || amount <= 0) {
      return sourceCanvas;
    }

    // Resize canvas if needed
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Create texture from source
    const texture = this.createTextureFromCanvas(sourceCanvas);
    if (!texture) return sourceCanvas;

    // Use program
    this.gl.useProgram(this.program);

    // Set up attributes
    this.setupAttributes(this.program);

    // Set uniforms
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.uniform1i(this.gl.getUniformLocation(this.program, 'u_image'), 0);
    this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'u_vignette'), amount / 100); // Normalize to 0..1

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    // Cleanup
    this.gl.deleteTexture(texture);

    return this.canvas;
  }

  public dispose() {
    super.dispose();
    if (this.gl && this.program) {
      this.gl.deleteProgram(this.program);
    }
  }
}

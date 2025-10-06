/**
 * WebGL Sharpness Filter
 * Applies unsharp mask sharpening using GPU acceleration
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
  uniform vec2 u_resolution;
  uniform float u_sharpness;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec2 texel = 1.0 / u_resolution;
    
    // Sample center and surrounding pixels for 3x3 kernel
    vec4 center = texture2D(u_image, v_texCoord);
    
    vec4 top = texture2D(u_image, v_texCoord + vec2(0.0, -texel.y));
    vec4 bottom = texture2D(u_image, v_texCoord + vec2(0.0, texel.y));
    vec4 left = texture2D(u_image, v_texCoord + vec2(-texel.x, 0.0));
    vec4 right = texture2D(u_image, v_texCoord + vec2(texel.x, 0.0));
    
    vec4 topLeft = texture2D(u_image, v_texCoord + vec2(-texel.x, -texel.y));
    vec4 topRight = texture2D(u_image, v_texCoord + vec2(texel.x, -texel.y));
    vec4 bottomLeft = texture2D(u_image, v_texCoord + vec2(-texel.x, texel.y));
    vec4 bottomRight = texture2D(u_image, v_texCoord + vec2(texel.x, texel.y));
    
    // Gaussian-weighted blur for unsharp mask
    vec4 blurred = (
      topLeft + top * 2.0 + topRight +
      left * 2.0 + center * 4.0 + right * 2.0 +
      bottomLeft + bottom * 2.0 + bottomRight
    ) / 16.0;
    
    // Unsharp mask: original + strength * (original - blurred)
    float strength = u_sharpness * 5.0; // Scale for visible effect
    vec4 sharpened = center + strength * (center - blurred);
    
    gl_FragColor = vec4(sharpened.rgb, center.a);
  }
`;

export class WebGLSharpness extends WebGLFilterBase {
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
    this.gl.uniform2f(this.gl.getUniformLocation(this.program, 'u_resolution'), this.canvas.width, this.canvas.height);
    this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'u_sharpness'), amount / 100); // Normalize to 0..1

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

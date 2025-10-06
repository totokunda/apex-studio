/**
 * WebGL Brightness Filter
 * Adjusts image brightness using GPU acceleration
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
  uniform float u_brightness;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Apply brightness adjustment
    // brightness range: -1.0 to 1.0
    color.rgb += u_brightness;
    
    gl_FragColor = vec4(color.rgb, color.a);
  }
`;

export class WebGLBrightness extends WebGLFilterBase {
  private program: WebGLProgram | null = null;

  constructor() {
    super();
    if (this.gl) {
      this.program = this.createProgram(vertexShader, fragmentShader);
    }
  }

  public apply(sourceCanvas: HTMLCanvasElement, brightness: number): HTMLCanvasElement {
    if (!this.gl || !this.program || brightness === 0) {
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
    this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'u_brightness'), brightness);

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

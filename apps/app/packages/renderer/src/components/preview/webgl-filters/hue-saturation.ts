/**
 * WebGL Hue and Saturation Filter
 * Adjusts hue and saturation using GPU acceleration
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
  uniform float u_hue;
  uniform float u_saturation;
  
  varying vec2 v_texCoord;
  
  // RGB to HSL conversion
  vec3 rgb2hsl(vec3 color) {
    float maxC = max(max(color.r, color.g), color.b);
    float minC = min(min(color.r, color.g), color.b);
    float delta = maxC - minC;
    
    float h = 0.0;
    float s = 0.0;
    float l = (maxC + minC) / 2.0;
    
    if (delta > 0.0001) {
      s = l < 0.5 ? delta / (maxC + minC) : delta / (2.0 - maxC - minC);
      
      if (color.r >= maxC) {
        h = (color.g - color.b) / delta + (color.g < color.b ? 6.0 : 0.0);
      } else if (color.g >= maxC) {
        h = (color.b - color.r) / delta + 2.0;
      } else {
        h = (color.r - color.g) / delta + 4.0;
      }
      h /= 6.0;
    }
    
    return vec3(h, s, l);
  }
  
  // HSL to RGB conversion
  float hue2rgb(float p, float q, float t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0/2.0) return q;
    if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
  }
  
  vec3 hsl2rgb(vec3 hsl) {
    float h = hsl.x;
    float s = hsl.y;
    float l = hsl.z;
    
    if (s == 0.0) {
      return vec3(l, l, l);
    }
    
    float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
    float p = 2.0 * l - q;
    
    return vec3(
      hue2rgb(p, q, h + 1.0/3.0),
      hue2rgb(p, q, h),
      hue2rgb(p, q, h - 1.0/3.0)
    );
  }
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    
    // Convert to HSL
    vec3 hsl = rgb2hsl(color.rgb);
    
    // Adjust hue (rotate)
    hsl.x = mod(hsl.x + u_hue, 1.0);
    
    // Adjust saturation
    hsl.y = clamp(hsl.y * (1.0 + u_saturation), 0.0, 1.0);
    
    // Convert back to RGB
    vec3 rgb = hsl2rgb(hsl);
    
    gl_FragColor = vec4(rgb, color.a);
  }
`;

export class WebGLHueSaturation extends WebGLFilterBase {
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
    hue: number,
    saturation: number,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (!gl || !this.program || (hue === 0 && saturation === 0)) {
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
    // Normalize hue from -100..100 to degrees, then to 0..1 range
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_hue"),
      (hue * 3.6) / 360,
    );
    // Normalize saturation from -100..100 to -1..1
    gl.uniform1f(
      gl.getUniformLocation(this.program, "u_saturation"),
      saturation / 100,
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

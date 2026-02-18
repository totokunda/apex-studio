/**
 * WebGL Interlace Filter
 * Simulates interlacing artifacts from old camcorders and CRT displays.
 * Shows alternating scan line fields with optional horizontal field shift.
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
  uniform float u_seed;
  
  varying vec2 v_texCoord;
  
  // Simple pseudo-random function
  float random(float x) {
    return fract(sin(x * 12.9898 + u_seed) * 43758.5453);
  }
  
  void main() {
    vec2 uv = v_texCoord;
    float y = gl_FragCoord.y;
    
    // Determine if this is an odd or even scanline
    bool isOddLine = mod(y, 2.0) < 1.0;
    
    // Apply horizontal field shift to odd lines (simulates interlace combing)
    if (isOddLine) {
      float fieldShift = u_intensity * 0.003 * (random(floor(y)) - 0.5);
      uv.x += fieldShift;
    }
    
    vec4 color = texture2D(u_image, uv);
    
    // Dim alternate lines slightly to simulate field difference
    float dimFactor = isOddLine ? (1.0 - u_intensity * 0.15) : 1.0;
    
    // Add subtle vertical blending artifact between fields
    vec2 offsetUV = uv + vec2(0.0, 1.0 / u_resolution.y);
    vec4 neighborColor = texture2D(u_image, offsetUV);
    float blendAmount = isOddLine ? u_intensity * 0.2 : 0.0;
    color = mix(color, neighborColor, blendAmount);
    
    gl_FragColor = vec4(color.rgb * dimFactor, color.a);
  }
`;

export class WebGLInterlace extends WebGLFilterBase {
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

    // Update seed for per-frame variation
    this.seed = Math.random() * 1000;

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

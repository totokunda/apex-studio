/**
 * Base class for WebGL filters
 * Provides common functionality for shader compilation, texture management, and rendering
 */

export abstract class WebGLMaskBase {
    protected gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
    protected canvas: HTMLCanvasElement;
    protected vertexBuffer: WebGLBuffer | null = null;
    protected texCoordBuffer: WebGLBuffer | null = null;
    private contextType: 'webgl' | 'webgl2' = 'webgl';
  
    constructor(contextType: 'webgl' | 'webgl2' = 'webgl') {
      this.contextType = contextType;
      this.canvas = document.createElement('canvas');
      this.initContext();
      
      // Add context lost/restored event listeners
      this.canvas.addEventListener('webglcontextlost', this.handleContextLost.bind(this), false);
      this.canvas.addEventListener('webglcontextrestored', this.handleContextRestored.bind(this), false);
    }
    
    private handleContextLost(event: Event) {
      event.preventDefault();
      console.warn('WebGL context lost, will attempt to restore');
      this.onContextLost();
    }
    
    private handleContextRestored(_event: Event) {
      console.log('WebGL context restored, reinitializing');
      this.initContext();
      this.onContextRestored();
    }
    
    // Subclasses can override these to handle context loss/restoration
    protected onContextLost() {
      // Default: do nothing
    }
    
    protected onContextRestored() {
      // Default: do nothing
    }
    
    private initContext() {
      // Try to get the requested context type
      if (this.contextType === 'webgl2') {
        this.gl = this.canvas.getContext('webgl2', {
          premultipliedAlpha: false,
          preserveDrawingBuffer: true
        }) as WebGL2RenderingContext | null;
        
        // Fallback to WebGL1 if WebGL2 is not available
        if (!this.gl) {
          console.warn('WebGL2 not available, falling back to WebGL1');
          this.gl = this.canvas.getContext('webgl', {
            premultipliedAlpha: false,
            preserveDrawingBuffer: true
          });
        }
      } else {
        this.gl = this.canvas.getContext('webgl', {
          premultipliedAlpha: false,
          preserveDrawingBuffer: true
        });
      }
  
      if (this.gl) {
        this.initBuffers();
      } else {
        console.error('Failed to initialize WebGL context');
      }
    }
  
    protected initBuffers() {
      if (!this.gl) return;
  
      // Full-screen quad vertices
      const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
      // Texture coordinates without Y-flip to prevent upside-down rendering
      const texCoords = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
  
      this.vertexBuffer = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
  
      this.texCoordBuffer = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
    }
  
    protected createShader(type: number, source: string): WebGLShader | null {
      if (!this.gl) return null;
  
      const shader = this.gl.createShader(type);
      if (!shader) return null;
  
      this.gl.shaderSource(shader, source);
      this.gl.compileShader(shader);
  
      if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
        this.gl.deleteShader(shader);
        return null;
      }
  
      return shader;
    }
  
    protected createProgram(vertexSource: string, fragmentSource: string): WebGLProgram | null {
      if (!this.gl) return null;
  
      const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
      const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
  
      if (!vertexShader || !fragmentShader) return null;
  
      const program = this.gl.createProgram();
      if (!program) return null;
  
      this.gl.attachShader(program, vertexShader);
      this.gl.attachShader(program, fragmentShader);
      this.gl.linkProgram(program);
  
      if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
        console.error('Program link error:', this.gl.getProgramInfoLog(program));
        return null;
      }
  
      return program;
    }
  
    protected createTextureFromCanvas(canvas: HTMLCanvasElement): WebGLTexture | null {
      if (!this.gl) return null;
  
      const texture = this.gl.createTexture();
      this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
      // Use shader-based flip (u_originTopLeft) instead of flipping uploads
      this.gl.pixelStorei(this.gl.UNPACK_FLIP_Y_WEBGL, false);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
      this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, canvas);
  
      return texture;
    }
  
    protected setupAttributes(program: WebGLProgram) {
      if (!this.gl) return;
  
      const positionLocation = this.gl.getAttribLocation(program, 'a_position');
      const texCoordLocation = this.gl.getAttribLocation(program, 'a_texCoord');
  
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
      this.gl.enableVertexAttribArray(positionLocation);
      this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
  
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
      this.gl.enableVertexAttribArray(texCoordLocation);
      this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 0, 0);
    }
  
    protected resizeCanvas(width: number, height: number) {
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        if (this.gl) {
          this.gl.viewport(0, 0, width, height);
        }
      }
    }
  
    public dispose() {
      if (this.gl) {
        if (this.vertexBuffer) this.gl.deleteBuffer(this.vertexBuffer);
        if (this.texCoordBuffer) this.gl.deleteBuffer(this.texCoordBuffer);
      }
    }
  
    // Abstract method that subclasses must implement
    public abstract apply(sourceCanvas: HTMLCanvasElement, ...params: any[]): HTMLCanvasElement;

    protected debugDownloadCanvas(filename = 'shape-mask-debug.png') {
      if (!this.canvas) return;
      // Prefer toBlob (async, crisp); fallback to dataURL if not supported.
      if ('toBlob' in this.canvas) {
        (this.canvas as HTMLCanvasElement).toBlob((blob) => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        }, 'image/png');
      } else {
        const url = (this.canvas as HTMLCanvasElement).toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    }
  
    /**
     * (Optional) Draw a debug rectangle over the GL output so we can
     * visually confirm the interpreted bounds. This uses a 2D temp canvas
     * and then writes back into our GL canvas.
     */
    protected debugAnnotateBoundsOnTop(
      srcCanvas: HTMLCanvasElement,
      bounds: { x: number; y: number; width: number; height: number }
    ) {
      const { width: W, height: H } = srcCanvas;
      const tmp = document.createElement('canvas');
      tmp.width = W;
      tmp.height = H;
      const ctx = tmp.getContext('2d');
      if (!ctx) return;
  
      // draw GL result first
      ctx.drawImage(this.canvas, 0, 0);
  
      // draw bounds (top-left origin)
      ctx.save();
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = 'rgba(255,0,0,0.9)';
      ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
      ctx.restore();
  
      // copy annotated back into our GL canvas
      const gl2d = this.canvas.getContext('2d');
      if (gl2d) {
        gl2d.clearRect(0, 0, W, H);
        gl2d.drawImage(tmp, 0, 0);
      }
    }
  }
  
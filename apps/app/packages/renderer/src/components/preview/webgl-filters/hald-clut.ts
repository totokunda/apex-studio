/**
 * WebGL2 HALD CLUT Filter
 * Applies color grading using HALD Color Lookup Table images
 * Uses 3D texture sampling for efficient and accurate color lookup
 * Implements caching to avoid reloading the same CLUT images
 */

import { WebGLFilterBase } from "./WebGLFilterBase";
import { readFileBuffer } from "@app/preload";
import * as twgl from "twgl.js";

// Cache for loaded CLUT images
const clutImageCache = new Map<string, HTMLImageElement>();

// Track loading promises to avoid duplicate requests
const loadingPromises = new Map<string, Promise<HTMLImageElement>>();

export class WebGLHaldClut extends WebGLFilterBase {
  private program: WebGLProgram | null = null;
  private clutTextures: Map<string, WebGLTexture> = new Map();
  private clutLevels: Map<string, number> = new Map();
  private clutWidths: Map<string, number> = new Map();
  private loadedClutPaths: Set<string> = new Set();
  private contextLost = false;

  constructor() {
    // Request dedicated WebGL2 context from base class
    super("webgl2", "preview-webgl-filter-hald");
    this.initShaders();
  }

  protected onContextLost(): void {
    super.onContextLost();
    this.contextLost = true;
    this.program = null;
    console.warn("[HaldClut] WebGL context lost");
  }

  protected onContextRestored(): void {
    super.onContextRestored();
    this.contextLost = false;
    console.info("[HaldClut] WebGL context restored, reinitializing resources");
    void this.reinitializeResources();
  }

  private initShaders() {
    const gl = this.ensureContext();
    if (!gl) return;
    if (!(gl instanceof WebGL2RenderingContext)) {
      console.error("WebGLHaldClut requires a WebGL2 context");
      this.program = null;
      return;
    }

    const vertexShaderSource = `#version 300 es
      in vec2 a_position;
      in vec2 a_texCoord;
      out vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

    const fragmentShaderSource = `#version 300 es
      precision highp float;
      
      in vec2 v_texCoord;
      out vec4 fragColor;
      
      uniform sampler2D u_sourceTexture;
      uniform sampler2D u_clutTexture;
      uniform float u_clutSize;
      uniform float u_clutWidth;
      uniform float u_intensity;
      
      // Helper function to get Hald CLUT texture coordinates from RGB indices
      // Using ImageMagick's exact formula: offset = r + level*g + cubeSize*b
      vec2 getHaldCoord(vec3 rgb, float level, float cubeSize, float width) {
        float offset = rgb.r + level * rgb.g + cubeSize * rgb.b;
        float x = mod(offset, width);
        float y = floor(offset / width);
        
        // Add 0.5 to sample from center of texel
        return (vec2(x, y) + 0.5) / width;
      }
      
      vec3 applyHaldClut(vec3 color) {
        float level = u_clutSize;
        float levelMinusOne = level - 1.0;
        float cubeSize = level * level;
        float width = u_clutWidth;
        
        // Clamp color to valid range
        color = clamp(color, 0.0, 1.0);
        
        // Scale RGB to cube coordinates [0, level-1]
        vec3 scaledColor = color * levelMinusOne;
        
        // Get integer and fractional parts
        vec3 baseColor = floor(scaledColor);
        vec3 nextColor = min(baseColor + 1.0, levelMinusOne);
        vec3 frac = scaledColor - baseColor;
        
        // Sample all 8 corners for trilinear interpolation
        vec3 c000 = texture(u_clutTexture, getHaldCoord(vec3(baseColor.r, baseColor.g, baseColor.b), level, cubeSize, width)).rgb;
        vec3 c001 = texture(u_clutTexture, getHaldCoord(vec3(baseColor.r, baseColor.g, nextColor.b), level, cubeSize, width)).rgb;
        vec3 c010 = texture(u_clutTexture, getHaldCoord(vec3(baseColor.r, nextColor.g, baseColor.b), level, cubeSize, width)).rgb;
        vec3 c011 = texture(u_clutTexture, getHaldCoord(vec3(baseColor.r, nextColor.g, nextColor.b), level, cubeSize, width)).rgb;
        vec3 c100 = texture(u_clutTexture, getHaldCoord(vec3(nextColor.r, baseColor.g, baseColor.b), level, cubeSize, width)).rgb;
        vec3 c101 = texture(u_clutTexture, getHaldCoord(vec3(nextColor.r, baseColor.g, nextColor.b), level, cubeSize, width)).rgb;
        vec3 c110 = texture(u_clutTexture, getHaldCoord(vec3(nextColor.r, nextColor.g, baseColor.b), level, cubeSize, width)).rgb;
        vec3 c111 = texture(u_clutTexture, getHaldCoord(vec3(nextColor.r, nextColor.g, nextColor.b), level, cubeSize, width)).rgb;
        
        // Trilinear interpolation
        vec3 c00 = mix(c000, c100, frac.r);
        vec3 c01 = mix(c001, c101, frac.r);
        vec3 c10 = mix(c010, c110, frac.r);
        vec3 c11 = mix(c011, c111, frac.r);
        
        vec3 c0 = mix(c00, c10, frac.g);
        vec3 c1 = mix(c01, c11, frac.g);
        
        return mix(c0, c1, frac.b);
      }
      
      void main() {
        vec4 sourceColor = texture(u_sourceTexture, v_texCoord);
        
        // Apply Hald CLUT to RGB channels
        vec3 transformedColor = applyHaldClut(sourceColor.rgb);
        
        // Blend between original and transformed based on intensity
        vec3 finalColor = mix(sourceColor.rgb, transformedColor, u_intensity);
        
        // Preserve alpha channel
        fragColor = vec4(finalColor, sourceColor.a);
        
        // Debug: uncomment to verify shader is running
        // fragColor = vec4(transformedColor, 1.0);
      }
    `;

    try {
      this.program = null;
      this.program = twgl.createProgramFromSources(gl, [
        vertexShaderSource,
        fragmentShaderSource,
      ]);
      if (!this.program) {
        console.error("Failed to create shader program");
      } else {
        console.log("Hald CLUT shader program created successfully");
      }
    } catch (error) {
      console.error("Error creating shader program:", error);
    }
  }

  private async reinitializeResources(): Promise<void> {
    const gl = this.ensureContext();
    if (!gl) return;
    this.initBuffers();
    this.initShaders();
    const paths = Array.from(this.loadedClutPaths);
    this.clutTextures.clear();
    this.clutLevels.clear();
    this.clutWidths.clear();
    for (const p of paths) {
      try {
        await this.preloadClut(p);
      } catch (e) {
        console.error(
          "[HaldClut] Failed to reload CLUT after context restore:",
          p,
          e,
        );
      }
    }
  }

  private async loadClutImage(imagePath: string): Promise<HTMLImageElement> {
    // Check cache first
    if (clutImageCache.has(imagePath)) {
      return clutImageCache.get(imagePath)!;
    }

    // Check if already loading
    if (loadingPromises.has(imagePath)) {
      return loadingPromises.get(imagePath)!;
    }

    // Load the image using the same pattern as fetchImage
    const loadPromise = (async () => {
      try {
        // Read file buffer using Electron's secure method
        const buffer = await readFileBuffer(imagePath);
        const blob = new Blob([buffer as unknown as ArrayBuffer]);
        const url = URL.createObjectURL(blob);

        const img = new Image();
        img.decoding = "async";
        img.src = url;

        await img.decode();

        // Clean up the object URL after loading
        URL.revokeObjectURL(url);

        clutImageCache.set(imagePath, img);
        loadingPromises.delete(imagePath);

        return img;
      } catch (error) {
        loadingPromises.delete(imagePath);
        console.error("Failed to load CLUT image:", { imagePath, error });
        throw new Error(`Failed to load CLUT image: ${imagePath}`);
      }
    })();

    loadingPromises.set(imagePath, loadPromise);
    return loadPromise;
  }

  /**
   * Preload a CLUT image and upload it to GPU.
   * Call this before using apply() for better performance.
   * Supports preloading multiple CLUTs - each is stored in a map by path.
   */
  public async preloadClut(clutImagePath: string): Promise<void> {
    const gl = this.ensureContext();
    if (!gl) return;
    this.loadedClutPaths.add(clutImagePath);
    if (
      this.contextLost ||
      ((gl as WebGLRenderingContext).isContextLost &&
        (gl as WebGLRenderingContext).isContextLost())
    ) {
      return;
    }

    // Check if already loaded
    if (this.clutTextures.has(clutImagePath)) {
      return;
    }

    if (!(gl instanceof WebGL2RenderingContext)) {
      console.error("WebGLHaldClut requires WebGL2 to preload CLUT textures");
      return;
    }
    const clutImage = await this.loadClutImage(clutImagePath);

    // Calculate Hald CLUT level from image dimensions
    const length = Math.min(clutImage.width, clutImage.height);
    let level = 2;
    while (level * level * level < length) {
      level++;
    }
    level = level * level; // ImageMagick squares the level

    // Create texture for this CLUT
    const texture = gl.createTexture();
    if (!texture) {
      console.error("Failed to create WebGL texture for CLUT:", clutImagePath);
      return;
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set texture parameters BEFORE uploading data for better compatibility
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    // Upload texture with RGB8 format for better precision
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGB8,
      gl.RGB,
      gl.UNSIGNED_BYTE,
      clutImage,
    );

    // Store texture and metadata
    this.clutTextures.set(clutImagePath, texture);
    this.clutLevels.set(clutImagePath, level);
    this.clutWidths.set(clutImagePath, clutImage.width);
  }

  public apply(
    sourceCanvas: HTMLCanvasElement,
    clutImagePath: string,
    intensity: number = 1.0,
  ): HTMLCanvasElement {
    const gl = this.ensureContext();
    if (!gl || !this.program) return sourceCanvas;
    if (
      (gl as WebGLRenderingContext).isContextLost &&
      (gl as WebGLRenderingContext).isContextLost()
    ) {
      this.contextLost = true;
      return sourceCanvas;
    }

    if (!(gl instanceof WebGL2RenderingContext)) {
      console.error("WebGLHaldClut requires WebGL2 to apply CLUT textures");
      return sourceCanvas;
    }

    // Look up the preloaded CLUT texture
    const clutTexture = this.clutTextures.get(clutImagePath);
    const level = this.clutLevels.get(clutImagePath);
    const clutWidth = this.clutWidths.get(clutImagePath);

    if (!clutTexture || !level || !clutWidth) {
      console.warn(
        "[HaldClut] CLUT not preloaded:",
        clutImagePath,
        "Call preloadClut() first.",
      );
      return sourceCanvas;
    }

    // Create source texture from canvas
    const sourceTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      sourceCanvas,
    );

    // Use the existing GL context's canvas for output
    this.resizeCanvas(sourceCanvas.width, sourceCanvas.height);

    // Clear the canvas
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use shader program
    gl.useProgram(this.program);

    // Set up geometry (fullscreen quad)
    const positions = new Float32Array([
      -1,
      -1, // bottom left
      1,
      -1, // bottom right
      -1,
      1, // top left
      1,
      1, // top right
    ]);

    const texCoords = new Float32Array([
      0,
      1, // bottom left (flipped Y)
      1,
      1, // bottom right
      0,
      0, // top left
      1,
      0, // top right
    ]);

    // Create and bind position buffer
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Create and bind texCoord buffer
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    const texCoordLoc = gl.getAttribLocation(this.program, "a_texCoord");
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const sourceTextureLoc = gl.getUniformLocation(
      this.program,
      "u_sourceTexture",
    );
    const clutTextureLoc = gl.getUniformLocation(this.program, "u_clutTexture");
    const clutSizeLoc = gl.getUniformLocation(this.program, "u_clutSize");
    const clutWidthLoc = gl.getUniformLocation(this.program, "u_clutWidth");
    const intensityLoc = gl.getUniformLocation(this.program, "u_intensity");


    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
    gl.uniform1i(sourceTextureLoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, clutTexture);
    gl.uniform1i(clutTextureLoc, 1);

    gl.uniform1f(clutSizeLoc, level);
    gl.uniform1f(clutWidthLoc, clutWidth);
    gl.uniform1f(intensityLoc, intensity);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Check for GL errors
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      if (
        (gl as any).CONTEXT_LOST_WEBGL &&
        error === (gl as any).CONTEXT_LOST_WEBGL
      ) {
        this.contextLost = true;
        console.warn("[HaldClut] WebGL context lost during draw");
      } else {
        console.error("WebGL error after drawing:", error);
      }
    }

    // Clean up temporary resources
    gl.deleteTexture(sourceTexture);
    gl.deleteBuffer(positionBuffer);
    gl.deleteBuffer(texCoordBuffer);

    return this.canvas;
  }

  public dispose() {
    const gl = this.gl;
    if (gl) {
      // Delete all CLUT textures
      for (const texture of this.clutTextures.values()) {
        gl.deleteTexture(texture);
      }
      if (this.program) {
        gl.deleteProgram(this.program);
        this.program = null;
      }
    }
    this.clutTextures.clear();
    this.clutLevels.clear();
    this.clutWidths.clear();
    this.loadedClutPaths.clear();
    super.dispose();
  }

  // Static method to clear the cache if needed
  public static clearCache() {
    clutImageCache.clear();
    loadingPromises.clear();
  }

  // Static method to get cache size
  public static getCacheSize(): number {
    return clutImageCache.size;
  }

  // Static method to remove specific item from cache
  public static removeCachedImage(imagePath: string): boolean {
    return clutImageCache.delete(imagePath);
  }

  // Instance method to check if a CLUT is loaded
  public isClutLoaded(clutImagePath: string): boolean {
    return this.clutTextures.has(clutImagePath);
  }

  // Instance method to get the number of loaded CLUTs
  public getLoadedClutCount(): number {
    return this.clutTextures.size;
  }

  // Instance method to get all loaded CLUT paths
  public getLoadedClutPaths(): string[] {
    return Array.from(this.clutTextures.keys());
  }
}

/**
 * NativeDecoder.ts — Worker-backed native decoder interface
 *
 * Calls the native decoder through the preload worker-thread bridge.
 * Used for scrubbing (decodeFrame) and playback (decodeNextFrame).
 */

import { useEffect, useRef, useState } from "react";
import {
  nativeDecoderLoadFile,
  nativeDecoderDecodeFrame,
  nativeDecoderDecodeNextFrame,
  type NativeDecoderFileInfo,
  fileURLToPath
} from "@app/preload";

export type FileInfo = NativeDecoderFileInfo;
export type DecodeResult = { view: Uint8Array; width: number; height: number };
export type DecodeNextFrameResult = DecodeResult & { timestamp: number };

// ─── WebGL shaders ───

const VERTEX_SHADER = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const FRAGMENT_SHADER = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  void main() {
    gl_FragColor = texture2D(u_texture, v_texCoord);
  }
`;

// ─── NativeDecoder class ───

class NativeDecoder {
  private _decoderId: string;
  private _filePath: string | null = null;
  private _info: FileInfo | null = null;
  private _loaded = false;
  private _width = 0;
  private _height = 0;


  // Canvas / WebGL
  private _canvas: HTMLCanvasElement | null = null;
  private _useWebGL = true;
  private _gl: WebGLRenderingContext | null = null;
  private _texture: WebGLTexture | null = null;
  private _program: WebGLProgram | null = null;
  private _vao: { position: WebGLBuffer; texCoord: WebGLBuffer } | null = null;
  private _imageData: ImageData | null = null;
  public _lastStartTime = -1;

  constructor(decoderId: string) {
    this._decoderId = decoderId;
  }

  get decoderId(): string {
    return this._decoderId;
  }
  get width(): number {
    return this._width;
  }
  get height(): number {
    return this._height;
  }
  get info(): FileInfo | null {
    return this._info;
  }
  get loaded(): boolean {
    return this._loaded;
  }

  // ─── Lifecycle ───

  async loadFile(filePath: string): Promise<FileInfo> {
    this._filePath = filePath;
    try {
      const info = await nativeDecoderLoadFile(filePath);
      this._info = info;
      this._width = info.video?.width ?? 0;
      this._height = info.video?.height ?? 0;
      this._loaded = true;
      this._lastStartTime = -1;
      return info;
    } catch (err) {
      this._loaded = false;
      throw err;
    }
  }

  get filePath(): string | null {
    return this._filePath;
  }

  destroy(): void {
    if (!this._loaded) return;
    this._loaded = false;
    this._info = null;
    this._filePath = null;
    this._width = 0;
    this._height = 0;
    this._lastStartTime = -1;
    this._resetCanvas();
  }

  // ─── One-shot decode (scrubbing) ───

  /**
   * Decode a single frame at the given timestamp.
   * Worker-thread backed call via preload bridge.
   */
  async decodeFrame(
    timestamp: number,
    keyframeOnly = false
  ): Promise<(DecodeResult & { timestamp: number }) | null> {
    if (!this._loaded || !this._filePath) return null;
    try {
      const { timestamp: ts, data } = await nativeDecoderDecodeFrame(
        this._filePath,
        this._width,
        this._height,
        timestamp,
        keyframeOnly
      );
      return {
        view: data,
        width: this._width,
        height: this._height,
        timestamp: ts,
      };
    } catch {
      return null;
    }
  }

  // ─── Sequential decode (playback) ───

  /**
   * Decode the next frame. For first frame after seek, pass startTime.
   * To continue from current position, pass -1 for startTime.
   * Returns null at EOF or when past endTime.
   */
  async decodeNextFrame(
    startTime: number,
    endTime: number
  ): Promise<DecodeNextFrameResult | null> {
    if (!this._loaded || !this._filePath) return null;
    try {
      const start = startTime >= 0 ? startTime : -1;
      const end = endTime >= 0 ? endTime : -1;
      const result = await nativeDecoderDecodeNextFrame(
        this._filePath,
        this._width,
        this._height,
        start,
        end
      );
      if (!result) return null;
      return {
        view: result.data,
        width: this._width,
        height: this._height,
        timestamp: result.timestamp,
      };
    } catch {
      return null;
    }
  }

  // ─── Canvas / WebGL rendering ───

  private _resetCanvas() {
    this._gl = null;
    this._texture = null;
    this._program = null;
    this._vao = null;
    this._imageData = null;
    this._useWebGL = true;
  }

  private _initWebGL(): boolean {
    if (!this._canvas) return false;
    if (this._gl) return true;
    const gl = this._canvas.getContext("webgl", {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      alpha: true,
    });
    if (!gl) return false;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, VERTEX_SHADER);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) return false;

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, FRAGMENT_SHADER);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) return false;

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return false;

    const texture = gl.createTexture()!;
    const positionBuffer = gl.createBuffer()!;
    const texCoordBuffer = gl.createBuffer()!;

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]),
      gl.STATIC_DRAW
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this._gl = gl;
    this._program = program;
    this._texture = texture;
    this._vao = { position: positionBuffer, texCoord: texCoordBuffer };
    return true;
  }

  /**
   * Render a frame (Uint8Array of RGBA) to the canvas.
   */
  renderFrame(view: Uint8Array): HTMLCanvasElement | null {
    if (this._width <= 0 || this._height <= 0) return null;

    if (!this._canvas) {
      this._canvas = document.createElement("canvas");
      this._canvas.width = this._width;
      this._canvas.height = this._height;
    } else if (
      this._canvas.width !== this._width ||
      this._canvas.height !== this._height
    ) {
      this._canvas.width = this._width;
      this._canvas.height = this._height;
      this._resetCanvas();
    }

    if (this._useWebGL) {
      if (!this._gl && !this._initWebGL()) this._useWebGL = false;
      if (this._useWebGL && this._renderWebGL(view)) return this._canvas;
      this._useWebGL = false;
    }

    if (this._render2D(view)) return this._canvas;
    return null;
  }

  getCanvas(view?: Uint8Array): HTMLCanvasElement | null {
    if (!view) return null;
    return this.renderFrame(view);
  }

  private _renderWebGL(view: Uint8Array): boolean {
    const gl = this._gl;
    if (!gl || !this._program || !this._texture || !this._vao) return false;

    gl.bindTexture(gl.TEXTURE_2D, this._texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      this._width,
      this._height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      view
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.viewport(0, 0, this._width, this._height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this._program);

    const posLoc = gl.getAttribLocation(this._program, "a_position");
    const texLoc = gl.getAttribLocation(this._program, "a_texCoord");
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._texture);
    gl.uniform1i(gl.getUniformLocation(this._program, "u_texture"), 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this._vao.position);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._vao.texCoord);
    gl.enableVertexAttribArray(texLoc);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(texLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return true;
  }

  private _render2D(view: Uint8Array): boolean {
    if (!this._canvas) return false;
    const ctx = this._canvas.getContext("2d");
    if (!ctx) return false;
    if (
      !this._imageData ||
      this._imageData.width !== this._width ||
      this._imageData.height !== this._height
    ) {
      this._imageData = ctx.createImageData(this._width, this._height);
    }
    this._imageData.data.set(view);
    ctx.putImageData(this._imageData, 0, 0);
    return true;
  }
}

export default NativeDecoder;

// ─── React hooks ───

export function useNativeDecoder(
  decoderId: string,
  filePath: string
): {
  decoder: NativeDecoder | null;
  info: FileInfo | null;
  loading: boolean;
  error: string | null;
} {
  const [decoder, setDecoder] = useState<NativeDecoder | null>(null);
  const [info, setInfo] = useState<FileInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const decoderRef = useRef<NativeDecoder | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!filePath || filePath.trim() === "") {
      setDecoder(null);
      setInfo(null);
      setLoading(false);
      setError(null);
      return;
    }
    const dec = new NativeDecoder(decoderId);
    decoderRef.current = dec;

    void (async () => {
      try {
        setLoading(true);
        setError(null);
        // convert file url to local path
        const localPath = fileURLToPath(filePath);
        const fileInfo = await dec.loadFile(localPath);
        if (cancelled) {
          dec.destroy();
          return;
        }
        setDecoder(dec);
        setInfo(fileInfo);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load";
        console.warn("[native-decoder] loadFile failed:", msg);
        if (!cancelled) setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      decoderRef.current?.destroy();
      decoderRef.current = null;
      setDecoder(null);
      setInfo(null);
    };
  }, [decoderId, filePath]);

  return { decoder, info, loading, error };
}

export function makeDecoderId(
  id: string,
  clipId: string,
  opts?: { decoderKey?: string; inputMode?: boolean; inputId?: string }
): string {
  const logicalClipKey = opts?.decoderKey ?? clipId;
  if (opts?.inputMode && opts?.inputId) {
    return `${id}::${logicalClipKey}::input::${opts.inputId}`;
  }
  return `${id}::${logicalClipKey}`;
}

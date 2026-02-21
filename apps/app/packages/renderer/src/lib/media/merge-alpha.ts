

// ─── Add these fields to AssetState ───────────────────────────────────────
// mergeGlCanvas:   OffscreenCanvas | null
// mergeGl:         WebGL2RenderingContext | null
// mergeGlProgram:  WebGLProgram | null
// mergeGlTexColor: WebGLTexture | null
// mergeGlTexAlpha: WebGLTexture | null
// mergeGlInitialized: boolean

import { EncodedPacketSink, Input, EncodedPacket} from "mediabunny";



// State
type AssetState = {
    decoder: VideoDecoder | null;
    alphaDecoder: VideoDecoder | null;
    sink: EncodedPacketSink | null;
    input: Input | null;
    mergeGlCanvas:      OffscreenCanvas | null;
    mergeGl:            WebGL2RenderingContext | null;
    mergeGlProgram:     WebGLProgram | null;
    mergeGlTexColor:    WebGLTexture | null;
    mergeGlTexAlpha:    WebGLTexture | null;
    mergeGlInitialized: boolean;
  
  // Caching
    cachedDecodedFrames: Map<number, VideoFrame>;
    keyPacketCache: Map<number, EncodedPacket>;
    isCachingKeyPackets: boolean;
  
    // Alpha merge (for codecs where alpha is stored separately in packet sideData)
    alphaFramesByTimestamp: Map<number, VideoFrame>;
    pendingColorFramesByTimestamp: Map<
      number,
      {
        frame: VideoFrame;
        requestId: number;
      }
    >;
    // Reused canvases to avoid reallocating for every frame
    mergeCanvas: OffscreenCanvas | null;
    mergeCtx: OffscreenCanvasRenderingContext2D | null;
    alphaCanvas: OffscreenCanvas | null;
    alphaCtx: OffscreenCanvasRenderingContext2D | null;
  
    // Seek state
    seekTargetTimestamp: number | null;
    seekDone: boolean;
    currentRequestId: number;
    lastSeekTime: number;
    lastSeekTimestamp: number;
    showingPreview: boolean;
    config: VideoDecoderConfig | null;
    pendingSeekFrame: VideoFrame | null;
    pendingSeekFrameTime: number;
  
    // Iteration flow control
    iterationInFlight: number;
    iterationResume: (() => void) | null;
  
    // Output Handling with dynamic dispatch
    customOutputHandler: ((frame: VideoFrame) => void) | null;
  };

export const VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x, -a_pos.y) * 0.5 + 0.5; // flip Y for VideoFrame orientation
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG_SRC = `#version 300 es
precision mediump float;
uniform sampler2D u_color;
uniform sampler2D u_alpha;
in vec2 v_uv;
out vec4 fragColor;
void main() {
  vec4 c = texture(u_color, v_uv);
  float a = texture(u_alpha, v_uv).r;
  fragColor = vec4(c.rgb, a);
}`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`Shader compile error: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

function ensureMergeGl(state: AssetState, width: number, height: number): boolean {
  // Create canvas and context if not yet done
  if (!state.mergeGlCanvas) {
    state.mergeGlCanvas = new OffscreenCanvas(width, height);
    state.mergeGl = state.mergeGlCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
    state.mergeGlProgram = null;
    state.mergeGlTexColor = null;
    state.mergeGlTexAlpha = null;
    state.mergeGlInitialized = false;
  }

  if (!state.mergeGl) return false;
  const gl = state.mergeGl;

  // Resize if needed
  if (state.mergeGlCanvas.width !== width)  state.mergeGlCanvas.width  = width;
  if (state.mergeGlCanvas.height !== height) state.mergeGlCanvas.height = height;

  // Compile and link program once
  if (!state.mergeGlInitialized) {
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER,   VERT_SRC));
    gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`WebGL program link error: ${gl.getProgramInfoLog(prog)}`);
    }
    state.mergeGlProgram = prog;
    gl.useProgram(prog);

    // Full-screen quad
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, 1]),
      gl.STATIC_DRAW,
    );
    const posLoc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Bind samplers to texture units
    gl.uniform1i(gl.getUniformLocation(prog, 'u_color'), 0);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_alpha'), 1);

    // Create reusable textures
    const makeTexture = () => {
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return tex;
    };
    state.mergeGlTexColor = makeTexture();
    state.mergeGlTexAlpha = makeTexture();

    state.mergeGlInitialized = true;
  }

  gl.viewport(0, 0, width, height);
  return true;
}

export function mergeAlphaIntoColor(
  state: AssetState,
  colorFrame: VideoFrame,
  alphaFrame: VideoFrame,
): VideoFrame {
  const width  = colorFrame.displayWidth  || (colorFrame as any).codedWidth  || 0;
  const height = colorFrame.displayHeight || (colorFrame as any).codedHeight || 0;

  if (!width || !height) return colorFrame;

  if (!ensureMergeGl(state, width, height)) {
    // WebGL2 unavailable — fall back to original 2D canvas path
    return mergeAlphaIntoColorFallback(state, colorFrame, alphaFrame, width, height);
  }

  const gl = state.mergeGl!;

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.mergeGlTexColor!);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, colorFrame as any);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.mergeGlTexAlpha!);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, alphaFrame as any);

  gl.drawArrays(gl.TRIANGLES, 0, 6);

  return new VideoFrame(state.mergeGlCanvas as any, {
    timestamp: colorFrame.timestamp,
    duration:  colorFrame.duration ?? undefined,
  });
}


function ensureMergeCanvases(state: AssetState, width: number, height: number) {
    if (
      !state.mergeCanvas ||
      state.mergeCanvas.width !== width ||
      state.mergeCanvas.height !== height
    ) {
      state.mergeCanvas = new OffscreenCanvas(width, height);
      state.mergeCtx = state.mergeCanvas.getContext("2d", {
        willReadFrequently: true,
      }) as OffscreenCanvasRenderingContext2D | null;
    }
    if (
      !state.alphaCanvas ||
      state.alphaCanvas.width !== width ||
      state.alphaCanvas.height !== height
    ) {
      state.alphaCanvas = new OffscreenCanvas(width, height);
      state.alphaCtx = state.alphaCanvas.getContext("2d", {
        willReadFrequently: true,
      }) as OffscreenCanvasRenderingContext2D | null;
    }
  }
  

// Original implementation kept as fallback
function mergeAlphaIntoColorFallback(
  state: AssetState,
  colorFrame: VideoFrame,
  alphaFrame: VideoFrame,
  width: number,
  height: number,
): VideoFrame {
  ensureMergeCanvases(state, width, height);
  const ctx  = state.mergeCtx;
  const aCtx = state.alphaCtx;
  if (!ctx || !aCtx || !state.mergeCanvas || !state.alphaCanvas) return colorFrame;

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(colorFrame as any, 0, 0, width, height);
  const colorImage = ctx.getImageData(0, 0, width, height);

  aCtx.clearRect(0, 0, width, height);
  aCtx.drawImage(alphaFrame as any, 0, 0, width, height);
  const alphaImage = aCtx.getImageData(0, 0, width, height);

  const c = colorImage.data;
  const a = alphaImage.data;
  for (let i = 0; i < c.length; i += 4) {
    c[i + 3] = a[i];
  }
  ctx.putImageData(colorImage, 0, 0);

  return new VideoFrame(state.mergeCanvas, {
    timestamp: colorFrame.timestamp,
    duration:  colorFrame.duration ?? undefined,
  });
}
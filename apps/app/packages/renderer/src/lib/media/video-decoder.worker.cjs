"use strict";
(() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));

  // packages/renderer/src/lib/media/video-decoder.worker.ts
  var import_mediabunny = __require("mediabunny");

  // packages/renderer/src/lib/media/merge-alpha.ts
  var VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x, -a_pos.y) * 0.5 + 0.5; // flip Y for VideoFrame orientation
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;
  var FRAG_SRC = `#version 300 es
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
  function compileShader(gl, type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`Shader compile error: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }
  function ensureMergeGl(state, width, height) {
    if (!state.mergeGlCanvas) {
      state.mergeGlCanvas = new OffscreenCanvas(width, height);
      state.mergeGl = state.mergeGlCanvas.getContext("webgl2");
      state.mergeGlProgram = null;
      state.mergeGlTexColor = null;
      state.mergeGlTexAlpha = null;
      state.mergeGlInitialized = false;
    }
    if (!state.mergeGl) return false;
    const gl = state.mergeGl;
    if (state.mergeGlCanvas.width !== width) state.mergeGlCanvas.width = width;
    if (state.mergeGlCanvas.height !== height) state.mergeGlCanvas.height = height;
    if (!state.mergeGlInitialized) {
      const prog = gl.createProgram();
      gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, VERT_SRC));
      gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(`WebGL program link error: ${gl.getProgramInfoLog(prog)}`);
      }
      state.mergeGlProgram = prog;
      gl.useProgram(prog);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        gl.STATIC_DRAW
      );
      const posLoc = gl.getAttribLocation(prog, "a_pos");
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
      gl.uniform1i(gl.getUniformLocation(prog, "u_color"), 0);
      gl.uniform1i(gl.getUniformLocation(prog, "u_alpha"), 1);
      const makeTexture = () => {
        const tex = gl.createTexture();
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
  function mergeAlphaIntoColor(state, colorFrame, alphaFrame) {
    const width = colorFrame.displayWidth || colorFrame.codedWidth || 0;
    const height = colorFrame.displayHeight || colorFrame.codedHeight || 0;
    if (!width || !height) return colorFrame;
    if (!ensureMergeGl(state, width, height)) {
      return mergeAlphaIntoColorFallback(state, colorFrame, alphaFrame, width, height);
    }
    const gl = state.mergeGl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, state.mergeGlTexColor);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, colorFrame);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, state.mergeGlTexAlpha);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, alphaFrame);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    return new VideoFrame(state.mergeGlCanvas, {
      timestamp: colorFrame.timestamp,
      duration: colorFrame.duration ?? void 0
    });
  }
  function ensureMergeCanvases(state, width, height) {
    if (!state.mergeCanvas || state.mergeCanvas.width !== width || state.mergeCanvas.height !== height) {
      state.mergeCanvas = new OffscreenCanvas(width, height);
      state.mergeCtx = state.mergeCanvas.getContext("2d", {
        willReadFrequently: true
      });
    }
    if (!state.alphaCanvas || state.alphaCanvas.width !== width || state.alphaCanvas.height !== height) {
      state.alphaCanvas = new OffscreenCanvas(width, height);
      state.alphaCtx = state.alphaCanvas.getContext("2d", {
        willReadFrequently: true
      });
    }
  }
  function mergeAlphaIntoColorFallback(state, colorFrame, alphaFrame, width, height) {
    ensureMergeCanvases(state, width, height);
    const ctx = state.mergeCtx;
    const aCtx = state.alphaCtx;
    if (!ctx || !aCtx || !state.mergeCanvas || !state.alphaCanvas) return colorFrame;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(colorFrame, 0, 0, width, height);
    const colorImage = ctx.getImageData(0, 0, width, height);
    aCtx.clearRect(0, 0, width, height);
    aCtx.drawImage(alphaFrame, 0, 0, width, height);
    const alphaImage = aCtx.getImageData(0, 0, width, height);
    const c = colorImage.data;
    const a = alphaImage.data;
    for (let i = 0; i < c.length; i += 4) {
      c[i + 3] = a[i];
    }
    ctx.putImageData(colorImage, 0, 0);
    return new VideoFrame(state.mergeCanvas, {
      timestamp: colorFrame.timestamp,
      duration: colorFrame.duration ?? void 0
    });
  }

  // packages/renderer/src/lib/media/video-decoder.worker.ts
  var nodeFs = __toESM(__require("node:fs/promises"), 1);
  function assert(x) {
    if (!x) {
      throw new Error("Assertion failed.");
    }
  }
  function fileURLToPathInWorker(raw) {
    try {
      const u = new URL(raw);
      if (u.protocol === "file:" || u.protocol === "app:") {
        const decoded = decodeURIComponent(u.pathname);
        if (/^\/[A-Za-z]:\//.test(decoded)) {
          return decoded.slice(1);
        }
        return decoded;
      }
      return raw;
    } catch {
      return raw;
    }
  }
  var assetStates = /* @__PURE__ */ new Map();
  var packetSinks = /* @__PURE__ */ new Map();
  var MAX_CACHE_SIZE = 16;
  var MAX_ITERATION_IN_FLIGHT = 4;
  var MAX_DECODE_QUEUE_SIZE = 8;
  var MAX_DECODE_QUEUE_WAIT_MS = 500;
  var MAX_RESYNC_ATTEMPTS = 4;
  var DECODE_QUEUE_SLEEP_MS = 5;
  var KEYFRAME_REQUIRED_RE = /key\s*frame/i;
  var DIAGNOSTIC_TICK_MS = 2e3;
  try {
    postMessage({
      type: "debug",
      scope: "video-decoder-worker",
      event: "worker-loaded"
    });
  } catch {
  }
  function getDecoderState(decoder) {
    if (!decoder) return "none";
    return decoder.state || "unknown";
  }
  function getOrCreatePacketSink(assetId, videoTrack) {
    let sink = packetSinks.get(assetId);
    if (!sink) {
      sink = new import_mediabunny.EncodedPacketSink(videoTrack);
      packetSinks.set(assetId, sink);
    }
    return sink;
  }
  function createNodeFileSource(filePath) {
    let fileHandle = null;
    let knownSize = null;
    return new import_mediabunny.StreamSource({
      getSize: async () => {
        if (!fileHandle) {
          fileHandle = await nodeFs.open(filePath, "r");
        }
        if (knownSize == null) {
          const stats = await fileHandle.stat();
          knownSize = Number(stats.size);
        }
        return knownSize;
      },
      read: async (start, end) => {
        if (!fileHandle) {
          fileHandle = await nodeFs.open(filePath, "r");
        }
        const length = Math.max(0, end - start);
        const buffer = new Uint8Array(length);
        if (length === 0) return buffer;
        const readResult = await fileHandle.read(buffer, 0, length, start);
        if (readResult.bytesRead === length) return buffer;
        return buffer.subarray(0, readResult.bytesRead);
      },
      dispose: async () => {
        if (!fileHandle) return;
        try {
          await fileHandle.close();
        } catch {
        } finally {
          fileHandle = null;
        }
      },
      prefetchProfile: "fileSystem"
    });
  }
  function emitDiagnostics(event, assetId, requestId) {
    const assets = [];
    let totalFrameCache = 0;
    let totalKeyPacketCache = 0;
    let totalPendingAlpha = 0;
    let totalPendingColor = 0;
    let totalIterationInFlight = 0;
    for (const [id, state] of assetStates) {
      const frameCache = state.cachedDecodedFrames.size;
      const keyPacketCache = state.keyPacketCache.size;
      const pendingAlpha = state.alphaFramesByTimestamp.size;
      const pendingColor = state.pendingColorFramesByTimestamp.size;
      const iterationInFlight = state.iterationInFlight;
      totalFrameCache += frameCache;
      totalKeyPacketCache += keyPacketCache;
      totalPendingAlpha += pendingAlpha;
      totalPendingColor += pendingColor;
      totalIterationInFlight += iterationInFlight;
      assets.push({
        assetId: id,
        frameCache,
        keyPacketCache,
        pendingAlpha,
        pendingColor,
        iterationInFlight,
        seekTargetTimestamp: state.seekTargetTimestamp,
        decoderState: getDecoderState(state.decoder),
        decoderQueueSize: state.decoder && state.decoder.state !== "closed" ? state.decoder.decodeQueueSize : 0,
        alphaDecoderState: getDecoderState(state.alphaDecoder),
        alphaDecoderQueueSize: state.alphaDecoder && state.alphaDecoder.state !== "closed" ? state.alphaDecoder.decodeQueueSize : 0,
        showingPreview: state.showingPreview,
        customOutputHandler: !!state.customOutputHandler,
        hasPendingSeekFrame: !!state.pendingSeekFrame,
        currentRequestId: state.currentRequestId
      });
    }
    try {
      postMessage({
        type: "debug",
        scope: "video-decoder-worker",
        event: "diag-state",
        assetId,
        requestId,
        payload: {
          event,
          nowMs: performance.now(),
          assetCount: assetStates.size,
          totals: {
            frameCache: totalFrameCache,
            keyPacketCache: totalKeyPacketCache,
            pendingAlpha: totalPendingAlpha,
            pendingColor: totalPendingColor,
            iterationInFlight: totalIterationInFlight
          },
          assets
        }
      });
    } catch {
    }
  }
  setInterval(() => {
    emitDiagnostics("tick");
  }, DIAGNOSTIC_TICK_MS);
  function getOrCreateState(assetId) {
    let state = assetStates.get(assetId);
    if (!state) {
      state = {
        decoder: null,
        alphaDecoder: null,
        sink: null,
        input: null,
        mergeGlCanvas: null,
        mergeGl: null,
        mergeGlProgram: null,
        mergeGlTexColor: null,
        mergeGlTexAlpha: null,
        mergeGlInitialized: false,
        cachedDecodedFrames: /* @__PURE__ */ new Map(),
        keyPacketCache: /* @__PURE__ */ new Map(),
        isCachingKeyPackets: false,
        alphaFramesByTimestamp: /* @__PURE__ */ new Map(),
        pendingColorFramesByTimestamp: /* @__PURE__ */ new Map(),
        mergeCanvas: null,
        mergeCtx: null,
        alphaCanvas: null,
        alphaCtx: null,
        seekTargetTimestamp: null,
        seekDone: false,
        currentRequestId: 0,
        lastSeekTime: 0,
        lastSeekTimestamp: 0,
        showingPreview: false,
        config: null,
        pendingSeekFrame: null,
        pendingSeekFrameTime: 0,
        iterationInFlight: 0,
        iterationResume: null,
        customOutputHandler: null
      };
      assetStates.set(assetId, state);
    }
    return state;
  }
  function resetAlphaMergeQueues(state) {
    for (const f of state.alphaFramesByTimestamp.values()) f.close();
    state.alphaFramesByTimestamp.clear();
    for (const v of state.pendingColorFramesByTimestamp.values()) v.frame.close();
    state.pendingColorFramesByTimestamp.clear();
  }
  function createAlphaFrameHandler(assetId) {
    return (alphaFrame) => {
      const state = assetStates.get(assetId);
      if (!state) {
        alphaFrame.close();
        return;
      }
      const ts = alphaFrame.timestamp;
      const pending = state.pendingColorFramesByTimestamp.get(ts);
      if (pending) {
        state.pendingColorFramesByTimestamp.delete(ts);
        if (pending.requestId !== state.currentRequestId) {
          pending.frame.close();
          alphaFrame.close();
          return;
        }
        let merged = null;
        try {
          merged = mergeAlphaIntoColor(state, pending.frame, alphaFrame);
        } catch {
          merged = pending.frame;
        } finally {
          if (merged !== pending.frame) pending.frame.close();
          alphaFrame.close();
        }
        dispatchDecodedFrame(assetId, merged);
        return;
      }
      if (state.alphaFramesByTimestamp.size > 120) {
        const firstKey = state.alphaFramesByTimestamp.keys().next().value;
        if (firstKey !== void 0) {
          state.alphaFramesByTimestamp.get(firstKey)?.close();
          state.alphaFramesByTimestamp.delete(firstKey);
        }
      }
      state.alphaFramesByTimestamp.set(ts, alphaFrame);
    };
  }
  function ensureAlphaDecoder(state, assetId) {
    if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
      return;
    }
    if (!state.config) {
      return;
    }
    const alphaDecoder = new VideoDecoder({
      output: createAlphaFrameHandler(assetId),
      error: (e) => {
        console.error("Alpha VideoDecoder error", e);
        if (state.alphaDecoder === alphaDecoder) {
          try {
            alphaDecoder.close();
          } catch {
          }
          state.alphaDecoder = null;
        }
        const suppressError = state.seekTargetTimestamp !== null || state.customOutputHandler !== null;
        if (!suppressError) {
          try {
            postMessage({
              type: "error",
              error: e.message ?? "Alpha VideoDecoder error",
              assetId
            });
          } catch {
          }
        }
      }
    });
    state.alphaDecoder = alphaDecoder;
    try {
      const cfgAny = { ...state.config };
      delete cfgAny.alpha;
      state.alphaDecoder.configure(cfgAny);
    } catch (e) {
      try {
        state.alphaDecoder.configure(state.config);
      } catch (e2) {
        console.error("Alpha VideoDecoder configure failed", e, e2);
        try {
          state.alphaDecoder.close();
        } catch {
        }
        state.alphaDecoder = null;
      }
    }
  }
  function dispatchDecodedFrame(assetId, frame) {
    const state = assetStates.get(assetId);
    if (!state) {
      frame.close();
      return;
    }
    if (state.customOutputHandler) {
      state.customOutputHandler(frame);
      return;
    }
    const frameTime = frame.timestamp / 1e6;
    cacheFrame(state, frame);
    if (state.showingPreview) {
      postFrame(assetId, frame, state.currentRequestId);
      state.showingPreview = false;
    }
    if (state.seekTargetTimestamp !== null) {
      const distance = Math.abs(frameTime - state.seekTargetTimestamp);
      if (!state.pendingSeekFrame || distance < Math.abs(state.pendingSeekFrameTime - state.seekTargetTimestamp)) {
        if (state.pendingSeekFrame) {
          state.pendingSeekFrame.close();
        }
        state.pendingSeekFrame = frame.clone();
        state.pendingSeekFrameTime = frameTime;
      }
      if (frameTime >= state.seekTargetTimestamp - 0.04) {
        state.seekDone = true;
        postFrame(assetId, frame, state.currentRequestId);
        state.seekTargetTimestamp = null;
        if (state.pendingSeekFrame) {
          state.pendingSeekFrame.close();
          state.pendingSeekFrame = null;
        }
      }
    }
    frame.close();
  }
  function findCachedFrame(state, timestamp) {
    for (const [t, frame] of state.cachedDecodedFrames) {
      if (Math.abs(t - timestamp) < 0.05) return frame;
    }
    return null;
  }
  function cacheFrame(state, frame) {
    const frameTime = frame.timestamp / 1e6;
    if (state.cachedDecodedFrames.has(frameTime)) return;
    if (state.cachedDecodedFrames.size >= MAX_CACHE_SIZE) {
      const firstKey = state.cachedDecodedFrames.keys().next().value;
      if (firstKey !== void 0) {
        state.cachedDecodedFrames.get(firstKey)?.close();
        state.cachedDecodedFrames.delete(firstKey);
      }
    }
    state.cachedDecodedFrames.set(frameTime, frame.clone());
  }
  async function cacheKeyPackets(state) {
    if (state.isCachingKeyPackets) return;
    state.isCachingKeyPackets = true;
    try {
    } catch (e) {
      console.warn("Background keyframe caching failed", e);
    }
  }
  var createFrameHandler = (assetId) => (frame) => {
    const state = assetStates.get(assetId);
    if (!state) {
      frame.close();
      return;
    }
    if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
      const ts = frame.timestamp;
      const alpha = state.alphaFramesByTimestamp.get(ts);
      if (alpha) {
        state.alphaFramesByTimestamp.delete(ts);
        let merged = null;
        try {
          merged = mergeAlphaIntoColor(state, frame, alpha);
        } catch {
          merged = frame;
        } finally {
          if (merged !== frame) frame.close();
          alpha.close();
        }
        dispatchDecodedFrame(assetId, merged);
        return;
      }
      state.pendingColorFramesByTimestamp.set(ts, {
        frame,
        requestId: state.currentRequestId
      });
      if (state.pendingColorFramesByTimestamp.size > 120) {
        const firstKey = state.pendingColorFramesByTimestamp.keys().next().value;
        if (firstKey !== void 0) {
          const v = state.pendingColorFramesByTimestamp.get(firstKey);
          v?.frame.close();
          state.pendingColorFramesByTimestamp.delete(firstKey);
        }
      }
      return;
    }
    dispatchDecodedFrame(assetId, frame);
  };
  function createVideoDecoder(state, assetId) {
    const decoder = new VideoDecoder({
      output: createFrameHandler(assetId),
      error: (e) => {
        console.error("VideoDecoder error", e);
        if (state.decoder === decoder) {
          try {
            decoder.close();
          } catch {
          }
          state.decoder = null;
        }
        const suppressError = state.seekTargetTimestamp !== null || state.customOutputHandler !== null;
        if (!suppressError) {
          try {
            postMessage({
              type: "error",
              error: e.message,
              assetId
            });
          } catch {
          }
        }
      }
    });
    state.decoder = decoder;
  }
  function ensureDecoderInstance(state, assetId) {
    if (state.decoder && state.decoder.state !== "closed") {
      return state.decoder;
    }
    try {
      state.decoder?.close();
    } catch {
    }
    createVideoDecoder(state, assetId);
    return state.decoder;
  }
  function postFrame(assetId, frame, reqId) {
    const clone = frame.clone();
    const msg = {
      type: "frame",
      frame: clone,
      timestamp: clone.timestamp / 1e6,
      duration: (clone.duration ?? 0) / 1e6,
      requestId: reqId,
      assetId
    };
    postMessage(msg, [clone]);
  }
  var sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  function isKeyframeRequiredError(err) {
    const msg = (err?.message ?? "").toString();
    return err?.name === "DataError" && KEYFRAME_REQUIRED_RE.test(msg);
  }
  function isQuotaExceededError(err) {
    return err?.name === "QuotaExceededError";
  }
  function isInvalidStateError(err) {
    const msg = (err?.message ?? "").toString();
    return err?.name === "InvalidStateError" || /closed/i.test(msg);
  }
  async function waitForDecodeQueue(decoder, maxSize = MAX_DECODE_QUEUE_SIZE) {
    let waited = 0;
    while (decoder.decodeQueueSize > maxSize && waited < MAX_DECODE_QUEUE_WAIT_MS) {
      await sleep(DECODE_QUEUE_SLEEP_MS);
      waited += DECODE_QUEUE_SLEEP_MS;
    }
  }
  async function getVerifiedKeyPacket(state, timestamp) {
    if (!state.sink) return null;
    try {
      const keyPacket = await state.sink.getKeyPacket(timestamp, {
        verifyKeyPackets: true
      });
      if (keyPacket) return keyPacket;
      const nearby = await state.sink.getPacket(timestamp);
      if (nearby) {
        const nextKey = await state.sink.getNextKeyPacket(nearby, {
          verifyKeyPackets: true
        });
        if (nextKey) return nextKey;
      }
      return await state.sink.getFirstPacket({ verifyKeyPackets: true });
    } catch {
      return null;
    }
  }
  async function getNextKeyPacketSafe(state, packet) {
    if (!state.sink) return null;
    try {
      let nextPacket = await state.sink.getNextKeyPacket(packet, {
        verifyKeyPackets: true
      });
      if (!nextPacket) {
        const fallbackPacket = await state.sink.getKeyPacket(packet.timestamp, {
          verifyKeyPackets: true
        });
        nextPacket = fallbackPacket ?? null;
      }
      return nextPacket;
    } catch {
      return null;
    }
  }
  async function flushDecoderIfNeeded(decoder) {
    if (!decoder || decoder.state !== "configured") return;
    if (decoder.decodeQueueSize === 0) return;
    try {
      await decoder.flush();
    } catch {
    }
  }
  function resetAndConfigureDecoders(state, assetId) {
    if (!state.config) return false;
    const decoder = ensureDecoderInstance(state, assetId);
    if (!decoder) return false;
    try {
      decoder.reset();
    } catch {
    }
    try {
      decoder.configure(state.config);
    } catch (e) {
      try {
        const fallbackConfig = { ...state.config };
        delete fallbackConfig.alpha;
        state.config = fallbackConfig;
        decoder.configure(fallbackConfig);
      } catch {
        return false;
      }
    }
    if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
      try {
        state.alphaDecoder.reset();
        const cfgAny = { ...state.config };
        delete cfgAny.alpha;
        state.alphaDecoder.configure(cfgAny);
      } catch {
      }
    }
    return true;
  }
  async function decodePacketSafe(state, assetId, packet, requestId) {
    if (!state.decoder || state.decoder.state === "closed") {
      return false;
    }
    for (let attempt = 0; attempt < 2; attempt++) {
      if (state.currentRequestId !== requestId) return false;
      await waitForDecodeQueue(state.decoder, MAX_DECODE_QUEUE_SIZE);
      if (!state.decoder || state.decoder.state === "closed") {
        return false;
      }
      try {
        state.decoder.decode(packet.toEncodedVideoChunk());
        break;
      } catch (e) {
        if (isQuotaExceededError(e)) {
          await waitForDecodeQueue(state.decoder, 2);
          continue;
        }
        if (isKeyframeRequiredError(e) || isInvalidStateError(e)) {
          return false;
        }
        return false;
      }
    }
    if (packet.sideData?.alpha) {
      ensureAlphaDecoder(state, assetId);
      if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
        await waitForDecodeQueue(state.alphaDecoder, MAX_DECODE_QUEUE_SIZE);
        try {
          state.alphaDecoder.decode(packet.alphaToEncodedVideoChunk());
        } catch {
        }
      }
    }
    return true;
  }
  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      postMessage({
        type: "debug",
        scope: "video-decoder-worker",
        event: "onmessage",
        assetId: msg.assetId,
        requestId: msg.requestId,
        payload: { type: msg.type }
      });
    } catch {
    }
    try {
      switch (msg.type) {
        case "configure": {
          await handleConfigure(msg, msg.requestId);
          break;
        }
        case "seek": {
          await handleSeek(msg.timestamp, msg.forceAccurate, msg.requestId, msg.assetId);
          break;
        }
        case "iterate": {
          await handleIterate(msg.startTime, msg.endTime, msg.requestId, msg.assetId);
          break;
        }
        case "ack": {
          const id = msg.assetId;
          if (!id) break;
          const state = assetStates.get(id);
          if (!state) break;
          if (msg.requestId === state.currentRequestId) {
            state.iterationInFlight--;
            if (state.iterationInFlight < MAX_ITERATION_IN_FLIGHT && state.iterationResume) {
              state.iterationResume();
              state.iterationResume = null;
            }
          }
          break;
        }
        case "dispose": {
          dispose(msg.assetId);
          break;
        }
      }
    } catch (err) {
      console.error("Worker Error:", err);
      postMessage({
        type: "error",
        error: err.message,
        // Some messages don't carry requestId; fall back to 0 in that case.
        requestId: msg.requestId || 0,
        assetId: msg.assetId
      });
    }
  };
  async function handleConfigure(msg, requestId) {
    const { assetId, config: cfg } = msg;
    const id = assetId ?? cfg.asset.id;
    if (!id) {
      throw new Error("configure message missing asset identifier");
    }
    dispose(id);
    const state = getOrCreateState(id);
    let formats = import_mediabunny.ALL_FORMATS;
    let input = null;
    const localFilePath = fileURLToPathInWorker(cfg.asset.path);
    input = new import_mediabunny.Input({ formats, source: createNodeFileSource(localFilePath) });
    state.input = input;
    const videoTrack = await state.input.getPrimaryVideoTrack();
    if (!videoTrack) throw new Error("No video track found in worker");
    state.sink = getOrCreatePacketSink(id, videoTrack);
    const configAny = {
      ...cfg.videoDecoderConfig,
      optimizeForLatency: true
    };
    if (configAny.alpha == null) {
      configAny.alpha = "keep";
    }
    state.config = configAny;
    createVideoDecoder(state, id);
    try {
      state?.decoder?.configure(state.config);
    } catch (e) {
      try {
        const fallbackConfig = { ...state.config };
        delete fallbackConfig.alpha;
        state.config = fallbackConfig;
        state?.decoder?.configure(fallbackConfig);
      } catch (e2) {
        console.error("VideoDecoder configure failed", e, e2);
        postMessage({
          type: "error",
          error: e2?.message ?? "VideoDecoder configure failed",
          assetId: id
        });
        return;
      }
    }
    void cacheKeyPackets(state);
    emitDiagnostics("configure", id, requestId);
    postMessage({
      type: "ready",
      requestId,
      assetId: id
    });
  }
  async function handleSeek(timestamp, forceAccurate, requestId, assetId) {
    const id = assetId;
    if (!id) return;
    const state = assetStates.get(id);
    if (!state || !state.sink || !state.config) return;
    ensureDecoderInstance(state, id);
    if (!state.decoder) return;
    state.customOutputHandler = null;
    if (state.iterationResume) {
      state.iterationResume();
      state.iterationResume = null;
    }
    state.currentRequestId = requestId;
    resetAlphaMergeQueues(state);
    if (state.pendingSeekFrame) {
      state.pendingSeekFrame.close();
      state.pendingSeekFrame = null;
    }
    const benchmarkStart = performance.now();
    const now = performance.now();
    const timeSinceLast = now - state.lastSeekTime;
    const dist = Math.abs(timestamp - (state.lastSeekTimestamp || 0));
    state.lastSeekTime = now;
    state.lastSeekTimestamp = timestamp;
    const isFastScrubbing = !forceAccurate && timeSinceLast < 150 && dist > 0.5;
    const cached = findCachedFrame(state, timestamp);
    if (cached) {
      const cacheHitMs = performance.now() - benchmarkStart;
      try {
        postMessage({
          type: "debug",
          scope: "video-decoder-worker",
          event: "decodeBenchmark",
          assetId: id,
          requestId,
          payload: {
            fromCache: true,
            totalMs: cacheHitMs,
            packetsDecoded: 0,
            targetTimestamp: timestamp
          }
        });
      } catch {
      }
      postFrame(id, cached, requestId);
      postMessage({
        type: "seekDone",
        requestId,
        assetId: id
      });
      return;
    }
    state.seekTargetTimestamp = timestamp;
    state.seekDone = false;
    state.showingPreview = false;
    await flushDecoderIfNeeded(state.decoder);
    await flushDecoderIfNeeded(state.alphaDecoder);
    const keyPacketStart = performance.now();
    const initialPacket = await getVerifiedKeyPacket(state, timestamp);
    const keyPacketMs = performance.now() - keyPacketStart;
    if (!initialPacket) {
      postMessage({
        type: "seekDone",
        requestId,
        assetId: id
      });
      return;
    }
    if (!state.keyPacketCache.has(initialPacket.timestamp)) {
      state.keyPacketCache.set(initialPacket.timestamp, initialPacket);
    }
    if (state.currentRequestId !== requestId) return;
    let currentPacket = initialPacket;
    let resyncAttempts = 0;
    let previewArmed = isFastScrubbing;
    let packetsDecoded = 0;
    const decodeLoopStart = performance.now();
    while (currentPacket && state.currentRequestId === requestId && resyncAttempts <= MAX_RESYNC_ATTEMPTS) {
      await flushDecoderIfNeeded(state.decoder);
      await flushDecoderIfNeeded(state.alphaDecoder);
      if (!resetAndConfigureDecoders(state, id)) break;
      const decodedKey = await decodePacketSafe(
        state,
        id,
        currentPacket,
        requestId
      );
      if (decodedKey) packetsDecoded++;
      if (!decodedKey) {
        const nextPacket = await getNextKeyPacketSafe(state, currentPacket);
        if (nextPacket) {
          currentPacket = nextPacket;
        } else {
          break;
        }
        resyncAttempts++;
        continue;
      }
      if (previewArmed) {
        state.showingPreview = true;
        await sleep(80);
        state.showingPreview = false;
        previewArmed = false;
      }
      let resyncPacket = null;
      try {
        const packets = state.sink.packets(currentPacket);
        await packets.next();
        for await (const packet of packets) {
          if (state.currentRequestId !== requestId) {
            break;
          }
          if (!state.decoder || state.decoder.state === "closed") {
            resyncPacket = await getNextKeyPacketSafe(state, packet);
            break;
          }
          if (packet.timestamp > timestamp + 0.1) {
            break;
          }
          const decoded = await decodePacketSafe(state, id, packet, requestId);
          if (decoded) packetsDecoded++;
          if (!decoded) {
            resyncPacket = await getNextKeyPacketSafe(state, packet);
            break;
          }
          if (state.seekDone) {
            break;
          }
        }
      } catch (e) {
        console.warn("Seek packet iteration failed", e);
      }
      if (!resyncPacket || state.seekDone || state.currentRequestId !== requestId) {
        break;
      }
      currentPacket = resyncPacket;
      resyncAttempts++;
    }
    if (forceAccurate && state.currentRequestId === requestId) {
      if (state.decoder && state.decoder.state !== "closed") {
        try {
          await state.decoder.flush();
        } catch (e) {
          const msg = String(e instanceof Error ? e.message : e);
          if (!/aborted|reset/i.test(msg)) {
            console.warn("flush failed", e);
          }
        }
      }
      if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
        try {
          await state.alphaDecoder.flush();
        } catch (e) {
          const msg = String(e instanceof Error ? e.message : e);
          if (!/aborted|reset/i.test(msg)) {
            console.warn("flush failed", e);
          }
        }
      }
    }
    if (!state.seekDone && state.pendingSeekFrame) {
      state.seekTargetTimestamp = null;
      postFrame(id, state.pendingSeekFrame, requestId);
      state.pendingSeekFrame.close();
      state.pendingSeekFrame = null;
    }
    const totalMs = performance.now() - benchmarkStart;
    const decodeLoopMs = performance.now() - decodeLoopStart;
    try {
      postMessage({
        type: "debug",
        scope: "video-decoder-worker",
        event: "decodeBenchmark",
        assetId: id,
        requestId,
        payload: {
          fromCache: false,
          totalMs,
          keyPacketMs,
          decodeLoopMs,
          packetsDecoded,
          targetTimestamp: timestamp,
          resyncAttempts
        }
      });
    } catch {
    }
    postMessage({
      type: "seekDone",
      requestId,
      assetId: id
    });
    emitDiagnostics("seekDone", id, requestId);
  }
  async function handleIterate(startTime, endTime, requestId, assetId) {
    const id = assetId;
    if (!id) return;
    const state = assetStates.get(id);
    if (!state || !state.sink || !state.config) return;
    ensureDecoderInstance(state, id);
    if (!state.decoder) return;
    const benchmarkStart = performance.now();
    state.currentRequestId = requestId;
    resetAlphaMergeQueues(state);
    const keyPacketStart = performance.now();
    const initialPacket = await getVerifiedKeyPacket(state, startTime);
    const keyPacketMs = performance.now() - keyPacketStart;
    if (!initialPacket) {
      postMessage({
        type: "iterateDone",
        requestId,
        assetId: id
      });
      return;
    }
    state.seekTargetTimestamp = null;
    state.iterationInFlight = 0;
    state.iterationResume = null;
    const iterationHandler = (frame) => {
      const frameTime = frame.timestamp / 1e6;
      if (frameTime < startTime || frameTime > endTime + 0.05) {
        frame.close();
        return;
      }
      state.iterationInFlight++;
      const clone = frame.clone();
      const msg = {
        type: "frame",
        frame: clone,
        timestamp: frameTime,
        duration: (frame.duration ?? 0) / 1e6,
        requestId,
        assetId: id
      };
      postMessage(msg, [clone]);
      frame.close();
    };
    state.customOutputHandler = iterationHandler;
    try {
      let currentPacket = initialPacket;
      let resyncAttempts = 0;
      let packetsDecoded = 0;
      while (currentPacket && state.currentRequestId === requestId && resyncAttempts <= MAX_RESYNC_ATTEMPTS) {
        await flushDecoderIfNeeded(state.decoder);
        await flushDecoderIfNeeded(state.alphaDecoder);
        if (!resetAndConfigureDecoders(state, id)) break;
        const decodedKey = await decodePacketSafe(
          state,
          id,
          currentPacket,
          requestId
        );
        if (decodedKey) packetsDecoded++;
        if (!decodedKey) {
          currentPacket = await getNextKeyPacketSafe(state, currentPacket);
          resyncAttempts++;
          continue;
        }
        let resyncPacket = null;
        try {
          const packets = state.sink.packets(currentPacket);
          await packets.next();
          for await (const packet of packets) {
            if (state.currentRequestId !== requestId) break;
            if (packet.timestamp > endTime + 0.1) break;
            if (!state.decoder || state.decoder.state === "closed") {
              resyncPacket = await getNextKeyPacketSafe(state, packet);
              break;
            }
            while (state.iterationInFlight >= MAX_ITERATION_IN_FLIGHT) {
              if (state.currentRequestId !== requestId) break;
              await new Promise((r) => state.iterationResume = r);
            }
            const decoded = await decodePacketSafe(state, id, packet, requestId);
            if (decoded) packetsDecoded++;
            if (!decoded) {
              resyncPacket = await getNextKeyPacketSafe(state, packet);
              break;
            }
          }
        } catch (e) {
          console.warn("Iteration packet loop failed", e);
        }
        if (!resyncPacket || state.currentRequestId !== requestId) {
          break;
        }
        currentPacket = resyncPacket;
        resyncAttempts++;
      }
      const totalMs = performance.now() - benchmarkStart;
      try {
        postMessage({
          type: "debug",
          scope: "video-decoder-worker",
          event: "iterateBenchmark",
          assetId: id,
          requestId,
          payload: {
            totalMs,
            keyPacketMs,
            packetsDecoded,
            startTime,
            endTime,
            resyncAttempts
          }
        });
      } catch {
      }
      if (state.decoder && state.decoder.state !== "closed") {
        try {
          await state.decoder.flush();
        } catch {
        }
      }
      if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
        try {
          await state.alphaDecoder.flush();
        } catch {
        }
      }
    } finally {
      state.customOutputHandler = null;
    }
    postMessage({
      type: "iterateDone",
      requestId,
      assetId: id
    });
    emitDiagnostics("iterateDone", id, requestId);
  }
  function dispose(assetId) {
    if (assetId) {
      const state = assetStates.get(assetId);
      if (!state) return;
      state.currentRequestId++;
      if (state.decoder && state.decoder.state !== "closed") {
        state.decoder.close();
        state.decoder = null;
      }
      if (state.alphaDecoder && state.alphaDecoder.state !== "closed") {
        try {
          state.alphaDecoder.close();
        } catch {
        }
        state.alphaDecoder = null;
      }
      resetAlphaMergeQueues(state);
      for (const frame of state.cachedDecodedFrames.values()) {
        frame.close();
      }
      state.cachedDecodedFrames.clear();
      state.keyPacketCache.clear();
      state.input = null;
      state.sink = null;
      state.seekTargetTimestamp = null;
      state.pendingSeekFrame?.close();
      state.pendingSeekFrame = null;
      state.seekDone = false;
      state.customOutputHandler = null;
      state.iterationInFlight = 0;
      state.iterationResume = null;
      emitDiagnostics("dispose", assetId);
      return;
    }
    for (const [id] of assetStates) {
      dispose(id);
    }
    assetStates.clear();
    emitDiagnostics("dispose-all");
  }
})();

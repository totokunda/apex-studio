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

  // packages/renderer/src/lib/media/audio-decoder.worker.ts
  var import_mediabunny = __require("mediabunny");
  var nodeFs = __toESM(__require("node:fs/promises"), 1);
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
  var MAX_ITERATION_IN_FLIGHT = 8;
  var MAX_DECODE_QUEUE_SIZE = 50;
  try {
    postMessage({
      type: "debug",
      scope: "audio-decoder-worker",
      event: "worker-loaded"
    });
  } catch {
  }
  function getOrCreateState(assetId) {
    let state = assetStates.get(assetId);
    if (!state) {
      state = {
        decoder: null,
        sink: null,
        input: null,
        config: null,
        currentRequestId: 0,
        iterationInFlight: 0,
        iterationResume: null,
        cachedSeekTimestamp: null,
        cachedKeyPacket: null,
        hasSuccessfullyIterated: false
      };
      assetStates.set(assetId, state);
    }
    return state;
  }
  function createAudioDecoder(state, assetId, requestId, onOutput) {
    const decoder = new AudioDecoder({
      output: (data) => {
        try {
          const sample = new import_mediabunny.AudioSample(data);
          onOutput(sample);
        } catch (e) {
          console.error("AudioDecoder output error", e);
        }
      },
      error: (e) => {
        console.error("AudioDecoder error", e);
        postMessage({
          type: "error",
          error: e.message ?? "AudioDecoder error",
          assetId,
          requestId
        });
      }
    });
    state.decoder = decoder;
    return decoder;
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
  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      postMessage({
        type: "debug",
        scope: "audio-decoder-worker",
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
        case "iterate": {
          await handleIterate(
            msg.startTime,
            msg.endTime,
            msg.requestId,
            msg.assetId
          );
          break;
        }
        case "preseek": {
          await handlePreseek(msg.assetId, msg.timestamp, msg.requestId);
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
      console.error("Audio Worker Error:", err);
      postMessage({
        type: "error",
        error: err.message,
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
    state.cachedSeekTimestamp = null;
    state.cachedKeyPacket = null;
    state.hasSuccessfullyIterated = false;
    let formats = import_mediabunny.ALL_FORMATS;
    let input = null;
    const filePath = fileURLToPathInWorker(cfg.asset.path);
    if (!filePath) {
      throw new Error("Missing file path for audio source");
    }
    input = new import_mediabunny.Input({ formats, source: createNodeFileSource(filePath) });
    state.input = input;
    const audioTrack = await state.input.getPrimaryAudioTrack();
    if (!audioTrack) throw new Error("No audio track found in worker");
    state.sink = new import_mediabunny.EncodedPacketSink(audioTrack);
    state.config = cfg.audioDecoderConfig;
    postMessage({
      type: "ready",
      requestId,
      assetId: id
    });
  }
  async function handlePreseek(assetId, timestamp, requestId) {
    const state = assetStates.get(assetId);
    if (!state || !state.sink) {
      postMessage({
        type: "preseekDone",
        requestId,
        assetId
      });
      return;
    }
    try {
      const keyPacket = await state.sink.getKeyPacket(timestamp, { verifyKeyPackets: true });
      state.cachedSeekTimestamp = timestamp;
      state.cachedKeyPacket = keyPacket;
    } catch (e) {
      state.cachedSeekTimestamp = null;
      state.cachedKeyPacket = null;
    }
    postMessage({
      type: "preseekDone",
      requestId,
      assetId
    });
  }
  async function handleIterate(startTime, endTime, requestId, assetId) {
    const id = assetId;
    if (!id) return;
    const state = assetStates.get(id);
    if (!state || !state.sink || !state.config) return;
    state.currentRequestId = requestId;
    state.iterationInFlight = 0;
    state.iterationResume = null;
    if (state.decoder && state.decoder.state !== "closed") {
      try {
        state.decoder.close();
      } catch {
      }
    }
    const decoder = createAudioDecoder(state, id, requestId, (sample) => {
      const frameTime = sample.timestamp;
      if (frameTime < startTime - 0.1 || frameTime > endTime + 0.1) {
        sample.close();
        return;
      }
      state.iterationInFlight++;
      const numChannels = sample.numberOfChannels;
      const numFrames = sample.numberOfFrames;
      const channelData = [];
      for (let ch = 0; ch < numChannels; ch++) {
        const chData = new Float32Array(numFrames);
        sample.copyTo(chData, { planeIndex: ch, format: "f32-planar" });
        channelData.push(chData);
      }
      const msg = {
        type: "audioData",
        channelData,
        sampleRate: sample.sampleRate,
        timestamp: sample.timestamp,
        duration: sample.duration,
        requestId,
        assetId: id
      };
      const transferables = channelData.map((arr) => arr.buffer);
      postMessage(msg, transferables);
      sample.close();
    });
    decoder.configure(state.config);
    let iterationSucceeded = false;
    try {
      let keyPacket;
      if (state.hasSuccessfullyIterated && state.cachedKeyPacket && state.cachedSeekTimestamp !== null && Math.abs(state.cachedSeekTimestamp - startTime) < 0.5) {
        keyPacket = state.cachedKeyPacket;
        state.cachedKeyPacket = null;
        state.cachedSeekTimestamp = null;
      } else {
        keyPacket = await state.sink.getKeyPacket(startTime, { verifyKeyPackets: true });
      }
      const packets = state.sink.packets(keyPacket || void 0);
      for await (const packet of packets) {
        if (state.currentRequestId !== requestId) break;
        if (packet.timestamp > endTime + 1) break;
        if (decoder.decodeQueueSize >= MAX_DECODE_QUEUE_SIZE) {
          await new Promise((resolve) => {
            decoder.addEventListener("dequeue", () => resolve(), { once: true });
          });
        }
        while (state.iterationInFlight >= MAX_ITERATION_IN_FLIGHT) {
          if (state.currentRequestId !== requestId) break;
          await new Promise((r) => state.iterationResume = r);
        }
        if (state.currentRequestId !== requestId) break;
        decoder.decode(packet.toEncodedAudioChunk());
      }
      if (decoder.state !== "closed") {
        await decoder.flush();
      }
      iterationSucceeded = true;
    } catch (e) {
      console.warn("Audio iteration failed", e);
    } finally {
      if (iterationSucceeded) {
        state.hasSuccessfullyIterated = true;
      }
      if (decoder.state !== "closed") {
        decoder.close();
      }
    }
    postMessage({
      type: "iterateDone",
      requestId,
      assetId: id
    });
  }
  function dispose(assetId) {
    if (assetId) {
      const state = assetStates.get(assetId);
      if (!state) return;
      state.currentRequestId++;
      if (state.decoder && state.decoder.state !== "closed") {
        try {
          state.decoder.close();
        } catch {
        }
        state.decoder = null;
      }
      state.input = null;
      state.sink = null;
      state.iterationInFlight = 0;
      state.iterationResume = null;
      state.cachedSeekTimestamp = null;
      state.cachedKeyPacket = null;
      state.hasSuccessfullyIterated = false;
      return;
    }
    for (const [id] of assetStates) {
      dispose(id);
    }
    assetStates.clear();
  }
})();

import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { useControlsStore } from "@/lib/control";
import { useInputControlsStore } from "@/lib/inputControl";
import { MediaInfo, AudioClipProps } from "@/lib/types";
import { getMediaInfo, getMediaInfoCached } from "@/lib/media/utils";
import { getAudioIterator, preconfigureAudioWorker, preseekAudioWorker } from "@/lib/media/audio";
import { WrappedAudioBuffer } from "mediabunny";
import type { BaseClipApplicator } from "./apply/base";
import { useClipStore } from "@/lib/clip";

// Schedules audio playback for a clip in sync with the timeline. Renders nothing.
const AudioPreview: React.FC<
  AudioClipProps & {
    framesToPrefetch?: number;
    rectWidth?: number;
    rectHeight?: number;
    applicators?: BaseClipApplicator[];
    overlap?: boolean;
    overrideClip?: AudioClipProps;
    inputMode?: boolean;
    inputId?: string;
    disabled?: boolean;
    /**
     * If true, the clip is allowed to prebuffer/schedule audio but should be silent.
     * Used with playback overlap to avoid boundary gaps without double-audio.
     */
    muted?: boolean;
    /**
     * If true (and inputMode is true), preserves the clip's startFrame/endFrame
     * instead of normalizing to 0. Used for group children where timing within
     * the group must be respected.
     */
    preserveInputTiming?: boolean;
  }
> = (props) => {

  const {
    assetId,
    startFrame = 0,
    endFrame,
    trimStart,
    volume = 0,
    fadeIn = 0,
    fadeOut = 0,
    speed: _speed,
    framesToPrefetch = 5,
  } = props;

  const {
    inputMode = false,
    inputId,
    disabled = false,
    muted = false,
    preserveInputTiming = false,
  } = props as {
    inputMode?: boolean;
    inputId?: string;
    disabled?: boolean;
    muted?: boolean;
    preserveInputTiming?: boolean;
  };

  const mediaInfoRef = useRef<MediaInfo | null>(
    getMediaInfoCached(assetId) || null,
  );
  const fpsFromControls = useControlsStore((s) => s.fps);
  const fpsByInputId = useInputControlsStore((s) => s.fpsByInputId);
  const fpsFromInputs = fpsByInputId[inputId || ""] ?? fpsFromControls;
  const fps = inputMode ? fpsFromInputs : fpsFromControls;
  const focusFrameFromControls = useControlsStore((s) => s.focusFrame);
  const focusFrameByInputId = useInputControlsStore(
    (s) => s.focusFrameByInputId,
  );

  const focusFrameFromInputs = focusFrameByInputId[inputId || ""] ?? 0;
  const focusFrame = inputMode ? focusFrameFromInputs : focusFrameFromControls;
  const getAssetById = useClipStore((s) => s.getAssetById);

  // In input mode, the AudioPreview expects a clip-local focusFrame (0..span), so we normalize
  // absolute start/end into a 0-based window when needed.
  // However, if preserveInputTiming is true (e.g., for group children), we preserve the actual
  // startFrame/endFrame so that timing within the group is respected.
  const startFrameUsed = useMemo(() => {
    if (inputMode && !preserveInputTiming) return 0;
    return startFrame;
  }, [inputMode, startFrame, preserveInputTiming]);
  const endFrameUsed = useMemo(() => {
    if (!inputMode || preserveInputTiming) {
      return typeof endFrame === "number" ? endFrame : undefined;
    }
    if (typeof endFrame === "number" && typeof startFrame === "number") {
      return Math.max(0, endFrame - startFrame);
    }
    return typeof endFrame === "number" ? endFrame : undefined;
  }, [endFrame, inputMode, startFrame, preserveInputTiming]);

  const isInFrame = useMemo(() => {
    const f = Number(focusFrame);
    if (!Number.isFinite(f)) return true;
    const s = Number(startFrameUsed ?? 0);
    if (!Number.isFinite(s)) return true;
    const e =
      typeof endFrameUsed === "number" && Number.isFinite(endFrameUsed)
        ? endFrameUsed
        : Infinity;
    return f >= Number(s) - framesToPrefetch && f <= e;
  }, [focusFrame, startFrameUsed, endFrameUsed, framesToPrefetch]);

  // In input mode, focusFrame is clip-local and the input timeline is [0, span],
  // so we must not subtract the absolute clip start.
  //
  // NOTE: We always apply trimStart, even when the playhead is slightly before
  // the clip start (overlap prebuffer). This ensures split segments (where the
  // right-hand clip has a non-zero trimStart) decode/schedule from the correct
  // offset instead of restarting from the beginning.
  const currentFrame = useMemo(() => {
    return focusFrame - startFrameUsed + (trimStart || 0);
  }, [focusFrame, startFrameUsed, trimStart]);
  const isPlayingFromControls = useControlsStore((s) => s.isPlaying);
  const isPlayingByInputId = useInputControlsStore((s) => s.isPlayingByInputId);
  const isPlayingFromInputs = !!isPlayingByInputId[inputId || ""];
  const isPlaying = inputMode ? isPlayingFromInputs : isPlayingFromControls;
  const prevIsPlayingRef = useRef<boolean>(isPlaying);
  const startTimeRef = useRef(0);
  const iteratorRef = useRef<AsyncIterable<WrappedAudioBuffer | null> | null>(
    null,
  );
  const currentStartFrameRef = useRef<number>(0);
  const playbackTimeAtStartRef = useRef<number>(0);
  const audioQueueRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const lastResyncTimeRef = useRef<number>(0);
  const soundtouchNodeRef = useRef<AudioWorkletNode | null>(null);
  const soundtouchInitPromiseRef = useRef<Promise<boolean> | null>(null);
  const soundtouchUnavailableRef = useRef<boolean>(false);
  const soundtouchInitCtxRef = useRef<AudioContext | null>(null);
  const lastConfiguredSpeedRef = useRef<number | null>(null);
  const asset = useMemo(() => getAssetById(assetId), [assetId]);
  useEffect(() => {
    const wasPlaying = prevIsPlayingRef.current;
    if (!wasPlaying && isPlaying && isInFrame) {
      try {
        const detail = {
          assetId,
          currentFrame,
          fps,
          timeSec: fps ? currentFrame / fps : 0,
          inputMode: !!inputMode,
          inputId: inputMode ? String(inputId || "") : null,
        } as any;
        window.dispatchEvent(
          new CustomEvent("apex:playback:playing", { detail }),
        );
      } catch {}
    }
    if (wasPlaying && !isPlaying && isInFrame) {
      try {
        const detail = {
          assetId,
          currentFrame,
          fps,
          timeSec: fps ? currentFrame / fps : 0,
          inputMode: !!inputMode,
          inputId: inputMode ? String(inputId || "") : null,
        } as any;
        window.dispatchEvent(
          new CustomEvent("apex:playback:paused", { detail }),
        );
      } catch {}
    }
    prevIsPlayingRef.current = isPlaying;
  }, [isPlaying, assetId, currentFrame, fps, inputMode, inputId, isInFrame]);

  const { ctx, gainNode } = useMemo<{
    ctx: AudioContext;
    gainNode: GainNode;
  }>(() => {
    const AudioContext: any =
      (window as any).AudioContext || (window as any).webkitAudioContext;
    // Use low latency hint for minimal audio delay
    const ctx = new AudioContext({ latencyHint: "interactive" });
    const gainNode = ctx.createGain();
    gainNode.connect(ctx.destination);
    return { ctx, gainNode };
  }, []);

  // Keep AudioContext warm - resume on any user interaction to avoid cold-start latency
  useEffect(() => {
    const warmUp = () => {
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }
    };
    // Resume on common user interactions before they press play
    window.addEventListener("click", warmUp, { passive: true });
    window.addEventListener("keydown", warmUp, { passive: true });
    window.addEventListener("pointerdown", warmUp, { passive: true });
    // Also try to resume immediately
    warmUp();
    return () => {
      window.removeEventListener("click", warmUp);
      window.removeEventListener("keydown", warmUp);
      window.removeEventListener("pointerdown", warmUp);
    };
  }, [ctx]);

  // Mute/unmute at the output gain so we can prebuffer without audible playback.
  useEffect(() => {
    if (!ctx || !gainNode) return;
    try {
      const now = ctx.currentTime;
      const target = muted ? 0 : 1;
      // tiny ramp to avoid clicks
      gainNode.gain.cancelScheduledValues(now);
      gainNode.gain.setValueAtTime(gainNode.gain.value, now);
      gainNode.gain.linearRampToValueAtTime(target, now + 0.01);
    } catch {}
  }, [muted, ctx, gainNode]);

  // Prepare analyser node and expose it for visualizers when in input mode
  const analyserRef = useRef<AnalyserNode | null>(null);
  const announceAnalyser = useCallback(() => {
    try {
      const key = inputMode && inputId ? String(inputId) : null;
      if (!key) return;
      const store: any = window as any;
      store.__apexAudioAnalysers =
        store.__apexAudioAnalysers ||
        new Map<string, { ctx: AudioContext; analyser: AnalyserNode }>();
      if (analyserRef.current) {
        store.__apexAudioAnalysers.set(key, {
          ctx,
          analyser: analyserRef.current,
        });
        window.dispatchEvent(
          new CustomEvent("apex:audio:analyser-ready", {
            detail: { inputId: key },
          }),
        );
      }
    } catch {}
  }, [ctx, inputMode, inputId]);

  // Create the analyser node and announce it (original behavior)
  useEffect(() => {
    if (!ctx || !gainNode) return;
    if (analyserRef.current) {
      announceAnalyser();
      return;
    }
    try {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512; // higher resolution for smoother ring
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;
      analyser.smoothingTimeConstant = 0.85;
      try {
        gainNode.disconnect();
      } catch {}
      gainNode.connect(analyser);
      analyser.connect(ctx.destination);
      analyserRef.current = analyser;
      announceAnalyser();
    } catch {}
  }, [ctx, gainNode, announceAnalyser]);

  // Re-announce analyser when this clip becomes active (in-frame and playing)
  // This ensures the visualizer connects to the currently playing clip's analyser
  useEffect(() => {
    if (!analyserRef.current) return;
    if (isInFrame && isPlaying) {
      announceAnalyser();
    }
  }, [isInFrame, isPlaying, announceAnalyser]);

  // Cleanup global analyser map entry on unmount
  useEffect(() => {
    return () => {
      try {
        const key = inputMode && inputId ? String(inputId) : null;
        if (key) {
          const store: any = window as any;
          if (
            store.__apexAudioAnalysers &&
            typeof store.__apexAudioAnalysers.delete === "function"
          ) {
            store.__apexAudioAnalysers.delete(key);
          }
        }
      } catch {}
    };
  }, [inputMode, inputId]);

  // Convert dB to linear gain (0 dB = 1.0, -60 dB ≈ 0.001, +20 dB = 10.0)
  const dbToGain = (db: number) => Math.pow(10, db / 20);
  const speed = useMemo(() => {
    const s = Number(_speed ?? 1);
    return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
  }, [_speed]);
  // playback time is derived from the audio clock on demand to avoid rerenders
  const getPlaybackTime = () => {
    if (isPlaying) {
      return (
        ctx.currentTime - startTimeRef.current + playbackTimeAtStartRef.current
      );
    }
    return playbackTimeAtStartRef.current;
  };

  // Load media info to detect audio track and preconfigure worker
  useEffect(() => {
    if (!asset) return;
    const info = getMediaInfoCached(asset.path);
    if (info) {
      mediaInfoRef.current = info;
      // Preconfigure audio worker early so it's ready when playback starts
      if (info.audio) {
        void preconfigureAudioWorker(asset.path, info);
      }
    }
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo(asset?.path ?? "");
        if (!cancelled) {
          mediaInfoRef.current = info;
          // Preconfigure audio worker early so it's ready when playback starts
          if (info?.audio) {
            void preconfigureAudioWorker(asset.path, info);
          }
        }
      } catch {}
    })();
  }, [asset]);

  // Pre-seek audio worker when scrubbing while paused for instant playback start
  useEffect(() => {
    if (isPlaying || !asset || !fps || !isInFrame) return;
    if (!mediaInfoRef.current?.audio) return;
    
    // Calculate the media timestamp we would seek to
    const speedFactor = Math.max(0.1, speed);
    const mediaStartOffset = mediaInfoRef.current?.startFrame || 0;
    const mediaFrameIndex = Math.max(0, Math.floor(currentFrame * speedFactor)) + mediaStartOffset;
    const timestamp = mediaFrameIndex / fps;
    
    // Fire off preseek (non-blocking)
    void preseekAudioWorker(asset.path, timestamp, mediaInfoRef.current);
  }, [isPlaying, asset, fps, currentFrame, speed, isInFrame]);

  useEffect(() => {
    const onPlaying = async (
      e: CustomEvent<{
        assetId: string;
        currentFrame: number;
        fps: number;
        timeSec: number;
        inputMode?: boolean;
        inputId?: string | null;
      }>,
    ) => {
      const d: any = e.detail || {};

      currentStartFrameRef.current = d.currentFrame;
      playbackTimeAtStartRef.current = d.timeSec;
      if (ctx.state === "suspended") {
        try {
          await ctx.resume();
        } catch {}
      }
    };
    const handler = (e: Event) => onPlaying(e as any);
    window.addEventListener("apex:playback:playing", handler);
    return () => window.removeEventListener("apex:playback:playing", handler);
  }, [ctx, inputMode, inputId]);

  useEffect(() => {
    const onPaused = (
      e: CustomEvent<{
        assetId: string;
        currentFrame: number;
        fps: number;
        timeSec: number;
        inputMode?: boolean;
        inputId?: string | null;
      }>,
    ) => {
      const d: any = e.detail || {};
      currentStartFrameRef.current = 0;
      // @ts-ignore
      iteratorRef.current?.return?.();
      iteratorRef.current = null;
      // lock in playback time at pause
      playbackTimeAtStartRef.current =
        d && typeof d.timeSec === "number" ? d.timeSec : getPlaybackTime();
      for (const node of audioQueueRef.current) {
        try {
          node.stop();
        } catch {}
      }
      audioQueueRef.current.clear();
    };
    const handler = (e: Event) => onPaused(e as any);
    window.addEventListener("apex:playback:paused", handler);
    return () => window.removeEventListener("apex:playback:paused", handler);
  }, [inputMode, inputId]);

  const ensureSoundtouchNode = useCallback(async () => {
    if (!ctx?.audioWorklet || soundtouchUnavailableRef.current) {
      return null;
    }
    // If existing node was created on a different context, drop it
    if (
      soundtouchNodeRef.current &&
      (soundtouchNodeRef.current as any).context !== ctx
    ) {
      try {
        soundtouchNodeRef.current.disconnect();
      } catch {}
      soundtouchNodeRef.current = null;
    }
    // Reset init promise if it was for another context
    if (
      soundtouchInitPromiseRef.current &&
      soundtouchInitCtxRef.current !== ctx
    ) {
      soundtouchInitPromiseRef.current = null;
    }
    if (!soundtouchInitPromiseRef.current) {
      const candidates = (() => {
        if (typeof window === "undefined") {
          return ["soundtouch-worklet.js"];
        }
        const baseHref = window.location?.href || "";
        const origin = window.location?.origin || "";
        const urls = new Set<string>();
        if (baseHref) {
          try {
            urls.add(new URL("soundtouch-worklet.js", baseHref).toString());
          } catch {}
        }
        if (origin) {
          try {
            urls.add(new URL("soundtouch-worklet.js", origin).toString());
          } catch {}
        }
        urls.add("/soundtouch-worklet.js");
        urls.add("soundtouch-worklet.js");
        return Array.from(urls);
      })();
      soundtouchInitPromiseRef.current = (async () => {
        for (const candidate of candidates) {
          try {
            await ctx.audioWorklet.addModule(candidate);
            return true;
          } catch (err) {
            console.warn("[audio] Failed to load soundtouch worklet", {
              candidate,
              err,
            });
          }
        }
        return false;
      })();
      soundtouchInitCtxRef.current = ctx;
    }
    const ready = await soundtouchInitPromiseRef.current;
    if (!ready) {
      soundtouchUnavailableRef.current = true;
      return null;
    }
    if (!soundtouchNodeRef.current) {
      try {
        const node = new AudioWorkletNode(ctx, "soundtouch-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 2,
          channelCountMode: "explicit",
          channelInterpretation: "speakers",
          outputChannelCount: [2],
        });
        node.connect(gainNode);
        soundtouchNodeRef.current = node;
      } catch (err) {
        console.warn("[audio] Failed to create soundtouch node", err);
        soundtouchUnavailableRef.current = true;
        soundtouchNodeRef.current = null;
        return null;
      }
    }
    return soundtouchNodeRef.current;
  }, [ctx, gainNode]);

  const configureSoundtouchForSpeed = useCallback(
    (playbackSpeed: number) => {
      if (!ctx) return;
      const node = soundtouchNodeRef.current;
      if (!node) return;
      const safeSpeed = Math.min(
        5,
        Math.max(
          0.1,
          Number.isFinite(playbackSpeed) && playbackSpeed > 0
            ? playbackSpeed
            : 1,
        ),
      );
      if (lastConfiguredSpeedRef.current === safeSpeed) return;
      lastConfiguredSpeedRef.current = safeSpeed;

      const desiredPitchRatio =
        Number.isFinite(safeSpeed) && safeSpeed > 0 ? 1 / safeSpeed : 1;
      const tempoParam = node.parameters.get("tempo");
      const rateParam = node.parameters.get("rate");
      const pitchParam = node.parameters.get("pitch");
      const pitchSemitoneParam = node.parameters.get("pitchSemitones");

      const minPitchLog = Math.log2(0.25);
      const maxPitchLog = Math.log2(4);
      const semitoneMin = -24;
      const semitoneMax = 24;

      const ratioLog = Math.log2(Math.max(1e-6, desiredPitchRatio));
      let semitoneValue = 0;
      if (ratioLog < minPitchLog) {
        semitoneValue = Math.max(
          semitoneMin,
          -Math.ceil((minPitchLog - ratioLog) * 12),
        );
      } else if (ratioLog > maxPitchLog) {
        semitoneValue = Math.min(
          semitoneMax,
          Math.ceil((ratioLog - maxPitchLog) * 12),
        );
      }
      let adjustedLog = ratioLog - semitoneValue / 12;
      adjustedLog = Math.min(maxPitchLog, Math.max(minPitchLog, adjustedLog));
      const basePitch = Math.pow(2, adjustedLog);

      const now = ctx.currentTime;
      tempoParam?.setValueAtTime(1, now);
      rateParam?.setValueAtTime(1, now);
      pitchParam?.setValueAtTime(basePitch, now);
      pitchSemitoneParam?.setValueAtTime(semitoneValue, now);
    },
    [ctx],
  );

  const startRendering = useCallback(async () => {
    // Only play audio when clip is actually active (not during prebuffer period)
    if (
      !isInFrame ||
      !isPlaying ||
      !ctx ||
      !mediaInfoRef.current ||
      !mediaInfoRef.current.audio ||
      !fps ||
      !Number.isFinite(currentFrame)
    ) {
      return;
    }
    
    // CRITICAL: Capture the intended start time IMMEDIATELY before any async operations.
    // This ensures audio sync even if ctx.resume() or getAudioIterator() takes time.
    const intendedWallTime = performance.now();
    
    // align starting frame and playback anchor with current UI state
    if (typeof currentFrame === "number" && fps) {
      currentStartFrameRef.current = currentFrame;
      playbackTimeAtStartRef.current = currentFrame / fps;
    }
    
    // Stop any existing buffers and iterator
    for (const node of audioQueueRef.current) {
      try {
        node.stop();
      } catch {}
    }
    audioQueueRef.current.clear();
    // @ts-ignore
    iteratorRef.current?.return?.();
    
    // Fire off context resume without blocking - it will be ready by the time we schedule
    const resumePromise = ctx.state === "suspended" ? ctx.resume() : Promise.resolve();
    
    // Sample from the correct media frame based on speed
    const speedFactor = Math.max(0.1, speed);
    const mediaStartOffset = mediaInfoRef.current?.startFrame || 0;
    const mediaFrameIndex =
      Math.max(0, Math.floor(currentStartFrameRef.current * speedFactor)) +
      mediaStartOffset;
    // IMPORTANT: mediaTimeAtStart uses the *signed* currentStartFrameRef to support
    // overlap prebuffering. When currentFrame is negative (playhead before clip start),
    // this pushes scheduled buffers into the future so audio starts exactly at the
    // clip boundary (no gap), without playing early.
    const mediaTimeAtStart =
      (currentStartFrameRef.current * speedFactor + mediaStartOffset) / fps;
    // Extend audio beyond clip boundary for seamless transitions with adjacent clips
    const endIndex = mediaInfoRef.current?.endFrame
      ? mediaInfoRef.current.endFrame
      : undefined;
    const asset = getAssetById(assetId);
    if (!asset) return;
    
    // Start iterator and soundtouch setup in parallel
    const [iteratorResult, soundtouchNode] = await Promise.all([
      getAudioIterator(asset.path, {
        mediaInfo: mediaInfoRef.current || undefined,
        fps,
        startIndex: mediaFrameIndex,
        endIndex,
      }),
      ensureSoundtouchNode(),
      resumePromise, // Also wait for resume to complete
    ]);
    
    iteratorRef.current = iteratorResult;
    
    if (soundtouchNode) {
      configureSoundtouchForSpeed(speed);
    }

    // Now that context is definitely running, calculate the actual start time.
    // Account for any delay that occurred during async setup by adjusting the anchor.
    const setupDelayMs = performance.now() - intendedWallTime;
    const setupDelaySec = setupDelayMs / 1000;
    
    // Set startTimeRef to ctx.currentTime, but compensate for setup delay
    // by pretending playback started (setupDelaySec) ago.
    startTimeRef.current = ctx.currentTime - setupDelaySec;
    lastResyncTimeRef.current = Date.now();

    // Calculate audio output latency once for this playback session
    // This compensates for the delay between scheduling and actual audio output
    const outputLatency = (ctx.outputLatency || 0) + (ctx.baseLatency || 0);

    for await (const buf of iteratorRef.current) {
      if (!buf) continue; // Skip null buffers but keep trying


      const buffer = buf.buffer || null;
      const duration = buf.duration || 0;
      const timestamp = buf.timestamp || 0;

      if (!buffer || duration <= 0) {
        continue; // Skip invalid buffers but keep trying
      }

      // Map media timestamp to wall clock according to speed.
      // Timeline seconds advance 1:1 with wall clock; media advances at `speed`.
      const speedVal = Math.max(0.1, speed);
      // Subtract outputLatency to compensate for audio system delay
      // This ensures audio reaches the speakers at the same time as video reaches the screen
      let startTimestamp =
        startTimeRef.current + (timestamp - mediaTimeAtStart) / speedVal - outputLatency;

      let bufferDurationToPlay = duration;

      // Clamp duration to clip boundary to prevent overlap/phasing with next clip
      if (
        typeof endFrameUsed === "number" &&
        Number.isFinite(endFrameUsed) &&
        fps
      ) {
        const wallSecondsRemaining =
          (endFrameUsed -
            startFrameUsed -
            (currentStartFrameRef.current - (trimStart || 0))) /
          fps;
        const absoluteEndTime = startTimeRef.current + wallSecondsRemaining;

        const wallDuration = duration / speedVal;

        if (startTimestamp >= absoluteEndTime) {
          continue;
        }

        if (startTimestamp + wallDuration > absoluteEndTime) {
          const allowedWallDuration = Math.max(
            0,
            absoluteEndTime - startTimestamp,
          );
          bufferDurationToPlay = allowedWallDuration * speedVal;
        }
      }

      // Ensure no gap at clip boundaries - if this buffer should start very soon, start it immediately
      const timeUntilStart = startTimestamp - ctx.currentTime;
      if (timeUntilStart > 0 && timeUntilStart < 0.001) {
        startTimestamp = ctx.currentTime;
      }

      const node = ctx.createBufferSource();
      node.buffer = buffer;
      const playbackRate = speedVal;
      node.playbackRate.setValueAtTime(playbackRate, ctx.currentTime);
      if (!soundtouchNode) {
        try {
          (node as any).preservesPitch = true;
        } catch {}
      }

      const nodeGain = ctx.createGain();
      node.connect(nodeGain);
      if (soundtouchNode && (soundtouchNode as any).context === ctx) {
        try {
          nodeGain.connect(soundtouchNode);
        } catch {
          try {
            nodeGain.connect(gainNode);
          } catch {}
        }
      } else {
        try {
          nodeGain.connect(gainNode);
        } catch {}
      }

      const baseGain = dbToGain(volume || 0);
      const fadeInDuration = fadeIn || 0;
      const fadeOutDuration = fadeOut || 0;
      const totalDuration = mediaInfoRef.current?.duration || duration;

      // Set up fade in
      if (fadeInDuration > 0 && timestamp < fadeInDuration) {
        const fadeProgress = Math.min(1, timestamp / fadeInDuration);
        nodeGain.gain.setValueAtTime(
          baseGain * fadeProgress,
          startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime,
        );
        if (timestamp + duration <= fadeInDuration) {
          const endFadeProgress = Math.min(
            1,
            (timestamp + duration) / fadeInDuration,
          );
          nodeGain.gain.linearRampToValueAtTime(
            baseGain * endFadeProgress,
            (startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime) + duration,
          );
        } else {
          nodeGain.gain.linearRampToValueAtTime(
            baseGain,
            (startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime) +
              (fadeInDuration - timestamp),
          );
        }
      } else if (
        fadeOutDuration > 0 &&
        timestamp + duration > totalDuration - fadeOutDuration
      ) {
        // Set up fade out
        const fadeStartTime = totalDuration - fadeOutDuration;
        if (timestamp >= fadeStartTime) {
          const fadeProgress =
            1 - (timestamp - fadeStartTime) / fadeOutDuration;
          nodeGain.gain.setValueAtTime(
            baseGain * fadeProgress,
            startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime,
          );
          const endFadeProgress = Math.max(
            0,
            1 - (timestamp + duration - fadeStartTime) / fadeOutDuration,
          );
          nodeGain.gain.linearRampToValueAtTime(
            baseGain * endFadeProgress,
            (startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime) + duration,
          );
        } else {
          nodeGain.gain.setValueAtTime(
            baseGain,
            startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime,
          );
          const bufferOverlap = timestamp + duration - fadeStartTime;
          nodeGain.gain.linearRampToValueAtTime(
            baseGain,
            (startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime) +
              (duration - bufferOverlap),
          );
          nodeGain.gain.linearRampToValueAtTime(
            0,
            (startTimestamp >= ctx.currentTime
              ? startTimestamp
              : ctx.currentTime) + duration,
          );
        }
      } else {
        // No fade - just apply constant volume
        nodeGain.gain.setValueAtTime(
          baseGain,
          startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime,
        );
      }

      if (startTimestamp >= ctx.currentTime) {
        node.start(startTimestamp, 0, bufferDurationToPlay);
      } else {
        const wallOffset = ctx.currentTime - startTimestamp;
        const bufferOffset = wallOffset * speedVal;
        if (bufferDurationToPlay > bufferOffset) {
          node.start(
            ctx.currentTime,
            bufferOffset,
            bufferDurationToPlay - bufferOffset,
          );
        } else {
          continue;
        }
      }

      audioQueueRef.current.add(node);
      node.onended = () => {
        audioQueueRef.current.delete(node);
        try {
          node.disconnect();
        } catch {}
        try {
          nodeGain.disconnect();
        } catch {}
      };
      // More aggressive buffering for seamless playback - queue up to 2 seconds ahead
      if (
        (timestamp - mediaTimeAtStart) / Math.max(0.1, speed) -
          (ctx.currentTime - startTimeRef.current) >=
        2
      ) {
        await new Promise<void>((resolve) => {
          const id = setInterval(() => {
            const ahead =
              (timestamp - mediaTimeAtStart) / Math.max(0.1, speed) -
              (ctx.currentTime - startTimeRef.current);
            if (ahead < 1.5) {
              clearInterval(id);
              resolve();
            }
          }, 50);
        });
      }
    }
  }, [
    isInFrame,
    isPlaying,
    ctx,
    mediaInfoRef.current,
    fps,
    assetId,
    gainNode,
    volume,
    fadeIn,
    fadeOut,
    speed,
    ensureSoundtouchNode,
    configureSoundtouchForSpeed,
  ]);

  // Start or restart rendering when playback starts or media becomes ready
  useEffect(() => {
    if (!isPlaying || disabled || !isInFrame) return;
    void startRendering();
  }, [
    isPlaying,
    ctx,
    mediaInfoRef.current,
    assetId,
    fps,
    startRendering,
    volume,
    fadeIn,
    fadeOut,
    disabled,
    isInFrame,
  ]);

  // If the playhead leaves this clip while playing, stop scheduling and stop any queued nodes.
  // This keeps audio accurate during scrubbing / jumping frames.
  useEffect(() => {
    if (isInFrame) return;
    // @ts-ignore
    iteratorRef.current?.return?.();
    iteratorRef.current = null;
    for (const node of audioQueueRef.current) {
      try {
        node.stop();
      } catch {}
    }
    audioQueueRef.current.clear();
  }, [isInFrame]);

  // Cleanup on unmount: let scheduled audio complete for smooth transitions
  useEffect(() => {

    return () => {
      // Don't stop audio nodes - let them finish their scheduled playback naturally
      // This prevents gaps when transitioning between adjacent clips
      // The nodes will clean themselves up via their onended handlers
      audioQueueRef.current.clear();

      // Stop the audio iterator to prevent scheduling new buffers
      // @ts-ignore
      iteratorRef.current?.return?.();
      iteratorRef.current = null;
      try {
        soundtouchNodeRef.current?.disconnect();
      } catch {}
      soundtouchNodeRef.current = null;
    };
  }, []);

  return null;
};

export default AudioPreview;

import React, {  useCallback, useEffect, useMemo, useRef } from 'react';
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, MediaInfo, AudioClipProps } from '@/lib/types';
import { getMediaInfo, getMediaInfoCached } from '@/lib/media/utils';
import { getAudioIterator } from '@/lib/media/audio';
import { WrappedAudioBuffer } from 'mediabunny';

type ClipWithSrc = Extract<AnyClipProps, { src: string }>;

// Schedules audio playback for a clip in sync with the timeline. Renders nothing.
const AudioPreview: React.FC<ClipWithSrc & {framesToPrefetch?: number}> = (props) => {
  const { src, startFrame = 0, framesToGiveStart, volume = 0, fadeIn = 0, fadeOut = 0, speed: _speed } = props as AudioClipProps;
  const mediaInfoRef = useRef<MediaInfo | null>(getMediaInfoCached(src) || null);
  const fps = useControlsStore((s) => s.fps);
  const focusFrame = useControlsStore((s) => s.focusFrame);
  const currentFrame = useMemo(() => focusFrame - startFrame + (framesToGiveStart || 0), [focusFrame, startFrame, framesToGiveStart]);
  const isPlaying = useControlsStore((s) => s.isPlaying);
  const prevIsPlayingRef = useRef<boolean>(isPlaying);
  const startTimeRef = useRef(0);
  const iteratorRef = useRef<AsyncIterable<WrappedAudioBuffer | null> | null>(null);
  const currentStartFrameRef = useRef<number>(0);
  const playbackTimeAtStartRef = useRef<number>(0);
  const audioQueueRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const lastResyncTimeRef = useRef<number>(0);
  const hasValidCurrentFrame = useMemo(() => currentFrame >= 0, [currentFrame]);
  

  useEffect(() => {
    const wasPlaying = prevIsPlayingRef.current;
    if (!wasPlaying && isPlaying) {
      try {
        const detail = { src, currentFrame, fps, timeSec: fps ? currentFrame / fps : 0 };
        window.dispatchEvent(new CustomEvent('apex:playback:playing', { detail }));
      } catch {}
    }
    if (wasPlaying && !isPlaying) {
      try {
        const detail = { src, currentFrame, fps, timeSec: fps ? currentFrame / fps : 0 };
        window.dispatchEvent(new CustomEvent('apex:playback:paused', { detail }));
      } catch {}
    }
    prevIsPlayingRef.current = isPlaying;
  }, [isPlaying, src, currentFrame, fps]);
  

  const { ctx, gainNode } = useMemo<{ ctx: AudioContext; gainNode: GainNode }>(() => {
    const AudioContext: any = (window as any).AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContext();
    const gainNode = ctx.createGain();
    gainNode.connect(ctx.destination);
    return { ctx, gainNode };
  }, []);

  // Convert dB to linear gain (0 dB = 1.0, -60 dB ≈ 0.001, +20 dB = 10.0)
  const dbToGain = (db: number) => Math.pow(10, db / 20);
  const speed = useMemo(() => {
    const s = Number(_speed ?? 1);
    return Number.isFinite(s) && s > 0 ? Math.min(5, Math.max(0.1, s)) : 1;
  }, [_speed]);
  // playback time is derived from the audio clock on demand to avoid rerenders
  const getPlaybackTime = () => {
    if (isPlaying) {
      return ctx.currentTime - startTimeRef.current + playbackTimeAtStartRef.current;
    }
    return playbackTimeAtStartRef.current;
  };

  // Load media info to detect audio track
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const info = await getMediaInfo(src);
        if (!cancelled) mediaInfoRef.current = info;
      } catch {}
    })();
    return () => { cancelled = true };
  }, [src]);

  useEffect(() => {
    const onPlaying = async (e: CustomEvent<{ src: string; currentFrame: number; fps: number; timeSec: number }>) => {
        currentStartFrameRef.current = e.detail.currentFrame;
        playbackTimeAtStartRef.current = e.detail.timeSec;
        if (ctx.state === 'suspended') {
          try { await ctx.resume(); } catch {}
        }
    };
    const handler = (e: Event) => onPlaying(e as any);
    window.addEventListener('apex:playback:playing', handler);
    return () => window.removeEventListener('apex:playback:playing', handler);
  }, [ctx]);

  useEffect(() => {
    const onPaused = (e: CustomEvent<{ src: string; currentFrame: number; fps: number; timeSec: number }>) => {
        currentStartFrameRef.current = 0;
        // @ts-ignore
        iteratorRef.current?.return?.();
        iteratorRef.current = null;
        // lock in playback time at pause
        playbackTimeAtStartRef.current = e.detail.timeSec ?? getPlaybackTime();
        for (const node of audioQueueRef.current) {
          try { node.stop(); } catch {}
        }
        audioQueueRef.current.clear();
    };
    const handler = (e: Event) => onPaused(e as any);
    window.addEventListener('apex:playback:paused', handler);
    return () => window.removeEventListener('apex:playback:paused', handler);
  }, []);


  const startRendering = useCallback(async () => {
    // Only play audio when clip is actually active (not during prebuffer period)
    if (!isPlaying || !ctx || !mediaInfoRef.current || !mediaInfoRef.current.audio || !fps || !hasValidCurrentFrame) {
      return;
    }
    // align starting frame and playback anchor with current UI state
    if (typeof currentFrame === 'number' && fps) {
      currentStartFrameRef.current = currentFrame
      playbackTimeAtStartRef.current = currentFrame / fps;
    }
    // Stop any existing buffers and iterator
    for (const node of audioQueueRef.current) {
      try { node.stop(); } catch {}
    }
    audioQueueRef.current.clear();
    // @ts-ignore
    iteratorRef.current?.return?.();

    startTimeRef.current = ctx.currentTime;
    lastResyncTimeRef.current = Date.now();
    
    // Sample from the correct media frame based on speed
    const mediaFrameIndex = Math.max(0, Math.floor(currentStartFrameRef.current * Math.max(0.1, speed))) + (mediaInfoRef.current?.startFrame || 0);
    const mediaTimeAtStart = mediaFrameIndex / fps;
    // Extend audio beyond clip boundary for seamless transitions with adjacent clips
    const endIndex = mediaInfoRef.current?.endFrame ? mediaInfoRef.current.endFrame : undefined;

    iteratorRef.current = await getAudioIterator(src, { mediaInfo:mediaInfoRef.current || undefined, fps, startIndex: mediaFrameIndex, endIndex });

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
      let startTimestamp = startTimeRef.current + (timestamp - mediaTimeAtStart) / Math.max(0.1, speed);
      
      // Ensure no gap at clip boundaries - if this buffer should start very soon, start it immediately
      const timeUntilStart = startTimestamp - ctx.currentTime;
      if (timeUntilStart > 0 && timeUntilStart < 0.001) {
        startTimestamp = ctx.currentTime;
      }
      
      const node = ctx.createBufferSource();
		  node.buffer = buffer;
      try { node.playbackRate.value = Math.max(0.1, speed);
        (node as any).preservesPitch = true;
       } catch {}
		  
		  // Apply volume and fade effects
		  const nodeGain = ctx.createGain();
		  node.connect(nodeGain);
		  nodeGain.connect(gainNode);
		  
		  const baseGain = dbToGain(volume || 0);
		  const fadeInDuration = fadeIn || 0;
		  const fadeOutDuration = fadeOut || 0;
		  const totalDuration = mediaInfoRef.current?.duration || duration;
		  
		  // Set up fade in
		  if (fadeInDuration > 0 && timestamp < fadeInDuration) {
		    const fadeProgress = Math.min(1, timestamp / fadeInDuration);
		    nodeGain.gain.setValueAtTime(baseGain * fadeProgress, startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime);
		    if (timestamp + duration <= fadeInDuration) {
		      const endFadeProgress = Math.min(1, (timestamp + duration) / fadeInDuration);
		      nodeGain.gain.linearRampToValueAtTime(baseGain * endFadeProgress, (startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime) + duration);
		    } else {
		      nodeGain.gain.linearRampToValueAtTime(baseGain, (startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime) + (fadeInDuration - timestamp));
		    }
		  } else if (fadeOutDuration > 0 && timestamp + duration > totalDuration - fadeOutDuration) {
		    // Set up fade out
		    const fadeStartTime = totalDuration - fadeOutDuration;
		    if (timestamp >= fadeStartTime) {
		      const fadeProgress = 1 - ((timestamp - fadeStartTime) / fadeOutDuration);
		      nodeGain.gain.setValueAtTime(baseGain * fadeProgress, startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime);
		      const endFadeProgress = Math.max(0, 1 - ((timestamp + duration - fadeStartTime) / fadeOutDuration));
		      nodeGain.gain.linearRampToValueAtTime(baseGain * endFadeProgress, (startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime) + duration);
		    } else {
		      nodeGain.gain.setValueAtTime(baseGain, startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime);
		      const bufferOverlap = (timestamp + duration) - fadeStartTime;
		      nodeGain.gain.linearRampToValueAtTime(baseGain, (startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime) + (duration - bufferOverlap));
		      nodeGain.gain.linearRampToValueAtTime(0, (startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime) + duration);
		    }
		  } else {
		    // No fade - just apply constant volume
		    nodeGain.gain.setValueAtTime(baseGain, startTimestamp >= ctx.currentTime ? startTimestamp : ctx.currentTime);
		  }

      if (startTimestamp >= ctx.currentTime) {
        node.start(startTimestamp);
      } else {
        node.start(ctx.currentTime, ctx.currentTime - startTimestamp);
      }

      audioQueueRef.current.add(node);
      node.onended = () => {
        audioQueueRef.current.delete(node);
      };

      // More aggressive buffering for seamless playback - queue up to 2 seconds ahead
      if ((timestamp - mediaTimeAtStart) / Math.max(0.1, speed) - (ctx.currentTime - startTimeRef.current) >= 2) {
        await new Promise<void>((resolve) => {
          const id = setInterval(() => {
            const ahead = (timestamp - mediaTimeAtStart) / Math.max(0.1, speed) - (ctx.currentTime - startTimeRef.current);
            if (ahead < 1.5) {
              clearInterval(id);
              resolve();
            }
          }, 50);
        });
      }
    }
  }, [isPlaying, ctx, mediaInfoRef.current, fps, src, gainNode, volume, fadeIn, fadeOut, speed, hasValidCurrentFrame]);



  // Start or restart rendering when playback starts or media becomes ready
  useEffect(() => {
    if (!isPlaying) return;
    void startRendering();
  }, [isPlaying, ctx, mediaInfoRef.current, src, fps, startRendering, volume, fadeIn, fadeOut]);

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
    };
  }, []);

  return null;
}

export default AudioPreview;



import React, {  useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useControlsStore } from '@/lib/control';
import { AnyClipProps, MediaInfo } from '@/lib/types';
import { getMediaInfo } from '@/lib/media/utils';
import { getAudioIterator } from '@/lib/media/audio';
import { WrappedAudioBuffer } from 'mediabunny';

type ClipWithSrc = Extract<AnyClipProps, { src: string }>;

// Schedules audio playback for a clip in sync with the timeline. Renders nothing.
const AudioPreview: React.FC<ClipWithSrc & {framesToPrefetch?: number}> = ({ src, startFrame = 0, framesToGiveStart}) => {
  const [mediaInfo, setMediaInfo] = useState<MediaInfo | null>(null);
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
        if (!cancelled) setMediaInfo(info);
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
    if (!isPlaying || !ctx || !mediaInfo || !mediaInfo.audio || !fps) {
      return;
    }
    // align starting frame and playback anchor with current UI state
    if (typeof currentFrame === 'number' && fps) {
      currentStartFrameRef.current = currentFrame;
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

    iteratorRef.current = await getAudioIterator(src, { mediaInfo:mediaInfo || undefined, fps, sampleSize: mediaInfo?.audio?.numberOfChannels == 2 ? 1024 : 2048, index: currentStartFrameRef.current });
    for await (const buf of iteratorRef.current) {
      if (!buf) return;
      const buffer = buf.buffer || null;
      const duration = buf.duration || 0;
      const timestamp = buf.timestamp || 0;

      if (!buffer || duration <= 0) {
        return; // Stop the loop
      }

      const startTimestamp = startTimeRef.current + timestamp - playbackTimeAtStartRef.current;
      const node = ctx.createBufferSource();
		  node.buffer = buffer;
		  node.connect(gainNode);
      if (startTimestamp >= ctx.currentTime) {
        node.start(startTimestamp);
      } else {
        node.start(ctx.currentTime, ctx.currentTime - startTimestamp);
      }

      audioQueueRef.current.add(node);
      node.onended = () => {
        audioQueueRef.current.delete(node);
      };

      if (timestamp - getPlaybackTime() >= 1) {
        await new Promise<void>((resolve) => {
          const id = setInterval(() => {
            if (timestamp - getPlaybackTime() < 1) {
              clearInterval(id);
              resolve();
            }
          }, 100);
        });
      }
    }
  }, [isPlaying, ctx, mediaInfo, fps, src, gainNode]);

  // Start or restart rendering when playback starts or media becomes ready
  useEffect(() => {
    if (!isPlaying) return;
    void startRendering();
  }, [isPlaying, ctx, mediaInfo, src, fps, startRendering]);

  // Detect scrubbing while playing: if the UI's frame diverges from audio clock, resync
  useEffect(() => {
    if (!isPlaying || !fps) return;
    const predictedFrame = Math.round(getPlaybackTime() * fps);
    if (Math.abs(currentFrame - predictedFrame) > 2) {
      currentStartFrameRef.current = currentFrame;
      playbackTimeAtStartRef.current = currentFrame / fps;
      void startRendering();
    }
  }, [currentFrame, isPlaying, fps, startRendering]);

  // Cleanup on unmount: stop all audio immediately
  useEffect(() => {
    return () => {
      // Stop all audio buffer source nodes
      for (const node of audioQueueRef.current) {
        try { node.stop(); } catch {}
      }
      audioQueueRef.current.clear();
      
      // Stop the audio iterator if it exists
      // @ts-ignore
      iteratorRef.current?.return?.();
      iteratorRef.current = null;
    };
  }, []);

  return null;
}

export default AudioPreview;



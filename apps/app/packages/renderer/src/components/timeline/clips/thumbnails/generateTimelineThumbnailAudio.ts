import { AudioClipProps, MediaInfo } from "@/lib/types";
import { useClipStore } from "@/lib/clip";
import { getAudioIterator } from "@/lib/media/audio";
import { useControlsStore } from "@/lib/control";
import { getMediaInfo } from "@/lib/media/utils";

// Per-clip cancellation for progressive waveform rendering
const audioRenderSeqByClipId = new Map<string, number>();

export const generateTimelineThumbnailAudio = async (
  clipType: string,
  currentClip: AudioClipProps,
  currentClipId: string,
  mediaInfoRef: MediaInfo | null,
  imageCanvas: HTMLCanvasElement,
  timelineHeight: number,
  currentStartFrame: number,
  currentEndFrame: number,
  timelineDuration: [number, number],
  timelineWidth: number,
  timelinePadding: number,
  groupRef: any,
  noShift: boolean = false,
) => {
  if (clipType !== "audio") return;
  const mySeq = (audioRenderSeqByClipId.get(currentClipId) ?? 0) + 1;
  audioRenderSeqByClipId.set(currentClipId, mySeq);
  const speed = Math.max(
    0.1,
    Math.min(5, Number((currentClip as any)?.speed ?? 1)),
  );

  const height = timelineHeight;
  const timelineShift = noShift ? 0 : currentStartFrame - (currentClip.trimStart ?? 0);
  const visibleStartFrame = Math.max(currentStartFrame, timelineDuration[0]);
  const visibleEndFrame =
    Math.min(currentEndFrame, timelineDuration[1]) * speed;
  const duration = timelineDuration[1] - timelineDuration[0];

  const getAssetById = useClipStore.getState().getAssetById;
  const pixelsPerFrame = timelineWidth / duration;
  const positionOffsetStart = Math.round(
    Math.max(0, (currentStartFrame - timelineDuration[0]) * pixelsPerFrame),
  );
  const tClipWidth =
    Math.round(
      pixelsPerFrame * (visibleEndFrame - visibleStartFrame) +
        (positionOffsetStart === 0 ? timelinePadding : 0),
    ) / speed;

  const asset = getAssetById(currentClip.assetId);
  if (!asset) return;

  const ctx = imageCanvas.getContext("2d");
  if (!ctx) return;

  const targetH = Math.max(1, Math.floor(height));
  const targetW = Math.max(1, Math.floor(imageCanvas.width));
  const waveformW = Math.max(1, Math.floor(tClipWidth));

  // Base draw: clear + background immediately (so UI updates before decode finishes)
  ctx.clearRect(0, 0, targetW, targetH);
  ctx.imageSmoothingEnabled = true;
  // Keep this close to the legacy waveform background.
  ctx.fillStyle = "#1A2138";
  ctx.fillRect(0, 0, targetW, targetH);

  const offset = Math.max(0, Math.round(targetW - waveformW - positionOffsetStart));

  // Progressive bar rendering config (intentionally lightweight)
  const barWidth = 2;
  const gap = 1;
  const totalBarSpace = barWidth + gap;
  const barCount = Math.max(1, Math.floor((waveformW + gap) / totalBarSpace));
  const peaks = new Float32Array(barCount);

  const projectFps = Math.max(1, Math.round(useControlsStore.getState().fps || 30));
  const startIndex = Math.max(0, Math.floor(visibleStartFrame - timelineShift));
  const endIndex = Math.max(startIndex + 1, Math.floor(visibleEndFrame - timelineShift));
  const segStartSec = startIndex / projectFps;
  const segEndSec = endIndex / projectFps;
  const segDurationSec = Math.max(1e-6, segEndSec - segStartSec);
  const barDurationSec = segDurationSec / barCount;

  const color = "#7791C4";
  const volumeDb = Number((currentClip as any)?.volume ?? 0);
  const fadeInSec = Math.max(0, Number((currentClip as any)?.fadeIn ?? 0));
  const fadeOutSec = Math.max(0, Number((currentClip as any)?.fadeOut ?? 0));
  const dbToGain = (db: number) => Math.pow(10, db / 20);
  const volumeGain = dbToGain(volumeDb);
  const fadeInPx = (fadeInSec / segDurationSec) * waveformW;
  const fadeOutPx = (fadeOutSec / segDurationSec) * waveformW;
  const fadeOutStartPx = waveformW - fadeOutPx;

  const drawRoundedTopBar = (
    c: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
  ) => {
    const radius = Math.max(0, Math.min(r, Math.min(w, h) / 2));
    if (h < radius || w < radius * 2) {
      c.fillRect(x, y, w, h);
      return;
    }
    c.beginPath();
    c.moveTo(x, y + h);
    c.lineTo(x, y + radius);
    c.arcTo(x, y, x + radius, y, radius);
    c.lineTo(x + w - radius, y);
    c.arcTo(x + w, y, x + w, y + radius, radius);
    c.lineTo(x + w, y + h);
    c.closePath();
    c.fill();
  };

  const drawBar = (barIndex: number, peak: number) => {
    const xLocal = barIndex * totalBarSpace;
    const x = offset + xLocal;
    if (x > targetW) return;
    if (x + barWidth < 0) return;

    // Apply volume + fades (best-effort, keeps waveform responsive)
    let mult = volumeGain;
    if (fadeInSec > 0 && xLocal < fadeInPx) {
      mult *= xLocal / Math.max(1, fadeInPx);
    }
    if (fadeOutSec > 0 && xLocal >= fadeOutStartPx) {
      mult *= (waveformW - xLocal) / Math.max(1, fadeOutPx);
    }

    const amp = Math.max(0, Math.min(1, peak * mult));
    const h = Math.max(1, Math.round(amp * targetH));
    const y = targetH - h;

    ctx.fillStyle = color;
    drawRoundedTopBar(ctx, x, y, barWidth, h, barWidth / 2);
  };

  // Decode and blit progressively from left -> right
  let nextBarToDraw = 0;
  let processedUntilSec = 0;
  let lastBatchDrawAt = 0;

  try {
    const ensuredMediaInfo =
      mediaInfoRef ?? (await getMediaInfo(asset.path).catch(() => undefined));

    const iter = await getAudioIterator(asset.path, {
      mediaInfo: ensuredMediaInfo,
      fps: projectFps,
      startIndex,
      endIndex,
    });

    for await (const item of iter) {
      if ((audioRenderSeqByClipId.get(currentClipId) ?? 0) !== mySeq) return;
      if (!item?.buffer) continue;

      const buf = item.buffer;
      const sr = Math.max(1, Math.round(buf.sampleRate || 1));
      const bufStartSec = item.timestamp;
      const bufEndSec = item.timestamp + item.duration;
      const bufRelStartSec = bufStartSec - segStartSec;
      const bufRelEndSec = bufEndSec - segStartSec;

      const relStart = Math.max(0, bufRelStartSec);
      const relEnd = Math.min(segDurationSec, bufRelEndSec);
      if (relEnd <= 0 || relStart >= segDurationSec) continue;

      const firstBar = Math.max(0, Math.floor(relStart / barDurationSec));
      const lastBar = Math.min(
        barCount - 1,
        Math.max(firstBar, Math.ceil(relEnd / barDurationSec) - 1),
      );

      // Pull channel views once for speed
      const chCount = Math.max(1, buf.numberOfChannels || 1);
      const channels: Float32Array[] = [];
      for (let ch = 0; ch < chCount; ch++) {
        try {
          channels.push(buf.getChannelData(ch));
        } catch {
          // ignore missing channel
        }
      }
      const channelsUsed = Math.max(1, channels.length);

      for (let b = firstBar; b <= lastBar; b++) {
        const barStartSec = b * barDurationSec;
        const barEndSec = Math.min(segDurationSec, (b + 1) * barDurationSec);
        const overlapStart = Math.max(barStartSec, bufRelStartSec);
        const overlapEnd = Math.min(barEndSec, bufRelEndSec);
        if (overlapEnd <= overlapStart) continue;

        const s0 = Math.max(0, Math.floor((overlapStart - bufRelStartSec) * sr));
        const s1 = Math.min(
          buf.length,
          Math.max(s0 + 1, Math.ceil((overlapEnd - bufRelStartSec) * sr)),
        );
        const span = Math.max(1, s1 - s0);
        const stride = Math.max(1, Math.floor(span / 64));

        let peak = 0;
        for (let i = s0; i < s1; i += stride) {
          let v = 0;
          for (let ch = 0; ch < channelsUsed; ch++) {
            v += Math.abs(channels[ch]![i] || 0);
          }
          v /= channelsUsed;
          if (v > peak) peak = v;
        }
        if (peak > peaks[b]!) peaks[b] = peak;
      }

      processedUntilSec = Math.max(processedUntilSec, relEnd);
      const drawUntil = Math.min(
        barCount,
        Math.floor(processedUntilSec / barDurationSec),
      );
      while (nextBarToDraw < drawUntil) {
        drawBar(nextBarToDraw, peaks[nextBarToDraw]!);
        nextBarToDraw++;
      }

      const now = performance.now();
      if (now - lastBatchDrawAt > 50) {
        lastBatchDrawAt = now;
        groupRef.current?.getLayer()?.batchDraw();
      }
    }
  } catch {
    // Best-effort: if streaming decode fails, leave the background in place.
  } finally {
    if ((audioRenderSeqByClipId.get(currentClipId) ?? 0) === mySeq) {
      // Draw any remaining bars (may be silent / 0)
      while (nextBarToDraw < barCount) {
        drawBar(nextBarToDraw, peaks[nextBarToDraw]!);
        nextBarToDraw++;
      }
      groupRef.current?.getLayer()?.batchDraw();
    }
  }

  //moveClipToEnd(currentClipId);
};

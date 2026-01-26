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

  // --- Modern waveform styling (mirrors `lib/media/timeline.ts`) ---
  ctx.imageSmoothingEnabled = true;
  // @ts-ignore - not available in all browsers/contexts
  ctx.imageSmoothingQuality = "high";

  const offset = Math.max(0, Math.round(targetW - waveformW - positionOffsetStart));

  // Progressive bar rendering config (intentionally lightweight)
  const barWidth = 2;
  const gap = 1;
  const totalBarSpace = barWidth + gap;
  const barCount = Math.max(1, Math.floor((waveformW + gap) / totalBarSpace));
  // Store per-bar max RMS (0..1). We'll map to 0..100 like the timeline renderer.
  const rmsByBar = new Float32Array(barCount);

  const SOURCE_DB_MIN = -60.0; // silence floor
  const SOURCE_DB_MAX = 0.0; // max loudness
  const TARGET_VISUAL_MIN = 0.0;
  const TARGET_VISUAL_MAX = 100.0;

  const amplitudeToDBFS = (amplitude: number, minDb: number) => {
    if (!Number.isFinite(amplitude) || amplitude <= 0) return minDb;
    const db = 20 * Math.log10(amplitude);
    return Math.max(minDb, db);
  };
  const mapRange = (
    v: number,
    inMin: number,
    inMax: number,
    outMin: number,
    outMax: number,
  ) => {
    if (inMax === inMin) return outMin;
    const t = (v - inMin) / (inMax - inMin);
    const clamped = Math.max(0, Math.min(1, t));
    return outMin + clamped * (outMax - outMin);
  };

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

  // --- Color utilities (local, no effect on data) ---
  const clamp255 = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const hexToRgb = (
    hex: string,
  ): { r: number; g: number; b: number } | null => {
    const m = hex.trim().replace("#", "");
    if (m.length === 3) {
      const r = parseInt(m[0] + m[0], 16);
      const g = parseInt(m[1] + m[1], 16);
      const b = parseInt(m[2] + m[2], 16);
      return { r, g, b };
    }
    if (m.length === 6) {
      const r = parseInt(m.slice(0, 2), 16);
      const g = parseInt(m.slice(2, 4), 16);
      const b = parseInt(m.slice(4, 6), 16);
      return { r, g, b };
    }
    return null;
  };
  const rgba = (hex: string, alpha: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
  };
  const lighten = (hex: string, pct: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    const r = clamp255(rgb.r + (255 - rgb.r) * (pct / 100));
    const g = clamp255(rgb.g + (255 - rgb.g) * (pct / 100));
    const b = clamp255(rgb.b + (255 - rgb.b) * (pct / 100));
    return `rgb(${r},${g},${b})`;
  };
  const darken = (hex: string, pct: number) => {
    const rgb = hexToRgb(hex);
    if (!rgb) return hex;
    const r = clamp255(rgb.r * (1 - pct / 100));
    const g = clamp255(rgb.g * (1 - pct / 100));
    const b = clamp255(rgb.b * (1 - pct / 100));
    return `rgb(${r},${g},${b})`;
  };

  // Background canvas (draw once; reused during progressive compositing)
  const bgCanvas = document.createElement("canvas");
  bgCanvas.width = targetW;
  bgCanvas.height = targetH;
  const bgctx = bgCanvas.getContext("2d");
  if (!bgctx) return;
  bgctx.imageSmoothingEnabled = true;
  // @ts-ignore - not available in all browsers/contexts
  bgctx.imageSmoothingQuality = "high";

  const backgroundGradient = bgctx.createLinearGradient(0, 0, 0, bgCanvas.height);
  backgroundGradient.addColorStop(0, "#121832");
  backgroundGradient.addColorStop(1, "#0B0F1F");
  bgctx.fillStyle = backgroundGradient;
  bgctx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

  const vignette = bgctx.createLinearGradient(0, 0, bgCanvas.width, 0);
  vignette.addColorStop(0, "rgba(0,0,0,0.35)");
  vignette.addColorStop(0.5, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,0.35)");
  bgctx.fillStyle = vignette;
  bgctx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

  // Grid and ticks (behind bars)
  bgctx.save();
  bgctx.strokeStyle = "rgba(255,255,255,0.06)";
  bgctx.lineWidth = 1;
  bgctx.setLineDash([2, 6]);
  const gridRows = [0.25, 0.5, 0.75];
  gridRows.forEach((fraction) => {
    const y = Math.round(bgCanvas.height * fraction) + 0.5;
    bgctx.beginPath();
    bgctx.moveTo(0, y);
    bgctx.lineTo(bgCanvas.width, y);
    bgctx.stroke();
  });
  bgctx.setLineDash([]);

  for (let x = 0; x <= bgCanvas.width; x += 10) {
    const isMajor = x % 50 === 0;
    const tickHeight = isMajor ? 8 : 4;
    const alpha = isMajor ? 0.12 : 0.06;
    bgctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    bgctx.beginPath();
    const xAligned = Math.round(x) + 0.5;
    bgctx.moveTo(xAligned, 0);
    bgctx.lineTo(xAligned, tickHeight);
    bgctx.moveTo(xAligned, bgCanvas.height);
    bgctx.lineTo(xAligned, bgCanvas.height - tickHeight);
    bgctx.stroke();
  }
  bgctx.restore();

  // Offscreen canvas for bars (for fast compositing and single-pass glow)
  const barsCanvas = document.createElement("canvas");
  barsCanvas.width = targetW;
  barsCanvas.height = targetH;
  const bctx = barsCanvas.getContext("2d");
  if (!bctx) return;
  bctx.imageSmoothingEnabled = true;
  // @ts-ignore - not available in all browsers/contexts
  bctx.imageSmoothingQuality = "high";

  const barColor = color;
  const lightBar = lighten(barColor, 28);
  const darkBar = darken(barColor, 18);
  const barFillGradient = bctx.createLinearGradient(0, 0, 0, targetH);
  barFillGradient.addColorStop(0, lightBar);
  barFillGradient.addColorStop(0.6, barColor);
  barFillGradient.addColorStop(1, darkBar);
  bctx.fillStyle = barFillGradient;

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

  const drawBar = (barIndex: number, rms: number) => {
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

    // Map RMS -> dBFS -> 0..100 visual like the timeline renderer
    const dbfsValue = amplitudeToDBFS(rms, SOURCE_DB_MIN);
    const visualValue = mapRange(
      dbfsValue,
      SOURCE_DB_MIN,
      SOURCE_DB_MAX,
      TARGET_VISUAL_MIN,
      TARGET_VISUAL_MAX,
    );

    const barHeight: number = Math.min(
      (visualValue / 100) * targetH * mult,
      targetH,
    );
    if (visualValue <= 0 || barHeight <= 0) return;
    const h = Math.round(barHeight);
    const y = targetH - h;

    drawRoundedTopBar(bctx, x, y, barWidth, h, barWidth / 2);
  };

  const simplifyDataForLine = (
    values: number[],
    chunkSize: number,
  ): { x: number; y: number }[] => {
    const points: { x: number; y: number }[] = [];
    if (chunkSize <= 0) return points;

    for (let i = 0; i < values.length; i += chunkSize) {
      const end = Math.min(values.length, i + chunkSize);
      if (end <= i) continue;

      let sum = 0;
      for (let j = i; j < end; j++) sum += values[j] ?? 0;
      const avg = sum / (end - i);

      const barHeight = (avg / 100) * targetH;
      const chunkCenterIndex = i + (end - i) / 2;
      const x = offset + chunkCenterIndex * totalBarSpace;
      const y = targetH - barHeight;
      points.push({ x, y });
    }
    return points;
  };

  const renderComposite = (valuesForLine: number[], drawLine: boolean) => {
    ctx.clearRect(0, 0, targetW, targetH);
    ctx.drawImage(bgCanvas, 0, 0);

    // Glow pass + crisp bars (barsCanvas already contains bars drawn so far)
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    // @ts-ignore - filter not available everywhere
    ctx.filter = "blur(4px)";
    ctx.globalAlpha = 0.45;
    ctx.drawImage(barsCanvas, 0, 0);
    ctx.restore();

    ctx.drawImage(barsCanvas, 0, 0);

    if (drawLine) {
      const lineChunkSize = targetW > 240 ? 8 : 4;
      const linePoints = simplifyDataForLine(valuesForLine, lineChunkSize);
      if (linePoints.length > 1) {
        // Glow pass
        ctx.save();
        ctx.beginPath();
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = rgba(lighten(barColor, 25), 0.5);
        ctx.shadowBlur = 10;
        ctx.strokeStyle = rgba(lighten(barColor, 15), 0.85);
        ctx.moveTo(linePoints[0].x, linePoints[0].y);
        for (let i = 1; i < linePoints.length - 1; i++) {
          const xc = (linePoints[i].x + linePoints[i + 1].x) / 2;
          const yc = (linePoints[i].y + linePoints[i + 1].y) / 2;
          ctx.quadraticCurveTo(linePoints[i].x, linePoints[i].y, xc, yc);
        }
        const lastGlow = linePoints.length - 1;
        ctx.lineTo(linePoints[lastGlow].x, linePoints[lastGlow].y);
        ctx.stroke();
        ctx.restore();

        // Main line pass
        ctx.beginPath();
        const lineGradient = ctx.createLinearGradient(0, 0, 0, targetH);
        lineGradient.addColorStop(0, "#F2F2F2");
        lineGradient.addColorStop(1, rgba("#E8E8E8", 0.85));
        ctx.strokeStyle = lineGradient;
        ctx.lineWidth = 1.6;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.moveTo(linePoints[0].x, linePoints[0].y);
        for (let i = 1; i < linePoints.length - 1; i++) {
          const xc = (linePoints[i].x + linePoints[i + 1].x) / 2;
          const yc = (linePoints[i].y + linePoints[i + 1].y) / 2;
          ctx.quadraticCurveTo(linePoints[i].x, linePoints[i].y, xc, yc);
        }
        const last = linePoints.length - 1;
        ctx.lineTo(linePoints[last].x, linePoints[last].y);
        ctx.stroke();
      }
    }

    // Subtle top highlight overlay
    const topHighlight = ctx.createLinearGradient(0, 0, 0, targetH);
    topHighlight.addColorStop(0, "rgba(255,255,255,0.06)");
    topHighlight.addColorStop(0.15, "rgba(255,255,255,0.02)");
    topHighlight.addColorStop(0.5, "rgba(255,255,255,0)");
    ctx.fillStyle = topHighlight;
    ctx.fillRect(0, 0, targetW, targetH);

    // Bottom edge subtle separator
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.floor(targetH - 0.5) + 0.5);
    ctx.lineTo(targetW, Math.floor(targetH - 0.5) + 0.5);
    ctx.stroke();
    ctx.restore();
  };

  // Base draw immediately (so UI updates before decode finishes)
  renderComposite([], false);
  groupRef.current?.getLayer()?.batchDraw();

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

        // Approximate RMS over the overlapped span (stride-sampled)
        let sumEnergy = 0;
        let count = 0;
        for (let i = s0; i < s1; i += stride) {
          let e = 0;
          for (let ch = 0; ch < channelsUsed; ch++) {
            const s = channels[ch]![i] || 0;
            e += s * s;
          }
          e /= channelsUsed;
          sumEnergy += e;
          count++;
        }
        const rms = Math.sqrt(sumEnergy / Math.max(1, count));
        if (rms > (rmsByBar[b] ?? 0)) rmsByBar[b] = rms;
      }

      processedUntilSec = Math.max(processedUntilSec, relEnd);
      const drawUntil = Math.min(
        barCount,
        Math.floor(processedUntilSec / barDurationSec),
      );
      while (nextBarToDraw < drawUntil) {
        drawBar(nextBarToDraw, rmsByBar[nextBarToDraw] ?? 0);
        nextBarToDraw++;
      }

      const now = performance.now();
      if (now - lastBatchDrawAt > 50) {
        lastBatchDrawAt = now;
        // Build line values only when we're going to draw (keeps hot path light)
        const valuesForLine: number[] = new Array(barCount);
        for (let i = 0; i < barCount; i++) {
          const dbfsValue = amplitudeToDBFS(rmsByBar[i] ?? 0, SOURCE_DB_MIN);
          const base = mapRange(
            dbfsValue,
            SOURCE_DB_MIN,
            SOURCE_DB_MAX,
            TARGET_VISUAL_MIN,
            TARGET_VISUAL_MAX,
          );
          // Apply volume + fades so the line matches the bars
          const xLocal = i * totalBarSpace;
          let mult = volumeGain;
          if (fadeInSec > 0 && xLocal < fadeInPx) {
            mult *= xLocal / Math.max(1, fadeInPx);
          }
          if (fadeOutSec > 0 && xLocal >= fadeOutStartPx) {
            mult *= (waveformW - xLocal) / Math.max(1, fadeOutPx);
          }
          valuesForLine[i] = Math.min(base * mult, 100);
        }
        renderComposite(valuesForLine, true);
        groupRef.current?.getLayer()?.batchDraw();
      }
    }
  } catch {
    // Best-effort: if streaming decode fails, leave the background in place.
  } finally {
    if ((audioRenderSeqByClipId.get(currentClipId) ?? 0) === mySeq) {
      // Draw any remaining bars (may be silent / 0)
      while (nextBarToDraw < barCount) {
        drawBar(nextBarToDraw, rmsByBar[nextBarToDraw] ?? 0);
        nextBarToDraw++;
      }
      const valuesForLine: number[] = new Array(barCount);
      for (let i = 0; i < barCount; i++) {
        const dbfsValue = amplitudeToDBFS(rmsByBar[i] ?? 0, SOURCE_DB_MIN);
        const base = mapRange(
          dbfsValue,
          SOURCE_DB_MIN,
          SOURCE_DB_MAX,
          TARGET_VISUAL_MIN,
          TARGET_VISUAL_MAX,
        );
        const xLocal = i * totalBarSpace;
        let mult = volumeGain;
        if (fadeInSec > 0 && xLocal < fadeInPx) {
          mult *= xLocal / Math.max(1, fadeInPx);
        }
        if (fadeOutSec > 0 && xLocal >= fadeOutStartPx) {
          mult *= (waveformW - xLocal) / Math.max(1, fadeOutPx);
        }
        valuesForLine[i] = Math.min(base * mult, 100);
      }
      renderComposite(valuesForLine, true);
      groupRef.current?.getLayer()?.batchDraw();
    }
  }

  //moveClipToEnd(currentClipId);
};

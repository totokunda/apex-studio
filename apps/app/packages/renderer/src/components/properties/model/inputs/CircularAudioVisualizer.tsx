import React, { useCallback, useEffect, useRef } from "react";

interface CircularAudioVisualizerProps {
  inputId: string;
  width: number;
  height: number;
  active: boolean;
}

// Visualizes current audio around a circle using analyser data keyed by inputId.
export const CircularAudioVisualizer: React.FC<
  CircularAudioVisualizerProps
> = ({ inputId, width, height, active }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataRef = useRef<Uint8Array | null>(null);
  const prevBarsRef = useRef<Float32Array | null>(null);
  const timeDomainRef = useRef<Uint8Array | null>(null);
  const zeroFrameStreakRef = useRef<number>(0);

  const attachAnalyser = useCallback(() => {
    try {
      const store: any = window as any;
      const map = store.__apexAudioAnalysers as
        | Map<string, { ctx: AudioContext; analyser: AnalyserNode }>
        | undefined;
      if (map && map.has(inputId)) {
        const entry = map.get(inputId)!;
        analyserRef.current = entry.analyser;
        dataRef.current = new Uint8Array(
          entry.analyser.frequencyBinCount,
        ) as unknown as Uint8Array;
        return true;
      }
    } catch {
      // ignore
    }
    return false;
  }, [inputId]);

  useEffect(() => {
    if (!attachAnalyser()) {
      const onReady = (e: Event) => {
        const detail = (e as CustomEvent).detail;
        if (detail?.inputId === String(inputId)) attachAnalyser();
      };
      window.addEventListener("apex:audio:analyser-ready", onReady as any, {
        once: true,
      });
      return () =>
        window.removeEventListener("apex:audio:analyser-ready", onReady as any);
    }
  }, [attachAnalyser, inputId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, [width, height]);

  useEffect(() => {
    const loop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      let analyser = analyserRef.current;
      let data = dataRef.current;
      if (!analyser || !data) {
        attachAnalyser();
        analyser = analyserRef.current;
        data = dataRef.current;
      }
      if (!canvas || !ctx) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      const w = Math.max(1, width);
      const h = Math.max(1, height);
      // prepare bar state
      const barCount = 72;
      if (!prevBarsRef.current || prevBarsRef.current.length !== barCount) {
        prevBarsRef.current = new Float32Array(barCount);
      }
      const bars = prevBarsRef.current;

      // sample spectrum if available, otherwise decay previous values
      if (active && analyser && data) {
        analyser.getByteFrequencyData(data as any);
        let zero = true;
        for (let i = 0; i < (data as Uint8Array).length; i++) {
          if ((data as Uint8Array)[i] > 0) {
            zero = false;
            break;
          }
        }
        if (zero) {
          zeroFrameStreakRef.current++;
        } else {
          zeroFrameStreakRef.current = 0;
        }
        if (zeroFrameStreakRef.current >= 3) {
          // fallback to time-domain RMS when freq bins are flat (e.g., small signals or platform quirks)
          if (
            !timeDomainRef.current ||
            timeDomainRef.current.length !== analyser.fftSize
          ) {
            timeDomainRef.current = new Uint8Array(analyser.fftSize);
          }
          analyser.getByteTimeDomainData(timeDomainRef.current as any);
          let sum = 0;
          for (let i = 0; i < timeDomainRef.current.length; i++) {
            const v = (timeDomainRef.current[i] - 128) / 128; // -1..1
            sum += v * v;
          }
          const rms = Math.min(
            1,
            Math.sqrt(sum / timeDomainRef.current.length) * 2,
          );
          for (let i = 0; i < barCount; i++) {
            const jitter = (Math.sin((i * 12.9898) % 6.283) * 0.5 + 0.5) * 0.08;
            const target = Math.max(0, Math.min(1, rms * (0.85 + jitter)));
            const smooth = 0.35;
            bars[i] = bars[i] + (target - bars[i]) * smooth;
          }
        } else {
          for (let i = 0; i < barCount; i++) {
            const idx = Math.min(
              (data as Uint8Array).length - 1,
              Math.floor((i / barCount) * (data as Uint8Array).length),
            );
            const amp = (data as Uint8Array)[idx] / 255;
            const target = Math.max(0, Math.min(1, amp));
            const smooth = 0.35;
            bars[i] = bars[i] + (target - bars[i]) * smooth;
          }
        }
      } else {
        for (let i = 0; i < barCount; i++) bars[i] *= 0.92;
      }

      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h / 2;
      const minDim = Math.min(w, h);
      const innerRadius = Math.max(14, Math.floor(minDim * 0.22));
      const maxBar = Math.floor(minDim * 0.16);
      for (let i = 0; i < barCount; i++) {
        const angle = (i / barCount) * Math.PI * 2;
        const length = 2 + bars[i] * maxBar;
        const x0 = cx + Math.cos(angle) * innerRadius;
        const y0 = cy + Math.sin(angle) * innerRadius;
        const x1 = cx + Math.cos(angle) * (innerRadius + length);
        const y1 = cy + Math.sin(angle) * (innerRadius + length);
        const hue = Math.round((i / barCount) * 360);
        ctx.strokeStyle = `hsl(${hue} 90% 60%)`;
        ctx.lineWidth = Math.max(1, Math.floor(minDim * 0.006));
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
      // inner ring for visual stability
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = Math.max(1, Math.floor(minDim * 0.004));
      ctx.beginPath();
      ctx.arc(cx, cy, innerRadius - ctx.lineWidth * 0.5, 0, Math.PI * 2);
      ctx.stroke();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [active, attachAnalyser, width, height]);

  return (
    <div
      className="relative bg-brand-background-light/80 rounded-[7px] border border-brand-light/5"
      style={{ width: Math.max(1, width), height: Math.max(1, height) }}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-20 pointer-events-none"
      />
    </div>
  );
};

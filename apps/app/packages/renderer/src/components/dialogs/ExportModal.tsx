import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import NumberInput from "@/components/properties/model/inputs/NumberInput";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { getMediaInfoCached } from "@/lib/media/utils";
import { pickMediaPaths, resolvePath } from "@app/preload";
import { TbCancel, TbMovie } from "react-icons/tb";
import { ProgressBar } from "@/components/common/ProgressBar";
import { LuLoaderCircle } from "react-icons/lu";
import { ModelClipProps, VideoClipProps } from "@/lib/types";
type ResolutionOption = {
  label: string;
  value: number;
};

type ExportKind = "video" | "image";

type BitrateOption = {
  label: string;
  value: string;
};

type CodecOption = {
  label: string;
  value: "h264" | "hevc" | "vp9" | "av1" | "prores";
};

type FormatOption = {
  label: string;
  value: "mp4" | "mov" | "mkv" | "webm";
};

type ImageFormatOption = {
  label: string;
  value: "png" | "jpeg" | "webp";
};

type AudioFormatOption = {
  label: string;
  value: "wav" | "mp3";
};

export type ExportSettings =
  | {
      kind: "video";
      name: string;
      path: string;
      resolution: number;
      bitrate: string;
      codec: CodecOption["value"];
      format: FormatOption["value"];
      includeAudio: boolean;
      audioFormat?: AudioFormatOption["value"];
      preserveAlpha?: boolean;
    }
  | {
      kind: "image";
      name: string;
      path: string;
      resolution: number;
      imageFormat: ImageFormatOption["value"];
      frame: number; // 0-based project frame index
      preserveAlpha?: boolean;
    };

interface ExportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExport: (settings: ExportSettings) => void;
  defaultName?: string;
  defaultPath?: string;
  // Optional: external export state so this dialog can reflect progress even
  // when reopened while an export is already running.
  isExporting?: boolean;
  exportProgress?: number | null;
  /**
   * Optional callback invoked when the user explicitly requests export
   * cancellation from this dialog.
   */
  onCancelExport?: () => void;
}

const RESOLUTION_OPTIONS: ResolutionOption[] = [
  { label: "480P", value: 480 },
  { label: "720P", value: 720 },
  { label: "1080P", value: 1080 },
  { label: "2K", value: 2048 },
  { label: "4K", value: 4096 },
];

const BITRATE_OPTIONS: BitrateOption[] = [
  { label: "Low", value: "5M" },
  { label: "Medium", value: "10M" },
  { label: "High", value: "20M" },
];

const CODEC_OPTIONS: CodecOption[] = [
  { label: "H.264", value: "h264" },
  { label: "H.265 (HEVC)", value: "hevc" },
  { label: "VP9", value: "vp9" },
  { label: "AV1", value: "av1" },
  { label: "ProRes", value: "prores" },
];

const FORMAT_OPTIONS: FormatOption[] = [
  { label: "MP4", value: "mp4" },
  { label: "MOV", value: "mov" },
  { label: "WEBM", value: "webm" },
  { label: "MKV", value: "mkv" },
];

const IMAGE_FORMAT_OPTIONS: ImageFormatOption[] = [
  { label: "PNG", value: "png" },
  { label: "JPEG", value: "jpeg" },
  { label: "WebP", value: "webp" },
];

const AUDIO_FORMAT_OPTIONS: AudioFormatOption[] = [
  { label: "WAV", value: "wav" },
  { label: "MP3", value: "mp3" },
];

const fieldLabelClass =
  "text-brand-light text-[10.5px] font-medium mb-1.5 flex items-center justify-between";

export const ExportModal: React.FC<ExportModalProps> = ({
  open,
  onOpenChange,
  onExport,
  defaultName = "Export",
  defaultPath = "~/Downloads",
  isExporting = false,
  exportProgress = null,
  onCancelExport,
}) => {
  const playheadFrame = useControlsStore((s) => s.focusFrame);
  const playheadFps = useControlsStore((s) => s.fps);

  const [kind, setKind] = useState<ExportKind>("video");
  const [name, setName] = useState(defaultName);
  const [path, setPath] = useState(defaultPath);
  const [resolution, setResolution] = useState<number>(
    RESOLUTION_OPTIONS[0]?.value ?? 0,
  );
  const [bitrate, setBitrate] = useState<string>(
    BITRATE_OPTIONS[1]?.value ?? "",
  );
  const [codec, setCodec] = useState<CodecOption["value"]>(
    CODEC_OPTIONS[0]?.value ?? "h264",
  );
  const [format, setFormat] = useState<FormatOption["value"]>(
    FORMAT_OPTIONS[0]?.value ?? "mp4",
  );
  const [includeAudio, setIncludeAudio] = useState<boolean>(true);
  const [audioFormat, setAudioFormat] = useState<AudioFormatOption["value"]>(
    AUDIO_FORMAT_OPTIONS[0]?.value ?? "mp3",
  );

  const [imageFormat, setImageFormat] = useState<ImageFormatOption["value"]>(
    IMAGE_FORMAT_OPTIONS[0]?.value ?? "png",
  );
  const [imageFrame, setImageFrame] = useState<number>(0);

  const [preserveAlpha, setPreserveAlpha] = useState<boolean>(false);

  const [hasAnyAudio, setHasAnyAudio] = useState<boolean>(false);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [durationFrames, setDurationFrames] = useState<number | null>(null);
  const [durationFps, setDurationFps] = useState<number | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const exportStartRef = useRef<number | null>(null);
  const [tryCancelExport, setTryCancelExport] = useState<boolean>(false);

  const safeProgressPercent = useMemo(() => {
    const raw = typeof exportProgress === "number" ? exportProgress : 0;
    const clamped = Math.max(0, Math.min(1, raw));
    return Math.round(clamped * 100);
  }, [exportProgress]);

  const maxSelectableFrame = useMemo(() => {
    return Math.max(0, (durationFrames ?? 1) - 1);
  }, [durationFrames]);

  const safeImageFrame = useMemo(() => {
    const n = Number(imageFrame);
    return Number.isFinite(n) ? Math.max(0, Math.round(n)) : 0;
  }, [imageFrame]);

  useEffect(() => {
    // Compute duration and audio presence from global stores once per open
    try {
      const clipState = useClipStore.getState();
      const controlsState = useControlsStore.getState();
      const { clips, clipDuration } = clipState;
      const fps = controlsState.fps || 24;

      const anyAudio =
        Array.isArray(clips) &&
        clips.some((clip) => {
          if ((clip as any).hidden) return false;
          if (clip.type === "audio") return true;
          if (clip.type === "video") {
            const info = getMediaInfoCached((clip as VideoClipProps).assetId);
            return info?.audio !== null;
          }
          if (clip.type === "model") {
            const info = getMediaInfoCached((clip as ModelClipProps).assetId);
            return info?.audio !== null;
          }
          return false;
        });

      setHasAnyAudio(anyAudio);
      if (!anyAudio) {
        setIncludeAudio(false);
      }

      const totalFrames = Number(clipDuration || 0);
      if (
        Number.isFinite(totalFrames) &&
        totalFrames > 0 &&
        Number.isFinite(fps) &&
        fps > 0
      ) {
        const seconds = totalFrames / fps;
        setDurationSeconds(seconds);
        setDurationFrames(totalFrames);
        setDurationFps(fps);
      } else {
        setDurationSeconds(null);
        setDurationFrames(null);
        setDurationFps(null);
      }

      // Default the image frame picker to the current playhead (clamped).
      const maxFrame = Math.max(0, Math.max(0, totalFrames) - 1);
      const desired = Math.max(0, Math.round(controlsState.focusFrame || 0));
      setImageFrame(Math.min(desired, maxFrame));
    } catch {
      setHasAnyAudio(false);
      setDurationSeconds(null);
      setDurationFrames(null);
      setDurationFps(null);
    }
  }, [open]);

  // Ensure that when preserving alpha, we pick a format/codec combo that supports transparency.
  useEffect(() => {
    if (kind !== "video") return;
    if (!preserveAlpha) return;

    const formatSupportsAlpha = (f: FormatOption["value"]) =>
      f === "mov" || f === "webm" || f === "mkv";
    const codecSupportsAlpha = (c: CodecOption["value"]) =>
      c === "prores" || c === "vp9" || c === "av1";

    if (!formatSupportsAlpha(format) || !codecSupportsAlpha(codec)) {
      // Default to a well-supported alpha combo.
      setFormat("mov");
      setCodec("prores");
    }
  }, [kind, preserveAlpha, format, codec]);

  // JPEG cannot preserve alpha.
  useEffect(() => {
    if (kind !== "image") return;
    if (imageFormat === "jpeg" && preserveAlpha) {
      setPreserveAlpha(false);
    }
  }, [kind, imageFormat, preserveAlpha]);

  const durationLabel = useMemo(() => {
    const totalFrames = durationFrames;
    const effectiveFps = durationFps;
    if (
      totalFrames === null ||
      effectiveFps === null ||
      !Number.isFinite(totalFrames) ||
      !Number.isFinite(effectiveFps) ||
      effectiveFps <= 0
    ) {
      return "—";
    }
    const seconds = durationSeconds ?? totalFrames / effectiveFps;
    if (!Number.isFinite(seconds) || seconds <= 0) return "—";
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    }
    const mins = Math.floor(seconds / 60);
    const rem = seconds - mins * 60;
    const remStr = rem < 10 ? `0${rem.toFixed(1)}` : rem.toFixed(1);
    return `${mins}m ${remStr}s`;
  }, [durationSeconds, durationFrames, durationFps]);

  useEffect(() => {
    if (!isExporting) {
      exportStartRef.current = null;
      setEtaSeconds(null);
      return;
    }

    const progress =
      typeof exportProgress === "number"
        ? Math.max(0, Math.min(1, exportProgress))
        : 0;

    if (progress <= 0 || progress >= 1) {
      if (progress >= 1) {
        exportStartRef.current = null;
        setEtaSeconds(null);
      }
      return;
    }

    const now =
      typeof performance !== "undefined" &&
      typeof performance.now === "function"
        ? performance.now()
        : Date.now();

    if (exportStartRef.current === null) {
      exportStartRef.current = now;
      return;
    }

    const elapsedSec = (now - exportStartRef.current) / 1000;
    if (!Number.isFinite(elapsedSec) || elapsedSec <= 0) return;

    const estimatedTotalSec = elapsedSec / progress;
    const remainingSec = Math.max(0, estimatedTotalSec - elapsedSec);
    if (Number.isFinite(remainingSec)) setEtaSeconds(remainingSec);
  }, [isExporting, exportProgress]);

  const etaLabel = useMemo(() => {
    if (
      !isExporting ||
      etaSeconds === null ||
      !Number.isFinite(etaSeconds) ||
      etaSeconds <= 0
    ) {
      return null;
    }
    const totalSeconds = Math.round(etaSeconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    const parts: string[] = [];
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    if (hours === 0 && minutes === 0) {
      parts.push(`${seconds}s`);
    } else if (seconds > 0 && hours === 0) {
      parts.push(`${seconds}s`);
    }

    return parts.join(" ");
  }, [isExporting, etaSeconds]);

  const handleExportClick = () => {
    if (isExporting) return;
    const trimmedName = name.trim();
    const trimmedPath = resolvePath(path.trim());
    if (!trimmedName || !trimmedPath) return;

    if (kind === "image") {
      const safeFrame = Math.max(0, Math.min(maxSelectableFrame, safeImageFrame));
      onExport({
        kind: "image",
        name: trimmedName,
        path: trimmedPath,
        resolution,
        imageFormat,
        frame: safeFrame,
        preserveAlpha,
      });
      return;
    }

    onExport({
      kind: "video",
      name: trimmedName,
      path: trimmedPath,
      resolution,
      bitrate,
      codec,
      format,
      includeAudio,
      audioFormat: includeAudio ? audioFormat : undefined,
      preserveAlpha,
    });
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      onOpenChange(false);
    } else {
      onOpenChange(true);
    }
  };

  useEffect(() => {
    if (!isExporting) {
      setTryCancelExport(false);
    }
  }, [isExporting]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg w-full bg-brand-background/90 backdrop-blur-sm text-foreground p-0 gap-0 overflow-hidden dark font-poppins border border-brand-light/10">
        <DialogHeader className="px-5 py-3.5 border-b border-brand-light/10 flex-shrink-0 bg-brand-background">
          <DialogTitle className="text-start text-[13px] font-medium text-brand-light">
            Export
          </DialogTitle>
        </DialogHeader>
        <div className="px-5 pt-4 pb-3 flex flex-col gap-4">
          {!isExporting ? (
            <>
              <div className="grid grid-cols-1 gap-3">
                <div className="flex flex-col gap-1">
                  <label className={fieldLabelClass}>
                    <span>Type</span>
                  </label>
                  <Select
                    value={kind}
                    onValueChange={(v) => setKind(v as ExportKind)}
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
                    >
                      <SelectValue placeholder="Select export type" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                      <SelectItem
                        value="video"
                        className="text-[11px] font-medium"
                      >
                        Video
                      </SelectItem>
                      <SelectItem
                        value="image"
                        className="text-[11px] font-medium"
                      >
                        Image
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Name</span>
                </label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="My export"
                  className="h-7.5 !text-[11.5px] rounded-[6px]"
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Path</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Where to save the export
                  </span>
                </label>
                <div className="flex items-center gap-2">
                  <Input
                    value={path}
                    onChange={(e) => setPath(e.target.value)}
                    placeholder="/path/to/export"
                    className="h-7.5 !text-[11.5px] rounded-[6px]"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                    onClick={async () => {
                      try {
                        const picked = await pickMediaPaths({
                          directory: true,
                          title: "Choose export folder",
                          defaultPath:
                            path && path.trim().length > 0
                              ? path.trim()
                              : undefined,
                        });
                        const dir =
                          Array.isArray(picked) && picked.length > 0
                            ? picked[0]
                            : null;
                        if (dir && typeof dir === "string") {
                          setPath(dir);
                        }
                      } catch {
                        // Swallow errors; keep existing path untouched
                      }
                    }}
                  >
                    Browse
                  </Button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="flex flex-col gap-1">
                  <label className={fieldLabelClass}>
                    <span>Resolution</span>
                  </label>
                  <Select
                    value={resolution.toString()}
                    onValueChange={(value) => setResolution(Number(value))}
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
                    >
                      <SelectValue placeholder="Select resolution" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                      {RESOLUTION_OPTIONS.map((opt) => (
                        <SelectItem
                          key={opt.value}
                          value={opt.value.toString()}
                          className="text-[11px] font-medium"
                        >
                          {opt.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {kind === "video" ? (
                  <div className="flex flex-col gap-1">
                    <label className={fieldLabelClass}>
                      <span>Bitrate</span>
                    </label>
                    <Select
                      value={bitrate}
                      onValueChange={(value) => setBitrate(value)}
                    >
                      <SelectTrigger
                        size="sm"
                        className="w-full !h-7.5 text-[11px] bg-brand-background/70 dark rounded-[6px]"
                      >
                        <SelectValue placeholder="Select bitrate" />
                      </SelectTrigger>
                      <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                        {BITRATE_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                            className="text-[11px] font-medium"
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ) : (
                  <div className="flex flex-col gap-1">
                    <label className={fieldLabelClass}>
                      <span>Format</span>
                    </label>
                    <Select
                      value={imageFormat}
                      onValueChange={(value) =>
                        setImageFormat(value as ImageFormatOption["value"])
                      }
                    >
                      <SelectTrigger
                        size="sm"
                        className="w-full !h-7.5 text-[11px] bg-brand-background/70 dark rounded-[6px]"
                      >
                        <SelectValue placeholder="Select format" />
                      </SelectTrigger>
                      <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                        {IMAGE_FORMAT_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                            className="text-[11px] font-medium"
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              {kind === "video" && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1">
                    <label className={fieldLabelClass}>
                      <span>Codec</span>
                    </label>
                    <Select
                      value={codec}
                      onValueChange={(value) =>
                        setCodec(value as CodecOption["value"])
                      }
                    >
                      <SelectTrigger
                        size="sm"
                        className="w-full !h-7.5 text-[11px] bg-brand-background/70 dark rounded-[6px]"
                      >
                        <SelectValue placeholder="Select codec" />
                      </SelectTrigger>
                      <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                        {CODEC_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                            className="text-[11px] font-medium"
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className={fieldLabelClass}>
                      <span>Format</span>
                    </label>
                    <Select
                      value={format}
                      onValueChange={(value) =>
                        setFormat(value as FormatOption["value"])
                      }
                    >
                      <SelectTrigger
                        size="sm"
                        className="w-full !h-7.5 text-[11px] bg-brand-background/70 dark rounded-[6px]"
                      >
                        <SelectValue placeholder="Select format" />
                      </SelectTrigger>
                      <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                        {FORMAT_OPTIONS.map((opt) => (
                          <SelectItem
                            key={opt.value}
                            value={opt.value}
                            className="text-[11px] font-medium"
                          >
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {kind === "image" && (
                <div className="flex flex-col gap-2">
                  <label className={fieldLabelClass}>
                    <span>Frame</span>
                    <span className="text-[10px] text-brand-light/60 font-normal">
                      Use the timeline playhead or pick a frame
                    </span>
                  </label>

                  <div className="flex items-center gap-2">
                    <div className="flex-1 min-w-0">
                      <NumberInput
                        value={String(safeImageFrame)}
                        onChange={(v) => {
                          const n = Number(v);
                          const next = Number.isFinite(n) ? n : 0;
                          setImageFrame(Math.max(0, Math.min(maxSelectableFrame, Math.round(next))));
                        }}
                        min={0}
                        max={maxSelectableFrame}
                        step={1}
                        className="h-7.5 rounded-l-[6px]"
                      />
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() => {
                        setImageFrame(
                          Math.max(0, Math.min(maxSelectableFrame, Math.round(playheadFrame))),
                        );
                      }}
                    >
                      Use playhead
                    </Button>
                  </div>

                  <div className="pt-3">
                    <Slider
                      value={[safeImageFrame]}
                      min={0}
                      max={maxSelectableFrame}
                      step={1}
                      onValueChange={(v) => {
                        const next = Array.isArray(v) ? v[0] : 0;
                        setImageFrame(Math.max(0, Math.round(next)));
                      }}
                      className="w-full"
                    />
                    <div className="mt-1 text-[10px] text-brand-light/60 flex items-center justify-between">
                      <span>
                        0
                      </span>
                      <span>
                        {maxSelectableFrame}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-1 flex flex-col gap-1.5">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="preserve-alpha"
                    checked={preserveAlpha}
                    disabled={kind === "image" && imageFormat === "jpeg"}
                    onCheckedChange={(checked) =>
                      setPreserveAlpha(Boolean(checked))
                    }
                    className="mt-0.5"
                  />
                  <label
                    htmlFor="preserve-alpha"
                    className={cn(
                      "text-brand-light text-[11px] font-medium cursor-pointer",
                      kind === "image" && imageFormat === "jpeg" && "opacity-50",
                    )}
                  >
                    Preserve Alpha
                  </label>
                </div>
              </div>

              {kind === "video" && (
                <div className="mt-1 flex flex-col gap-1.5">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="include-audio"
                      checked={includeAudio}
                      disabled={!hasAnyAudio}
                      onCheckedChange={(checked) =>
                        setIncludeAudio(Boolean(checked))
                      }
                      className={cn(
                        "mt-0.5",
                        !hasAnyAudio && "opacity-40 cursor-not-allowed",
                      )}
                    />
                    <label
                      htmlFor="include-audio"
                      className={cn(
                        "text-brand-light text-[11px] font-medium cursor-pointer",
                        !hasAnyAudio && "text-brand-light/40 cursor-not-allowed",
                      )}
                    >
                      Include audio in export
                    </label>
                  </div>
                  {includeAudio && (
                    <div className="pl-5 flex flex-col gap-1">
                      <label className={cn(fieldLabelClass, "mb-1")}>
                        <span>Audio format</span>
                      </label>
                      <Select
                        value={audioFormat}
                        onValueChange={(value) =>
                          setAudioFormat(value as AudioFormatOption["value"])
                        }
                      >
                        <SelectTrigger
                          size="sm"
                          className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
                        >
                          <SelectValue placeholder="Select audio format" />
                        </SelectTrigger>
                        <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                          {AUDIO_FORMAT_OPTIONS.map((opt) => (
                            <SelectItem
                              key={opt.value}
                              value={opt.value}
                              className="text-[11px] font-medium"
                            >
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center justify-center gap-3 ">
              <div className="flex flex-col gap-1 w-full">
                <span className="text-[12px] font-medium text-brand-light/90 text-start">
                  {kind === "image" ? "Exporting your image…" : "Exporting your video…"}
                </span>
                <span className="text-[10px] text-brand-light/60 text-start">
                  You can close this dialog. The export will continue in the
                  background.
                </span>
              </div>
              <div className="w-full">
                <ProgressBar
                  percent={safeProgressPercent}
                  className="h-2.5 border-brand-light/20 bg-brand-background/80"
                  barClassName="bg-brand-accent"
                />
                <div className="mt-2 text-[10px] text-brand-light/75 flex items-center justify-between">
                  <div>{etaLabel ? `${etaLabel} remaining` : ""}</div>
                  <div>{safeProgressPercent}%</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-brand-light/10 bg-brand-background flex items-center justify-between">
          {!isExporting && kind === "video" && (
            <div className="text-[10px] text-brand-light/60 flex items-center gap-1">
              <TbMovie className="w-4 h-4" />
              <span className="font-normal">Duration:</span>
              <span className="font-normal">{durationLabel}</span>
            </div>
          )}
          {!isExporting && kind === "image" && (
            <div className="text-[10px] text-brand-light/60 flex items-center gap-2">
              <span className="font-normal">Frame:</span>
              <span className="font-normal">{safeImageFrame}</span>
              <span className="font-normal text-brand-light/40">
                (
                {(() => {
                  const fps = Math.max(1, durationFps ?? playheadFps ?? 24);
                  const sec = safeImageFrame / fps;
                  return `${sec.toFixed(2)}s`;
                })()}
                )
              </span>
            </div>
          )}
          <div
            className={cn(
              "flex items-center gap-2 ",
              isExporting && "justify-end w-full",
            )}
          >
            <Button
              type="button"
              variant="ghost"
              className="h-7 px-4 text-[11px] font-medium bg-brand-light/5 hover:bg-brand-light/10 rounded-[6px]"
              onClick={() => onOpenChange(false)}
            >
              {isExporting ? "Close" : "Cancel"}
            </Button>
            {!isExporting && (
              <Button
                type="button"
                className="h-7 px-5 bg-brand-accent  disabled:cursor-not-allowed hover:bg-brand-accent-two-shade text-white text-[11px] font-medium rounded-[6px] border border-brand-accent-two-shade"
                onClick={handleExportClick}
              >
                Export
              </Button>
            )}
            {isExporting && onCancelExport && (
              <Button
                type="button"
                variant="destructive"
                className="h-7 px-4 text-[11px] font-medium rounded-[6px] bg-red-500/50 hover:bg-red-500/60 border border-red-500/30"
                onClick={() => {
                  try {
                    setTryCancelExport(true);
                    onCancelExport();
                  } catch {
                    // ignore errors from user-supplied cancel handler
                  }
                }}
              >
                <div className="flex flex-row items-center gap-x-1">
                  {!tryCancelExport ? (
                    <TbCancel className="w-3.5! !h-3.5" />
                  ) : (
                    <LuLoaderCircle className="w-3.5! !h-3.5 animate-spin" />
                  )}
                  <div>Cancel</div>
                </div>
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ExportModal;

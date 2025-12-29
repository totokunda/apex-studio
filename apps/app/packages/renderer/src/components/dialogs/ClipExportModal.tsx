import React, { useState, useMemo } from "react";
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
import { ProgressBar } from "@/components/common/ProgressBar";
import { pickMediaPaths, resolvePath } from "@app/preload";

type ClipExportKind = "audio" | "image" | "video";

type ResolutionOption = {
  label: string;
  value: number;
};

type ImageFormat = "png" | "jpeg" | "webp";
type VideoFormat = "mp4" | "mov" | "webm" | "mkv";
type AudioFormat = "wav" | "mp3";

export interface ClipExportSettings {
  kind: ClipExportKind;
  name: string;
  path: string;
  resolution?: number;
  imageFormat?: ImageFormat;
  videoFormat?: VideoFormat;
  audioFormat?: AudioFormat;
}

interface ClipExportModalProps {
  open: boolean;
  kind: ClipExportKind;
  defaultName?: string;
  defaultPath?: string;
  isExporting?: boolean;
  exportProgress?: number | null;
  onOpenChange: (open: boolean) => void;
  onExport: (settings: ClipExportSettings) => void;
}

const RESOLUTION_OPTIONS: ResolutionOption[] = [
  { label: "480p", value: 480 },
  { label: "720p", value: 720 },
  { label: "1080p", value: 1080 },
  { label: "2K", value: 2048 },
  { label: "4K", value: 4096 },
];

const IMAGE_FORMAT_OPTIONS: { label: string; value: ImageFormat }[] = [
  { label: "PNG", value: "png" },
  { label: "JPEG", value: "jpeg" },
  { label: "WebP", value: "webp" },
];

const VIDEO_FORMAT_OPTIONS: { label: string; value: VideoFormat }[] = [
  { label: "MP4", value: "mp4" },
  { label: "MOV", value: "mov" },
  { label: "WEBM", value: "webm" },
  { label: "MKV", value: "mkv" },
];

const AUDIO_FORMAT_OPTIONS: { label: string; value: AudioFormat }[] = [
  { label: "WAV", value: "wav" },
  { label: "MP3", value: "mp3" },
];

const fieldLabelClass =
  "text-brand-light text-[10.5px] font-medium mb-1.5 flex items-center justify-between";

export const ClipExportModal: React.FC<ClipExportModalProps> = ({
  open,
  kind,
  defaultName = "export",
  defaultPath = "~/Downloads",
  isExporting = false,
  exportProgress = null,
  onOpenChange,
  onExport,
}) => {
  const [name, setName] = useState<string>(defaultName);
  const [path, setPath] = useState<string>(defaultPath);
  const [resolution, setResolution] = useState<number>(
    RESOLUTION_OPTIONS[2]?.value ?? 1080,
  );
  const [imageFormat, setImageFormat] = useState<ImageFormat>("png");
  const [videoFormat, setVideoFormat] = useState<VideoFormat>("mp4");
  const [audioFormat, setAudioFormat] = useState<AudioFormat>("mp3");

  const title = useMemo(() => {
    if (kind === "audio") return "Export Audio Clip";
    if (kind === "image") return "Export Image Clip";
    return "Export Video Clip";
  }, [kind]);

  const safeProgressPercent = useMemo(() => {
    const raw = typeof exportProgress === "number" ? exportProgress : 0;
    const clamped = Math.max(0, Math.min(1, raw));
    return Math.round(clamped * 100);
  }, [exportProgress]);

  const handleBrowse = async () => {
    try {
      const picked = await pickMediaPaths({
        directory: true,
        title: "Choose export folder",
      });
      const dir = Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
      if (dir && typeof dir === "string") {
        setPath(dir);
      }
    } catch {
      // ignore errors and keep existing path
    }
  };

  const handleExportClick = () => {
    if (isExporting) return;
    const trimmedName = name.trim();
    const trimmedPath = path.trim();
    if (!trimmedName || !trimmedPath) return;

    onExport({
      kind,
      name: trimmedName,
      path: resolvePath(trimmedPath),
      resolution: kind === "audio" ? undefined : resolution,
      imageFormat: kind === "image" ? imageFormat : undefined,
      videoFormat: kind === "video" ? videoFormat : undefined,
      audioFormat: kind === "audio" ? audioFormat : undefined,
    });
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      onOpenChange(false);
    } else {
      onOpenChange(true);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-sm w-full bg-brand-background/90 backdrop-blur-sm text-foreground p-0 gap-0 overflow-hidden dark font-poppins border border-brand-light/10">
        <DialogHeader className="px-5 py-3.5 border-b border-brand-light/10 flex-shrink-0 bg-brand-background">
          <DialogTitle className="text-start text-[13px] font-medium text-brand-light">
            {title}
          </DialogTitle>
        </DialogHeader>
        <div className="px-5 pt-4 pb-3 flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label className={fieldLabelClass}>
              <span>Name</span>
            </label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My export"
              className="h-7.5 !text-[11.5px] rounded-[6px]"
              disabled={isExporting}
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
                disabled={isExporting}
              />
              <Button
                type="button"
                variant="outline"
                className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                onClick={handleBrowse}
                disabled={isExporting}
              >
                Browse
              </Button>
            </div>
          </div>

          {(kind === "image" || kind === "video") && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Resolution</span>
                </label>
                <Select
                  value={resolution.toString()}
                  onValueChange={(v) => setResolution(Number(v))}
                  disabled={isExporting}
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

              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Format</span>
                </label>
                {kind === "image" ? (
                  <Select
                    value={imageFormat}
                    onValueChange={(v) => setImageFormat(v as ImageFormat)}
                    disabled={isExporting}
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
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
                ) : (
                  <Select
                    value={videoFormat}
                    onValueChange={(v) => setVideoFormat(v as VideoFormat)}
                    disabled={isExporting}
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
                    >
                      <SelectValue placeholder="Select format" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-[101] dark">
                      {VIDEO_FORMAT_OPTIONS.map((opt) => (
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
                )}
              </div>
            </div>
          )}

          {kind === "audio" && (
            <div className="flex flex-col gap-1">
              <label className={fieldLabelClass}>
                <span>Format</span>
              </label>
              <Select
                value={audioFormat}
                onValueChange={(v) => setAudioFormat(v as AudioFormat)}
                disabled={isExporting}
              >
                <SelectTrigger
                  size="sm"
                  className="w-full !h-7.5 text-[11px] bg-brand-background/70 rounded-[6px]"
                >
                  <SelectValue placeholder="Select format" />
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

          {isExporting && (
            <div className="mt-2 flex flex-col gap-1">
              <ProgressBar
                percent={safeProgressPercent}
                className="h-2 border-brand-light/20 bg-brand-background/80"
                barClassName="bg-brand-accent"
              />
              <div className="text-[10px] text-brand-light/75 text-right">
                {safeProgressPercent}%
              </div>
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-brand-light/10 bg-brand-background flex items-center justify-end gap-2">
          <Button
            type="button"
            variant="ghost"
            className="h-7 px-4 text-[11px] font-medium bg-brand-light/5 hover:bg-brand-light/10 rounded-[6px]"
            onClick={() => onOpenChange(false)}
            disabled={isExporting}
          >
            Cancel
          </Button>
          <Button
            type="button"
            className="h-7 px-5 bg-brand-accent hover:bg-brand-accent-two-shade text-white text-[11px] font-medium rounded-[6px] border border-brand-accent-two-shade"
            onClick={handleExportClick}
            disabled={isExporting}
          >
            {isExporting ? "Exporting…" : "Export"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ClipExportModal;

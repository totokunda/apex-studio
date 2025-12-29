import React, { useRef, useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  TbDots,
  TbFolderOpen,
  TbPencil,
  TbTrash,
  TbVideo,
  TbVideoOff,
} from "react-icons/tb";
import { LuLoaderCircle } from "react-icons/lu";

interface DeleteAlertDialogProps {
  onDelete: () => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogCancel,
  AlertDialogAction,
} from "@/components/ui/alert-dialog";
import { ClipType, MediaInfo } from "@/lib/types";
import {
  generateAudioWaveformCanvas,
  generatePosterCanvas,
} from "@/lib/media/timeline";
import Draggable from "@/components/dnd/Draggable";
import { revealMediaItemInFolder } from "@app/preload";

// Module-level thumbnail caches so we don't have to regenerate waveforms/posters
// every time a sidebar remounts or the user navigates back to a menu.
const audioWaveformCache = new Map<string, CanvasImageSource | null>();
const posterCanvasCache = new Map<string, CanvasImageSource | null>();

const DeleteAlertDialog: React.FC<
  React.PropsWithChildren<DeleteAlertDialogProps>
> = ({ onDelete, open, onOpenChange }) => {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="dark bg-brand font-poppins">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-brand-light text-base">
            Delete Media
          </AlertDialogTitle>
          <AlertDialogDescription className="text-brand-light/70 py-0 text-sm">
            Are you sure you want to delete this media? This action cannot be
            undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => onOpenChange(false)}
            className="bg-brand text-brand-light cursor-pointer text-[12.5px]"
          >
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              onDelete();
              onOpenChange(false);
            }}
            className="cursor-pointer bg-brand-lighter text-[12.5px]"
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export type MediaItem = {
  name: string;
  type: ClipType;
  absPath: string;
  assetUrl: string;
  originalAssetUrl?: string;
  dateAddedMs?: number;
  mediaInfo?: MediaInfo;
  fillCanvas?: boolean;
  hasProxy?: boolean;
};

interface ItemProps {
  item: MediaItem;
  renamingItem: string | null;
  setRenamingItem: (item: string | null) => void;
  renameContainerRef: React.RefObject<HTMLDivElement | null>;
  renameInputRef: React.RefObject<HTMLInputElement | null>;
  renameValue: string;
  setRenameValue: (value: string) => void;
  commitRename: (item: string) => void;
  setDeleteItem: (item: string | null) => void;
  setDeleteAlertOpen: (open: boolean) => void;
  deleteAlertOpen: boolean;
  deleteItem: string | null;
  handleDelete: (item: string) => void;
  startRename: (item: MediaItem) => void;
  isDragging?: boolean;
  onCreateProxy?: (item: MediaItem) => void;
  onRemoveProxy?: (item: MediaItem) => void;
  isCreatingProxy?: boolean;
  showName?: boolean;
}

// Stable, memoized thumbnail component to prevent re-renders on parent resize
export const MediaThumb: React.FC<{
  item: MediaItem;
  isDragging?: boolean;
  className?: string;
}> = React.memo(({ item,  isDragging, className }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mediaInfo, setMediaInfo] = useState<MediaInfo | null>(null);

  const formatDuration = (duration: number) => {
    const hours = Math.floor(duration / 3600);
    const minutes = Math.floor((duration % 3600) / 60);
    const seconds = Math.floor(duration % 60);
    return hours > 0
      ? `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
      : `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  };

  // Draw audio waveform once per URL
  useEffect(() => {
    if (item.type !== "audio") return;
    let cancelled = false;
    (async () => {
      try {
        const el = canvasRef.current;
        if (!el) return;
        const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
        const cssWidth = el.clientWidth || 240;
        const cssHeight = Math.round((cssWidth * 9) / 16);
        el.width = cssWidth * dpr;
        el.height = cssHeight * dpr;
        let waveform = audioWaveformCache.get(item.assetUrl) ?? null;
        if (!waveform) {
          waveform = await generateAudioWaveformCanvas(
            item.assetUrl,
            el.width,
            el.height,
            { color: "#7791C4", mediaInfo: item.mediaInfo },
          );
          audioWaveformCache.set(item.assetUrl, waveform ?? null);
        }
        if (item.mediaInfo) {
          setMediaInfo(item.mediaInfo);
        }
        if (!waveform || cancelled) return;
        const ctx = el.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, el.width, el.height);
        // Scale cached waveform into the current canvas size so we can reuse
        // the same underlying bitmap across remounts.
        ctx.drawImage(waveform as CanvasImageSource, 0, 0, el.width, el.height);
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [item.assetUrl, item.type]);

  // Draw poster frame once per URL for video/image without stretching to canvas width
  useEffect(() => {
    if (item.type === "audio") return;
    let cancelled = false;
    (async () => {
      try {
        let poster = posterCanvasCache.get(item.assetUrl) ?? null;
        if (!poster) {
          poster = await generatePosterCanvas(
            item.assetUrl,
            undefined,
            undefined,
            { mediaInfo: item.mediaInfo },
          );
          posterCanvasCache.set(item.assetUrl, poster ?? null);
        }
        if (item.mediaInfo) {
          setMediaInfo(item.mediaInfo);
        }
        const canvasEl = canvasRef.current;
        if (!poster || !canvasEl) return;
        if (cancelled) return;

        const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
        const cssWidth = canvasEl.clientWidth || 240;
        const cssHeight = Math.round((cssWidth * 9) / 16);
        canvasEl.width = cssWidth * dpr;
        canvasEl.height = cssHeight * dpr;
        const ctx = canvasEl.getContext("2d");
        if (!ctx) return;
        ctx.imageSmoothingEnabled = true;
        // @ts-ignore
        ctx.imageSmoothingQuality = "high";
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        // Use the source image width, scaled only by ideal height, and center horizontally.
        const srcCanvas = poster as any;
        const srcW: number = srcCanvas.width;
        const srcH: number = srcCanvas.height;
        if (srcW > 0 && srcH > 0) {
          if (item.fillCanvas) {
            // Fill the entire canvas
            ctx.drawImage(
              poster as CanvasImageSource,
              0,
              0,
              canvasEl.width,
              canvasEl.height,
            );
          } else {
            const scale = canvasEl.height / srcH; // scale to match ideal height only (device pixels)
            const drawW = Math.max(1, Math.floor(srcW * scale));
            const drawH = Math.max(1, Math.floor(srcH * scale));
            const drawX = Math.floor((canvasEl.width - drawW) / 2);
            const drawY = 0;
            ctx.drawImage(
              poster as CanvasImageSource,
              drawX,
              drawY,
              drawW,
              drawH,
            );
          }
        }
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [item.assetUrl, item.type]);

  return (
    <span
      className={cn(
        "flex flex-row gap-x-2.5 items-center justify-center relative overflow-hidden rounded-md",
        className,
      )}
    >
      <canvas ref={canvasRef} className="w-full h-full block" />
      {mediaInfo?.duration && (
        <span className="text-[10px] text-brand-light/90 absolute bottom-1 left-1 p-1 rounded bg-brand-background-dark/60 z-10">
          {formatDuration(mediaInfo.duration)}
        </span>
      )}
      {isDragging && mediaInfo && !mediaInfo?.duration && (
        <span className="text-[10px] text-brand-light/90 absolute bottom-1 left-1 p-1 rounded bg-brand-background-dark/60 z-10">
          {formatDuration(5)}
        </span>
      )}
    </span>
  );
});

const Item: React.FC<ItemProps> = ({
  item,
  renamingItem,
  setRenamingItem,
  renameContainerRef,
  renameInputRef,
  renameValue,
  setRenameValue,
  commitRename,
  setDeleteItem,
  setDeleteAlertOpen,
  deleteAlertOpen,
  deleteItem,
  handleDelete,
  startRename,
  onCreateProxy,
  onRemoveProxy,
  isCreatingProxy,
  showName = true,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <Draggable
      id={item.name}
      data={item}
      disabled={
        isOpen ||
        renamingItem === item.name ||
        deleteAlertOpen ||
        isCreatingProxy
      }
    >
      <div
        key={item.name}
        className={cn(
          "group rounded-md transition-colors p-2.5  hover:bg-brand-light/5 ",
          {
            "bg-brand-light/5": renamingItem === item.name,
          },
        )}
      >
        <div className="relative">
          <div
            className={cn(
              "w-full aspect-video overflow-hidden  rounded-md bg-brand relative",
              {
                "opacity-50": isCreatingProxy,
                "cursor-pointer": !isCreatingProxy,
              },
            )}
          >
            <MediaThumb item={item} />
            {item.hasProxy && !isCreatingProxy && (
              <span className="absolute bottom-1 right-1 text-[8.5px] bg-brand-background/90 text-brand-light/90 px-1.5 py-1 rounded border border-brand-light/10 shadow">
                Proxy
              </span>
            )}
            {isCreatingProxy && (
              <div className="absolute inset-0 flex items-center justify-center bg-brand-background/60 backdrop-blur-sm rounded-md">
                <LuLoaderCircle className="w-8 h-8 text-brand-light/90 animate-spin" />
              </div>
            )}
          </div>
          <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
            <div
              className={cn(
                "absolute top-2 right-2 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity",
                {
                  "pointer-events-none": isCreatingProxy,
                },
              )}
              title="Actions"
            >
              <DropdownMenuTrigger
                disabled={isCreatingProxy}
                className="bg-black/50 cursor-pointer hover:bg-black/70 duration-300 text-white p-1 rounded disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <TbDots className="w-4 h-4" />
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="start"
                className="dark w-40 flex flex-col text-brand-light bg-brand-background font-poppins"
              >
                {item.type === "video" && (
                  <>
                    {!item.hasProxy ? (
                      <DropdownMenuItem
                        className="py-1 rounded"
                        onClick={() => onCreateProxy?.(item)}
                      >
                        <TbVideo className="w-3.5 h-3.5" />
                        <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                          Create File Proxy
                        </span>
                      </DropdownMenuItem>
                    ) : (
                      <DropdownMenuItem
                        className="py-1 rounded"
                        onClick={() => onRemoveProxy?.(item)}
                      >
                        <TbVideoOff className="w-3.5 h-3.5" />
                        <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                          Remove File Proxy
                        </span>
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem
                  className="py-1 rounded"
                  onClick={() => startRename(item)}
                >
                  <TbPencil className="w-3.5 h-3.5" />
                  <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                    Rename
                  </span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="py-1 rounded"
                  onClick={() => {
                    void revealMediaItemInFolder(item.name);
                  }}
                >
                  <TbFolderOpen className="w-3.5 h-3.5" />
                  <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                    Open File Location
                  </span>
                </DropdownMenuItem>

                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="py-1 rounded"
                  onClick={() => {
                    setDeleteItem(item.name);
                    setDeleteAlertOpen(true);
                  }}
                >
                  <TbTrash className="w-3.5 h-3.5" />
                  <span className="flex flex-row gap-x-2.5 items-center justify-center text-[11px]">
                    Delete
                  </span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </div>
          </DropdownMenu>
        </div>
        {renamingItem === item.name ? (
          <div ref={renameContainerRef} className="px-0 pt-2">
            <Input
              ref={renameInputRef}
              autoFocus
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitRename(item.name);
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setRenamingItem(null);
                }
              }}
              className="h-6 !text-[10.5px] text-brand-light/90 px-2 py-0.5 rounded"
            />
          </div>
        ) : (
          <div className="px-0 pt-2 text-[10.5px] cursor-pointer text-brand-light/90 text-start truncate">
            {showName ? item.name : ""}
          </div>
        )}
      </div>
      <DeleteAlertDialog
        open={deleteAlertOpen}
        onOpenChange={setDeleteAlertOpen}
        onDelete={() => deleteItem && handleDelete(deleteItem)}
      />
    </Draggable>
  );
};

export default Item;

import { generateTimelineSamples } from "@/lib/media/timeline";
import { AudioClipProps, MediaInfo } from "@/lib/types";
import { useClipStore } from "@/lib/clip";

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
  const speed = Math.max(
    0.1,
    Math.min(5, Number((currentClip as any)?.speed ?? 1)),
  );

  const width = mediaInfoRef?.stats.audio?.averagePacketRate ?? 1;
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

  const samples = await generateTimelineSamples(
    currentClipId,
    asset.path,
    [0],
    width,
    height,
    tClipWidth,
    {
      mediaInfo: mediaInfoRef ?? undefined,
      startFrame: visibleStartFrame - timelineShift,
      endFrame: visibleEndFrame - timelineShift,
      volume: (currentClip as any)?.volume,
      fadeIn: (currentClip as any)?.fadeIn,
      fadeOut: (currentClip as any)?.fadeOut,
    },
  );

  if (samples?.[0]?.canvas) {
    const inputCanvas = samples?.[0]?.canvas as HTMLCanvasElement;
    let offset = Math.max(
      0,
      imageCanvas.width - tClipWidth - positionOffsetStart,
    );
    const ctx = imageCanvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
      ctx.drawImage(inputCanvas, offset, 0);
    }
    // Ensure Konva layer updates immediately after drawing audio waveform
    groupRef.current?.getLayer()?.batchDraw();
  }

  //moveClipToEnd(currentClipId);
};

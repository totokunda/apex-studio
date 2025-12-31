import {
  TbPlayerSkipBack,
  TbRewindBackward5,
  TbLadder,
  TbPlus,
  TbMinus,
} from "react-icons/tb";
import { TbRewindForward5 } from "react-icons/tb";
import { FaCirclePause, FaCirclePlay } from "react-icons/fa6";
import { FiTrash } from "react-icons/fi";
import { CgMergeVertical } from "react-icons/cg";
import { useControlsStore } from "@/lib/control";
import { useClipStore } from "@/lib/clip";
import { useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { MAX_DURATION } from "@/lib/settings";
import { LuSquareSplitVertical, LuCrop } from "react-icons/lu";
import { getMediaInfoCached } from "@/lib/media/utils";
import { PiSplitHorizontal } from "react-icons/pi";
import { MediaDialog } from "@/components/dialogs/MediaDialog";
import { useState } from "react";

const BackButton = () => {
  const { setFocusFrame, focusFrame } = useControlsStore();
  const { clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled = focusFrame === 0 || !hasClips;
  const handleBack = useCallback(() => {
    if (disabled) return;
    setFocusFrame(0);
  }, [focusFrame, hasClips, setFocusFrame]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        handleBack();
      }}
    >
      <TbPlayerSkipBack className="text-brand-light/60 h-4 w-4" />
    </div>
  );
};

const RewindBackward = () => {
  const { setFocusFrame, fps, focusFrame } = useControlsStore();
  const { clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled = focusFrame === 0 || !hasClips;
  const handleRewindBackward = useCallback(() => {
    if (disabled) return;
    const framesBack = Math.max(0, focusFrame - fps * 5);
    setFocusFrame(framesBack);
  }, [disabled, focusFrame, fps, setFocusFrame]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        handleRewindBackward();
      }}
    >
      <TbRewindBackward5 className="text-brand-light/60 h-5 w-5" />
    </div>
  );
};

const RewindForward = () => {
  const { setFocusFrame, fps, focusFrame } = useControlsStore();
  const { clipDuration, clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled = focusFrame === clipDuration || !hasClips;
  const handleRewindForward = useCallback(() => {
    if (disabled) return;
    const framesForward = Math.min(focusFrame + fps * 5, clipDuration);
    setFocusFrame(framesForward);
  }, [disabled, focusFrame, fps, clipDuration, setFocusFrame]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        handleRewindForward();
      }}
    >
      <TbRewindForward5 className="text-brand-light/60 h-5 w-5" />
    </div>
  );
};

const PauseButton = () => {
  return (
    <div
      onClick={(e) => {}}
      className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300"
    >
      <FaCirclePause className="text-brand-light/70 h-7 w-7" />
    </div>
  );
};

const PlayButton = () => {
  return (
    <div className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300">
      <FaCirclePlay className="text-brand-light/70 h-7 w-7" />
    </div>
  );
};

const ExtendTimelineButton: React.FC<{ numSeconds?: number }> = ({
  numSeconds = 15,
}) => {
  const {
    zoomLevel,
    fps,
    incrementTotalTimelineFrames,
    shiftTimelineDuration,
    setFocusFrame,
    focusFrame,
  } = useControlsStore();
  const { clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled = !hasClips || zoomLevel !== 1;
  const handleExtendTimeline = useCallback(() => {
    const increment = fps * numSeconds; // 15 second at 24fps default
    if (zoomLevel === 1) {
      incrementTotalTimelineFrames(increment);
      setFocusFrame(focusFrame + increment);
      shiftTimelineDuration(increment);
    }
  }, [
    fps,
    numSeconds,
    incrementTotalTimelineFrames,
    shiftTimelineDuration,
    focusFrame,
  ]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  relative justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={handleExtendTimeline}
    >
      <TbLadder className="h-6 w-6 rotate-90 text-brand-light" />
      <TbPlus className="absolute top-1.5 -right-2 h-1.5 w-1.5 text-brand-light stroke-3" />
    </div>
  );
};

const ReduceTimelineButton: React.FC<{ numSeconds?: number }> = ({
  numSeconds = 15,
}) => {
  const {
    timelineDuration,
    zoomLevel,
    fps,
    decrementTotalTimelineFrames,
    totalTimelineFrames,
    shiftTimelineDuration,
    setFocusFrame,
    focusFrame,
  } = useControlsStore();
  const { clipDuration, clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled =
    totalTimelineFrames <= MAX_DURATION ||
    totalTimelineFrames - fps * numSeconds < clipDuration ||
    zoomLevel !== 1 ||
    !hasClips;

  const handleReduceTimeline = useCallback(() => {
    if (disabled) return;
    const decrement = fps * numSeconds; // 15 second at 24fps default
    if (zoomLevel === 1) {
      const timelineEndFrame = timelineDuration[1];
      decrementTotalTimelineFrames(decrement);
      setFocusFrame(focusFrame - decrement);
      if (timelineEndFrame >= totalTimelineFrames - decrement) {
        shiftTimelineDuration(totalTimelineFrames - timelineEndFrame);
        // ensure focus frame is within the new timeline duration
      }
    }
  }, [
    zoomLevel,
    fps,
    numSeconds,
    timelineDuration,
    decrementTotalTimelineFrames,
    clipDuration,
    hasClips,
    focusFrame,
  ]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  relative justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        {
          "opacity-30 cursor-not-allowed hover:opacity-30": disabled,
          "opacity-60 hover:opacity-100": !disabled,
        },
      )}
      onClick={handleReduceTimeline}
    >
      <TbLadder className="h-6 w-6 rotate-90 text-brand-light" />
      <TbMinus className="absolute top-1.5 -left-2 h-1.5 w-1.5 text-brand-light stroke-3" />
    </div>
  );
};

const SplitButton = () => {
  const { splitClip, clipWithinFrame, getClipById } = useClipStore();
  const { focusFrame, setFocusFrame } = useControlsStore();
  const { clips, clipDuration } = useClipStore();
  const hasClips = clips.length > 0;

  const selectedClipIds = useControlsStore((state) => state.selectedClipIds);

  const disabled = useMemo(
    () =>
      !hasClips ||
      focusFrame === 0 ||
      selectedClipIds.length === 0 ||
      focusFrame >= clipDuration ||
      selectedClipIds.some((clipId) => {
        const clip = getClipById(clipId);
        // check if clip has running preprocessors
        let runningPreprocessors = false;
        if (clip?.type === "model") return true;
        if (
          clip &&
          (clip.type === "video" || clip.type === "image") &&
          clip.preprocessors &&
          clip.preprocessors.length > 0
        ) {
          runningPreprocessors = clip.preprocessors.some(
            (preprocessor) => preprocessor.status === "running",
          );
        }
        return (
          !clip || !clipWithinFrame(clip, focusFrame) || runningPreprocessors
        );
      }),
    [
      hasClips,
      focusFrame,
      selectedClipIds,
      clipDuration,
      clipWithinFrame,
      getClipById,
    ],
  );

  const handleSplit = useCallback(() => {
    if (disabled) return;
    selectedClipIds.forEach((clipId) => {
      const clip = getClipById(clipId);
      if (clip) {
        splitClip(focusFrame, clipId);
        setFocusFrame(focusFrame + 1);
      }
    });
  }, [disabled, splitClip, focusFrame, selectedClipIds]);
  return (
    <div
      className={cn(
        "flex shrink-0 transform-gpu items-center cursor-pointer gap-x-2 py-4 justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={handleSplit}
    >
      <PiSplitHorizontal className=" text-brand-light h-4 w-4" />
    </div>
  );
};

const TrashButton = () => {
  const { selectedClipIds, clearSelection } = useControlsStore();
  const { selectedPreprocessorId } = useClipStore();
  const {
    removeClip,
    clips,
    removePreprocessorFromClip,
    getClipFromPreprocessorId,
  } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled =
    (selectedClipIds.length === 0 && selectedPreprocessorId === null) ||
    !hasClips;
  const handleDelete = useCallback(() => {
    if (disabled) return;
    if (selectedClipIds.length === 0 && selectedPreprocessorId === null) return;
    if (selectedClipIds.length > 0) {
      selectedClipIds.forEach((clipId) => {
        removeClip(clipId);
      });
    }
    if (selectedPreprocessorId !== null) {
      const clip = getClipFromPreprocessorId(selectedPreprocessorId);
      removePreprocessorFromClip(clip?.clipId ?? "", selectedPreprocessorId);
    }
    clearSelection();
  }, [disabled, selectedClipIds, removeClip, clearSelection]);

  return (
    <div
      className={cn(
        "flex shrink-0 transform-gpu items-center cursor-pointer  gap-x-2 py-4 justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={handleDelete}
    >
      <FiTrash className=" text-brand-light h-4 w-4" />
    </div>
  );
};

const MergeButton = () => {
  const { selectedClipIds, clearSelection } = useControlsStore();
  const { mergeClips, clips } = useClipStore();
  const hasClips = clips.length > 0;
  const disabled = selectedClipIds.length < 2 || !hasClips;

  const handleMerge = useCallback(() => {
    if (disabled) return;
    if (selectedClipIds.length < 2) return;
    mergeClips(selectedClipIds);
    clearSelection();
  }, [selectedClipIds, mergeClips, clearSelection]);

  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        handleMerge();
      }}
    >
      <CgMergeVertical className="text-brand-light h-5 w-5" />
    </div>
  );
};

const PlayPauseButton = () => {
  const { play, pause, isPlaying } = useControlsStore();
  return (
    <div
      onClick={() => {
        if (isPlaying) {
          pause();
        } else {
          play();
        }
      }}
      className="flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300"
    >
      {isPlaying ? <PauseButton /> : <PlayButton />}
    </div>
  );
};

const SeparateButton = () => {
  const { separateClip } = useClipStore();
  const selectedClipIds = useControlsStore((state) => state.selectedClipIds);
  const getClipById = useClipStore((state) => state.getClipById);

  const selectedClip = useMemo(() => {
    if (selectedClipIds.length !== 1) return null;
    return getClipById(selectedClipIds[0]);
  }, [selectedClipIds, getClipById]);
  const hasAudio = useMemo(() => {
    if (selectedClip?.type === "video") {
      const mediaInfo = getMediaInfoCached(selectedClip.assetId);
      return mediaInfo?.audio !== null;
    }
    return false;
  }, [selectedClip]);
  const disabled = !selectedClip || selectedClip.type !== "video" || !hasAudio;
  const handleSeparate = useCallback(() => {
    if (disabled) return;
    separateClip(selectedClip.clipId);
  }, [disabled, separateClip, selectedClip]);
  return (
    <div
      className={cn(
        "flex items-center cursor-pointer  justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
        disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
      )}
      onClick={handleSeparate}
    >
      <LuSquareSplitVertical className=" text-brand-light h-4 w-4" />
    </div>
  );
};

const CropButton = () => {
  const selectedClipIds = useControlsStore((state) => state.selectedClipIds);
  const getClipById = useClipStore((state) => state.getClipById);

  const selectedClip = useMemo(() => {
    if (selectedClipIds.length !== 1) return null;
    return getClipById(selectedClipIds[0]);
  }, [selectedClipIds, getClipById]);

  const disabled =
    !selectedClip ||
    (selectedClip.type !== "video" && selectedClip.type !== "image");
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [focusFrame, setFocusFrame] = useState(0);

  const handleOpen = () => {
    if (disabled) return;
    setIsDialogOpen(true);
  };

  const { setClipTransform } = useClipStore();

  const handleConfirm = (data: {
    rotation: number;
    aspectRatio: string;
    crop?: { x: number; y: number; width: number; height: number };
    transformWidth?: number;
    transformHeight?: number;
    transformX?: number;
    transformY?: number;
  }) => {
    if (!selectedClip) return;
    setClipTransform(selectedClip.clipId, {
      crop: data.crop,
      width: data.transformWidth,
      height: data.transformHeight,
      x: data.transformX,
      y: data.transformY,
    }, true);
  };

  return (
    <>
      <div
        onClick={handleOpen}
        className={cn(
          "flex shrink-0 transform-gpu items-center cursor-pointer gap-x-2 py-4 justify-center opacity-60 hover:opacity-100 transition-opacity duration-300",
          disabled && "opacity-30 cursor-not-allowed hover:opacity-30",
        )}
      >
        <LuCrop className=" text-brand-light h-4 w-4" />
      </div>
      <MediaDialog
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        onConfirm={handleConfirm}
        focusFrame={focusFrame}
        setFocusFrame={setFocusFrame}
      />
    </>
  );
};

export {
  BackButton,
  RewindBackward,
  RewindForward,
  PauseButton,
  PlayButton,
  SplitButton,
  TrashButton,
  MergeButton,
  PlayPauseButton,
  ExtendTimelineButton,
  ReduceTimelineButton,
  SeparateButton,
  CropButton,
};

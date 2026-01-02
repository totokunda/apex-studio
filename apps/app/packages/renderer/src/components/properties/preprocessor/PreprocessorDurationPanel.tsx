import React, { useMemo } from "react";
import { useClipStore } from "@/lib/clip";
import Input from "../Input";
import { validatePreprocessorFrames } from "@/lib/preprocessorHelpers";
import { PreprocessorClipType } from "@/lib/types";

interface PreprocessorDurationPanelProps {
  preprocessorId: string;
}

const PreprocessorDurationPanel: React.FC<PreprocessorDurationPanelProps> = ({
  preprocessorId,
}) => {
  const preprocessor = useClipStore((s) =>
    s.getPreprocessorById(preprocessorId),
  )!;
  const updatePreprocessor = useClipStore((s) => s.updatePreprocessor);
  const clip = useClipStore((s) =>
    s.getClipFromPreprocessorId(preprocessorId),
  )! as PreprocessorClipType;
  const { startFrame, endFrame } = preprocessor;

  const clipDuration = useMemo(() => {
    return (clip?.endFrame ?? 0) - (clip?.startFrame ?? 0);
  }, [clip]);

  const setStartFrame = (value: number) => {
    const { isValid } = validatePreprocessorFrames(
      value,
      endFrame ?? 0,
      preprocessorId,
      clip.preprocessors,
      clipDuration,
    );
    if (!isValid) {
      return;
    }
    updatePreprocessor(clip?.clipId ?? "", preprocessorId, {
      startFrame: value,
    });
  };

  const setEndFrame = (value: number) => {
    const { isValid } = validatePreprocessorFrames(
      startFrame ?? 0,
      value,
      preprocessorId,
      clip.preprocessors,
      clipDuration,
    );
    if (!isValid) {
      return;
    }
    updatePreprocessor(clip?.clipId ?? "", preprocessorId, { endFrame: value });
  };

  const startFrameMax = Math.max(
    0,
    (clip?.endFrame ?? 0) - (clip?.startFrame ?? 0) - 1,
  );
  const endFrameMax = Math.max(
    0,
    (clip?.endFrame ?? 0) - (clip?.startFrame ?? 0),
  );

  return (
    <div className="p-4 px-5">
      <div className="flex flex-row items-center justify-between mb-4">
        <h4 className="text-brand-light text-[12px] font-medium text-start">
          Duration
        </h4>
      </div>
      <div className="flex flex-col gap-y-2">
        <div className="flex flex-row gap-x-2">
          <Input
            label="Start Frame"
            value={startFrame?.toString() ?? "0"}
            onChange={(value) => setStartFrame(Number(value))}
            startLogo="F"
            canStep
            step={1}
            min={0}
            max={startFrameMax}
          />
          <Input
            label="End Frame"
            value={endFrame?.toString() ?? "0"}
            onChange={(value) => setEndFrame(Number(value))}
            startLogo="F"
            canStep
            step={1}
            min={1}
            max={endFrameMax}
          />
        </div>
      </div>
    </div>
  );
};

export default PreprocessorDurationPanel;

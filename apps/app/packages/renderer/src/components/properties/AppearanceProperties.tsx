import { useClipStore } from "@/lib/clip";
import { VideoClipProps, ImageClipProps, ShapeClipProps } from "@/lib/types";
import { useMemo } from "react";
import { useState } from "react";
import React from "react";
import { IoRefreshOutline } from "react-icons/io5";
import PropertiesSlider from "./PropertiesSlider";
import Input from "./Input";
import ColorInput from "./ColorInput";

interface AppearancePropertiesProps {
  clipId: string;
}

const AppearanceProperties: React.FC<AppearancePropertiesProps> = ({
  clipId,
}) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as
    | VideoClipProps
    | ImageClipProps
    | ShapeClipProps;
  const { updateClip } = useClipStore();
  const [spinning, setSpinning] = useState(false);
  const opacity = useMemo(
    () => clip?.transform?.opacity ?? 100,
    [clip?.transform?.opacity],
  );

  const setOpacity = (value: number) => {
    const numValue = typeof value === "number" ? value : Number(value);
    if (isNaN(numValue) || !isFinite(numValue)) return;
    const clamped = Math.max(0, Math.min(100, numValue));
    updateClip(clipId, {
      transform: { ...clip?.transform!, opacity: clamped },
    });
  };

  const canUpdateCornerRadius = useMemo(() => {
    if (clip?.type === "image" || clip?.type === "video") return true;
    if (clip?.type === "shape") {
      if (clip?.shapeType === "rectangle" || clip?.shapeType === "polygon")
        return true;
    }
    return false;
  }, [clip?.type]);

  const canUpdateColor = useMemo(() => {
    if (clip?.type === "shape") return true;
    return false;
  }, [clip?.type]);

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 px-5 min-w-0">
        <div className="flex flex-row items-center justify-between mb-4">
          <h4 className="text-brand-light text-[12px] font-medium text-start">
            Appearance
          </h4>
          <span
            onClick={() => {
              updateClip(clipId, {
                transform: {
                  ...clip?.transform!,
                  opacity: 100,
                  cornerRadius: 0,
                },
              });
            }}
            className="text-brand-light text-sm cursor-pointer"
          >
            <IoRefreshOutline
              className={spinning ? "animate-spin duration-500" : ""}
              onAnimationEnd={() => setSpinning(false)}
            />
          </span>
        </div>
        <div className="flex flex-col gap-y-2">
          <PropertiesSlider
            label="Opacity"
            value={opacity}
            onChange={setOpacity}
            suffix="%"
            min={0}
            max={100}
            step={1}
            toFixed={0}
          />

          {canUpdateCornerRadius && (
            <div className="flex flex-row gap-x-2 mb-4">
              <Input
                label="Corner Radius"
                value={
                  clip?.transform?.cornerRadius?.toFixed(0).toString() ?? "0"
                }
                onChange={(value) => {
                  const numValue = Number(value);
                  if (!Number.isFinite(numValue)) return;
                  const clamped = Math.max(0, Math.min(100, numValue));
                  updateClip(clipId, {
                    transform: {
                      ...clip?.transform!,
                      cornerRadius: clamped,
                    },
                  });
                }}
                startLogo="R"
                canStep
                step={1}
                min={0}
                max={100}
              />
            </div>
          )}

          {canUpdateColor && (
            <div className="flex flex-col gap-y-3 ">
              <h4 className="text-brand-light text-[12px] font-medium text-start">
                Color
              </h4>
              <ColorInput
                value={(clip as ShapeClipProps)?.fill ?? "#000000"}
                percentValue={(clip as ShapeClipProps)?.fillOpacity ?? 100}
                setPercentValue={(value) =>
                  updateClip(clipId, { fillOpacity: value })
                }
                onChange={(value) => updateClip(clipId, { fill: value })}
                label="Fill"
              />
              <ColorInput
                value={(clip as ShapeClipProps)?.stroke ?? "#000000"}
                percentValue={(clip as ShapeClipProps)?.strokeOpacity ?? 100}
                setPercentValue={(value) =>
                  updateClip(clipId, { strokeOpacity: value })
                }
                onChange={(value) => updateClip(clipId, { stroke: value })}
                label="Stroke"
              />
              <Input
                label="Stroke Width"
                value={(clip as ShapeClipProps)?.strokeWidth?.toString() ?? "0"}
                onChange={(value) =>
                  updateClip(clipId, { strokeWidth: Number(value) })
                }
                startLogo="W"
                canStep
                step={1}
                min={0}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AppearanceProperties;

import React from "react";
import ImageInput, { ImageSelection } from "./ImageInput";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  LuImageOff,
  LuImagePlus,
  LuTrash,
} from "react-icons/lu";

interface ImageInputListProps {
  label?: string;
  description?: string;
  inputId: string;
  value: ImageSelection[];
  onChange: (value: ImageSelection[]) => void;
  clipId: string;
  panelSize: number;
  min?: number;
  max?: number;
}

const ImageInputList: React.FC<ImageInputListProps> = ({
  label,
  description,
  inputId,
  value = [],
  onChange,
  clipId,
  panelSize,
  min: _min,
  max,
}) => {
  const handleAdd = () => {
    const canAdd = typeof max === "number" ? value.length < max : true;
    if (!canAdd) return;
    onChange([...(value || []), null]);
  };

  const handleItemChange = (index: number, next: ImageSelection) => {
    const nextList = [...(value || [])];
    nextList[index] = next;
    onChange(nextList);
  };

  const handleItemRemove = (index: number) => {
    const nextList = [...(value || [])];
    nextList.splice(index, 1);
    onChange(nextList);
  };

  const canAddMore = typeof max === "number" ? value.length < max : true;

  // Estimate when horizontal scrolling is needed based on panelSize and item widths
  const estimatedItemWidth = 192 + 12; // card width (w-48 ≈ 192px) + gap
  const itemCountForWidth = value.length + (canAddMore ? 1 : 0);
  const totalNeededWidth = itemCountForWidth * estimatedItemWidth;
  const enableScroll =
    typeof panelSize === "number" ? panelSize < totalNeededWidth : true;

  const rowJustifyClass =  "justify-start" ;
  const rowWidthClass = enableScroll ? "w-max" : "w-full";



  return (
    <div className="w-full pr-1">
      <div className="flex flex-row items-center justify-between">
        <div className=" flex flex-col items-start justify-start">
          {label && (
            <label className="text-brand-light text-[10.5px] w-full font-medium text-start">
              {label}
            </label>
          )}
          {description && (
            <span className="text-brand-light/80 text-[9.5px] w-full text-start">
              {description}
            </span>
          )}
        </div>
        <div className="">
          <span className="text-brand-light text-[10.5px] font-medium">
            {value.length} / {max ?? "∞"}
          </span>
        </div>
      </div>
      <ScrollArea
        className="w-full"
        style={{
          maxWidth:
            typeof panelSize === "number" && panelSize > 0
              ? panelSize
              : undefined,
        }}
      >
        <div
          className={`flex flex-row items-start ${rowJustifyClass} gap-x-3 py-3 ${rowWidthClass}`}
          style={{
            minWidth: enableScroll
              ? typeof panelSize === "number" && panelSize > 0
                ? panelSize
                : undefined
              : "100%",
          }}
        >
          {value.length === 0 && (
            <div className="w-48 h-48 flex flex-col items-center justify-center gap-y-2 py-3 opacity-70 bg-brand rounded-md">
              <LuImageOff className="w-6 h-6 text-brand-light" />
              <span className="text-brand-light text-[11.5px] font-medium">
                No images added
              </span>
            </div>
          )}
          {value.map((item, index) => (
            <div
              key={`${inputId}-${index}`}
              className="w-48 shrink-0 relative"
            >
              <button
                onClick={() => handleItemRemove(index)}
                className="absolute z-51 shadow-lg -top-1 -right-1 rounded-full w-5 h-5 flex items-center justify-center  bg-brand-background-dark border border-red-500 hover:bg-brand-background transition-all duration-200"
              >
                <LuTrash className="w-2.5 h-2.5 text-red-500" />
              </button>
              <ImageInput
                height={192}
                inputId={`${inputId}_${index}`}
                clipId={clipId}
                label={undefined}
                description={undefined}
                value={item}
                panelSize={panelSize}
                onChange={(next) => handleItemChange(index, next)}
              />
            </div>
          ))}
          {canAddMore && (
            <button
              type="button"
              onClick={handleAdd}
              className="w-48 h-48 flex flex-col gap-y-2 items-center shadow-md border-dashed justify-center rounded-md bg-brand-background-light/75 border  border-brand-light/15 hover:bg-brand-background-light transition-all duration-200"
            >
              <LuImagePlus className="w-6 h-6 text-brand-light" />
              <span className="text-brand-light text-[11.5px] font-medium">
                Add Image
              </span>
            </button>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ImageInputList;

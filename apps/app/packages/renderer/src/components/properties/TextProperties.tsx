import { TextClipProps } from "@/lib/types";
import React, { useState, useEffect } from "react";
import { useClipStore } from "@/lib/clip";
import { Combobox, FontItem } from "./Combobox";
import PropertiesSlider from "./PropertiesSlider";
import {
  FaAlignLeft,
  FaAlignCenter,
  FaAlignRight,
  FaBold,
  FaItalic,
  FaUnderline,
} from "react-icons/fa";
import ButtonList from "./ButtonList";
import {
  RxLetterCaseLowercase,
  RxLetterCaseUppercase,
  RxTextAlignBottom,
  RxTextAlignMiddle,
  RxTextAlignTop,
} from "react-icons/rx";
import { RxLetterCaseCapitalize } from "react-icons/rx";
import { RxTextNone } from "react-icons/rx";
import { cn } from "@/lib/utils";
import ColorInput from "./ColorInput";
import TextStrokeSection from "./TextStrokeSection";
import TextShadowSection from "./TextShadowSection";
import TextBackgroundSection from "./TextBackgroundSection";

interface TextPropertiesProps {
  clipId: string;
}

const fonts: FontItem[] = [
  { value: "Arial", label: "Arial", isDownloaded: true },
  { value: "Helvetica", label: "Helvetica", isDownloaded: true },
  { value: "Times New Roman", label: "Times New Roman", isDownloaded: true },
  { value: "Georgia", label: "Georgia", isDownloaded: true },
  { value: "Verdana", label: "Verdana", isDownloaded: true },
  { value: "Courier New", label: "Courier New", isDownloaded: true },
  { value: "Comic Sans MS", label: "Comic Sans MS", isDownloaded: true },
  { value: "Impact", label: "Impact", isDownloaded: true },
  { value: "Trebuchet MS", label: "Trebuchet MS", isDownloaded: true },
  { value: "Poppins", label: "Poppins", isPremium: true, isDownloaded: true },
  {
    value: "Montserrat",
    label: "Montserrat",
    isPremium: true,
    isDownloaded: true,
  },
  { value: "Roboto", label: "Roboto", isPremium: true, isDownloaded: true },
  {
    value: "Open Sans",
    label: "Open Sans",
    isPremium: true,
    isDownloaded: true,
  },
  { value: "Lato", label: "Lato", isPremium: true, isDownloaded: true },
  { value: "Oswald", label: "Oswald", isPremium: true, isDownloaded: true },
  { value: "Raleway", label: "Raleway", isPremium: true, isDownloaded: true },
  { value: "PT Sans", label: "PT Sans", isPremium: true, isDownloaded: true },
  {
    value: "Merriweather",
    label: "Merriweather",
    isPremium: true,
    isDownloaded: true,
  },
  {
    value: "Playfair Display",
    label: "Playfair Display",
    isPremium: true,
    isDownloaded: true,
  },
  { value: "Nunito", label: "Nunito", isPremium: true, isDownloaded: true },
];

const applyTextTransform = (text: string, textTransform: string) => {
  if (textTransform === "uppercase") {
    return text.toUpperCase();
  }
  if (textTransform === "lowercase") {
    return text.toLowerCase();
  }
  if (textTransform === "capitalize") {
    return text
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }
  return text;
};

const TextProperties: React.FC<TextPropertiesProps> = ({ clipId }) => {
  const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
  const [fontFamily, setFontFamily] = useState<string>(
    clip?.fontFamily ?? "Arial",
  );
  const updateClip = useClipStore((s) => s.updateClip);

  useEffect(() => {
    setFontFamily(clip?.fontFamily ?? "Arial");
  }, [clip?.fontFamily]);

  const AlignmentButtons = [
    { value: "left", icon: <FaAlignLeft className="h-3 w-3" /> },
    { value: "center", icon: <FaAlignCenter className="h-3 w-3" /> },
    { value: "right", icon: <FaAlignRight className="h-3 w-3" /> },
  ];

  const VerticalAlignmentButtons = [
    { value: "top", icon: <RxTextAlignTop className="h-3 w-3" /> },
    { value: "middle", icon: <RxTextAlignMiddle className="h-3 w-3" /> },
    { value: "bottom", icon: <RxTextAlignBottom className="h-3 w-3" /> },
  ];

  const TextTransformButtons = [
    { value: "none", icon: <RxTextNone className="h-3 w-3" /> },
    { value: "uppercase", icon: <RxLetterCaseUppercase className="h-3 w-3" /> },
    { value: "lowercase", icon: <RxLetterCaseLowercase className="h-3 w-3" /> },
    {
      value: "capitalize",
      icon: <RxLetterCaseCapitalize className="h-3 w-3" />,
    },
  ];

  const selectFontFamily = (font: string) => {
    setFontFamily(font);
    updateClip(clipId, { fontFamily: font });
  };

  const setFontSize = (value: number) => {
    updateClip(clipId, { fontSize: value });
  };
  const setAlignment = (value: string) => {
    updateClip(clipId, { textAlign: value as "left" | "center" | "right" });
  };

  const setVerticalAlignment = (value: string) => {
    updateClip(clipId, { verticalAlign: value as "top" | "middle" | "bottom" });
  };
  const setTextTransform = (value: string) => {
    updateClip(clipId, {
      textTransform: value as "none" | "uppercase" | "lowercase" | "capitalize",
      text: applyTextTransform(clip?.text ?? "", value),
    });
  };

  const toggleBold = () => {
    const currentWeight = clip?.fontWeight ?? 400;
    updateClip(clipId, { fontWeight: currentWeight >= 700 ? 400 : 700 });
  };

  const toggleItalic = () => {
    const currentStyle = clip?.fontStyle ?? "normal";
    updateClip(clipId, {
      fontStyle: currentStyle === "italic" ? "normal" : "italic",
    });
  };

  const toggleUnderline = () => {
    const currentDecoration = clip?.textDecoration ?? "none";
    updateClip(clipId, {
      textDecoration: currentDecoration === "underline" ? "none" : "underline",
    });
  };

  return (
    <div className="flex flex-col gap-y-2 min-w-0">
      <div className="p-4 flex flex-col gap-y-5 px-5 min-w-0">
        <h3 className="text-brand-light text-xs font-medium text-start">
          Basic
        </h3>
        <textarea
          className="bg-brand border border-brand-light/5 rounded-md p-2 h-20 text-xs text-brand-light w-full resize-none custom-scrollbar"
          value={clip?.text}
          onChange={(e) => updateClip(clipId, { text: e.target.value })}
        />
        <div className="flex flex-row gap-x-2 w-full items-center">
          <h4 className="text-brand-light text-[10.5px] text-start w-1/3">
            Font Family
          </h4>
          <Combobox
            value={fontFamily}
            onValueChange={selectFontFamily}
            fonts={fonts}
            placeholder="Select font..."
          />
        </div>
        <PropertiesSlider
          label="Font Size"
          value={clip?.fontSize ?? 16}
          onChange={setFontSize}
          suffix="px"
          min={10}
          max={256}
          step={1}
          toFixed={0}
        />
        <div className="flex flex-row gap-x-2 items-center">
          <h4 className="text-brand-light text-[10.5px] text-start w-1/3">
            Style
          </h4>
          <div className="flex flex-row bg-brand divide-x divide-brand-light/10 rounded w-full">
            <button
              onClick={toggleBold}
              className={cn(
                "flex items-center p-1.5 justify-center w-full text-brand-light/70 cursor-pointer hover:text-brand-light hover:bg-brand-light/10 duration-200 transition-all rounded-l",
                (clip?.fontWeight ?? 400) >= 700 &&
                  "bg-brand-light/10 text-brand-light",
              )}
            >
              <FaBold className="h-3 w-3" />
            </button>
            <button
              onClick={toggleItalic}
              className={cn(
                "flex items-center p-1.5 justify-center w-full text-brand-light/70 cursor-pointer hover:text-brand-light hover:bg-brand-light/10 duration-200 transition-all",
                (clip?.fontStyle ?? "normal") === "italic" &&
                  "bg-brand-light/10 text-brand-light",
              )}
            >
              <FaItalic className="h-3 w-3" />
            </button>
            <button
              onClick={toggleUnderline}
              className={cn(
                "flex items-center p-1.5 justify-center w-full text-brand-light/70 cursor-pointer hover:text-brand-light hover:bg-brand-light/10 duration-200 transition-all rounded-r",
                (clip?.textDecoration ?? "none") === "underline" &&
                  "bg-brand-light/10 text-brand-light",
              )}
            >
              <FaUnderline className="h-3 w-3" />
            </button>
          </div>
        </div>
        <div className="flex flex-row gap-x-2 items-center">
          <h4 className="text-brand-light text-[10.5px] text-start w-1/3">
            Align
          </h4>
          <ButtonList
            buttons={AlignmentButtons}
            selected={clip?.textAlign ?? "left"}
            onSelect={setAlignment}
          />
        </div>
        <div className="flex flex-row gap-x-2 items-center">
          <h4 className="text-brand-light text-[10.5px] text-start w-1/3">
            Vertical Align
          </h4>
          <ButtonList
            buttons={VerticalAlignmentButtons}
            selected={clip?.verticalAlign ?? "top"}
            onSelect={setVerticalAlignment}
          />
        </div>
        <div className="flex flex-row gap-x-2 items-center">
          <h4 className="text-brand-light text-[10.5px] text-start w-1/3">
            Text Transform
          </h4>
          <ButtonList
            buttons={TextTransformButtons}
            selected={clip?.textTransform ?? "none"}
            onSelect={setTextTransform}
          />
        </div>
        <ColorInput
          percentValue={clip?.colorOpacity ?? 100}
          value={clip?.color ?? "#000000"}
          setPercentValue={(value) =>
            updateClip(clipId, { colorOpacity: value })
          }
          onChange={(value) => updateClip(clipId, { color: value })}
          label="Color"
        />
      </div>
      <TextStrokeSection clipId={clipId} />
      <TextShadowSection clipId={clipId} />
      <TextBackgroundSection clipId={clipId} />
    </div>
  );
};

export default TextProperties;

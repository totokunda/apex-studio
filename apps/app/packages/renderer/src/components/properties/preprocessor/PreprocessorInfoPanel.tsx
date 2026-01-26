import React, { useState, useRef, useEffect } from "react";
import { PreprocessorClipProps } from "@/lib/types";
import { LuChevronDown, LuChevronRight } from "react-icons/lu";

interface PreprocessorInfoPanelProps {
  preprocessor: PreprocessorClipProps;
}

interface InfoParamDescriptionProps {
  param: any;
  isExpanded: boolean;
  isTruncated: boolean;
  onToggle: () => void;
  onTruncationDetected: (isTruncated: boolean) => void;
}

const InfoParamDescription: React.FC<InfoParamDescriptionProps> = ({
  param,
  isExpanded,
  isTruncated,
  onToggle,
  onTruncationDetected,
}) => {
  const descRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (descRef.current && param.description && !isExpanded) {
      // Only check truncation when the element is clamped (not expanded)
      const checkTruncation =
        descRef.current.scrollHeight > descRef.current.clientHeight;
      if (checkTruncation !== isTruncated) {
        onTruncationDetected(checkTruncation);
      }
    }
  }, [param.description, isTruncated, isExpanded, onTruncationDetected]);

  if (!param.description) return null;

  return (
    <div className="flex flex-col gap-y-1">
      <span
        ref={descRef}
        className={`text-brand-light/90 text-[10px] text-start ${!isExpanded ? "line-clamp-1" : ""}`}
      >
        {param.description}
      </span>
      {isTruncated && (
        <button
          onClick={onToggle}
          className="text-brand-light/50 hover:text-brand-light text-[9px] text-start transition-colors duration-200"
        >
          {isExpanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
};

const PreprocessorInfoPanel: React.FC<PreprocessorInfoPanelProps> = ({
  preprocessor,
}) => {
  const hasParameters =
    preprocessor.preprocessor.parameters &&
    preprocessor.preprocessor.parameters.length > 0;
  const [isParametersExpanded, setIsParametersExpanded] = useState(false);
  const [expandedDescriptions, setExpandedDescriptions] = useState<
    Record<string, boolean>
  >({});
  const [truncatedDescriptions, setTruncatedDescriptions] = useState<
    Record<string, boolean>
  >({});
  const [isMainDescExpanded, setIsMainDescExpanded] = useState(false);
  const [isMainDescTruncated, setIsMainDescTruncated] = useState(false);
  const mainDescRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (
      mainDescRef.current &&
      preprocessor.preprocessor.description &&
      !isMainDescExpanded
    ) {
      const checkTruncation =
        mainDescRef.current.scrollHeight > mainDescRef.current.clientHeight;
      if (checkTruncation !== isMainDescTruncated) {
        setIsMainDescTruncated(checkTruncation);
      }
    }
  }, [
    preprocessor.preprocessor.description,
    isMainDescTruncated,
    isMainDescExpanded,
  ]);

  const formatParameterName = (name: string) => {
    return name
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  };

  const formatParameterType = (type: string) => {
    if (type === "int") {
      return "Integer";
    } else if (type === "float") {
      return "Float";
    } else if (type === "bool") {
      return "Boolean";
    } else if (type === "str") {
      return "String";
    } else if (type === "category") {
      return "Category";
    } else {
      return type;
    }
  };

  return (
    <div className="p-6 flex flex-col gap-y-4">
      <div className="flex flex-col gap-y-2">
        <div className="flex flex-col gap-y-1">
          <span className="text-brand-lighter text-[14px] font-semibold text-start">
            {preprocessor.preprocessor.name}
          </span>
          <span className="text-brand-light/80 text-[11px] text-start">
            {preprocessor.preprocessor.category}
          </span>
        </div>
        <div className="flex flex-col gap-y-1">
          <span
            ref={mainDescRef}
            className={`text-brand-light text-[12px] text-start ${!isMainDescExpanded ? "line-clamp-1" : ""}`}
          >
            {preprocessor.preprocessor.description}
          </span>
          {isMainDescTruncated && (
            <button
              onClick={() => setIsMainDescExpanded(!isMainDescExpanded)}
              className="text-brand-light/50 hover:text-brand-light text-[9px] text-start transition-colors duration-200"
            >
              {isMainDescExpanded ? "Show less" : "Show more"}
            </button>
          )}
        </div>
        <div className="flex flex-row items-center gap-x-3 mt-1">
          {preprocessor.preprocessor.supports_image && (
            <div className="flex items-center gap-x-1.5 py-1 px-3 bg-brand rounded-full border border-brand-light/10">
              <span className="text-brand-light text-[11px]">📷</span>
              <span className="text-brand-light text-[10px]">Image</span>
            </div>
          )}
          {preprocessor.preprocessor.supports_video && (
            <div className="flex items-center gap-x-1.5 py-1 px-3 bg-brand rounded-full border border-brand-light/10">
              <span className="text-brand-light text-[11px]">🎬</span>
              <span className="text-brand-light text-[10px]">Video</span>
            </div>
          )}
        </div>
      </div>

      {hasParameters && (
        <div className="flex flex-col gap-y-3 border-t border-brand-light/10 pt-4">
          <button
            onClick={() => setIsParametersExpanded(!isParametersExpanded)}
            className="flex flex-row items-center gap-x-2 hover:bg-brand-light/5 -mx-2 px-2 py-1.5 rounded transition-all duration-200"
          >
            {isParametersExpanded ? (
              <LuChevronDown className="text-brand-light w-3.5 h-3.5" />
            ) : (
              <LuChevronRight className="text-brand-light w-3.5 h-3.5" />
            )}
            <h4 className="text-brand-lighter text-[12px] font-semibold text-start">
              Parameters
            </h4>
          </button>

          {isParametersExpanded && (
            <div className="flex flex-col gap-y-3">
              {preprocessor.preprocessor.parameters!.map((param, index) => (
                <div
                  key={index}
                  className="flex flex-col gap-y-2 p-3 rounded-lg bg-brand-light/5 border border-brand-light/10"
                >
                  <div className="flex flex-row items-center gap-x-2 justify-between">
                    <span className="text-brand-lighter text-[11px] font-medium">
                      {param.display_name || formatParameterName(param.name)}
                    </span>
                    <div className="flex items-center gap-x-2">
                      <span className="text-brand-light/60 text-[10px]">
                        {formatParameterType(param.type)}
                      </span>
                      {param.required && (
                        <span className="text-red-400/80 text-[9px] px-1.5 py-0.5 bg-red-400/10 rounded-full border border-red-400/20">
                          Required
                        </span>
                      )}
                    </div>
                  </div>
                  <InfoParamDescription
                    param={param}
                    isExpanded={expandedDescriptions[param.name] || false}
                    isTruncated={truncatedDescriptions[param.name] || false}
                    onToggle={() =>
                      setExpandedDescriptions((prev) => ({
                        ...prev,
                        [param.name]: !prev[param.name],
                      }))
                    }
                    onTruncationDetected={(isTruncated) => {
                      setTruncatedDescriptions((prev) => {
                        if (prev[param.name] === isTruncated) return prev;
                        return {
                          ...prev,
                          [param.name]: isTruncated,
                        };
                      });
                    }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PreprocessorInfoPanel;

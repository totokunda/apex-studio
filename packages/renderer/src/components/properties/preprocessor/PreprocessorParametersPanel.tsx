import React, { useState, useMemo, useRef, useEffect } from "react";
import { PreprocessorClipProps } from "@/lib/types";
import Input from "../Input";
import BooleanToggle from "./BooleanToggle";
import PropertiesSlider from "../PropertiesSlider";
import CategorySelector from "./CategorySelector";
import { useClipStore } from "@/lib/clip";

interface PreprocessorParametersPanelProps {
  preprocessor: PreprocessorClipProps;
}

interface ParamDescriptionProps {
  param: any;
  isExpanded: boolean;
  isTruncated: boolean;
  onToggle: () => void;
  onTruncationDetected: (isTruncated: boolean) => void;
}

const ParamDescription: React.FC<ParamDescriptionProps> = ({
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
        className={`text-brand-light/70 text-[10px] text-start ${!isExpanded ? "line-clamp-1" : ""}`}
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

const PreprocessorParametersPanel: React.FC<
  PreprocessorParametersPanelProps
> = ({ preprocessor }) => {
  const updatePreprocessor = useClipStore((s) => s.updatePreprocessor);
  const getClipFromPreprocessorId = useClipStore(
    (s) => s.getClipFromPreprocessorId,
  );
  // Subscribe directly to the preprocessor's values from the store
  const storedPreprocessor = useClipStore((s) =>
    s.getPreprocessorById(preprocessor.id),
  );

  // Get the actual clipId from the clip that contains this preprocessor
  const parentClip = getClipFromPreprocessorId(preprocessor.id);
  const actualClipId = parentClip?.clipId ?? "";

  const formatParameterName = (name: string) => {
    return name
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  };

  const getDefaultValueForType = (type: string, options?: any[]): any => {
    if (type === "int" || type === "float") return 0;
    if (type === "bool") return false;
    if (type === "str") return "";
    if (type === "category" && options && options.length > 0) {
      return options[0].value;
    }
    return "";
  };

  const initialParams = useMemo(() => {
    const params: Record<string, any> = {};
    // Use stored values from the store subscription
    const storedValues = storedPreprocessor?.values || {};

    if (preprocessor.preprocessor.parameters) {
      preprocessor.preprocessor.parameters.forEach((param) => {
        // Use stored values first, then default, then type default
        if (storedValues[param.name] !== undefined) {
          params[param.name] = storedValues[param.name];
        } else if (param.default !== undefined) {
          params[param.name] = param.default;
        } else {
          params[param.name] = getDefaultValueForType(
            param.type,
            param.options,
          );
        }
      });
    }
    return params;
  }, [
    preprocessor.preprocessor.parameters,
    JSON.stringify(storedPreprocessor?.values),
  ]);

  const [paramValues, setParamValues] =
    useState<Record<string, any>>(initialParams);
  const [expandedDescriptions, setExpandedDescriptions] = useState<
    Record<string, boolean>
  >({});
  const [truncatedDescriptions, setTruncatedDescriptions] = useState<
    Record<string, boolean>
  >({});

  // Update paramValues when values prop changes (e.g., when navigating back to tab)
  useEffect(() => {
    setParamValues(initialParams);
  }, [initialParams, storedPreprocessor?.values]);

  const handleParamChange = (paramName: string, value: any) => {
    const newValues = {
      ...paramValues,
      [paramName]: value,
    };
    setParamValues(newValues);

    // Auto-save to store immediately
    if (actualClipId) {
      updatePreprocessor(actualClipId, preprocessor.id, { values: newValues });
    }
  };

  const hasParameters =
    preprocessor.preprocessor.parameters &&
    preprocessor.preprocessor.parameters.length > 0;

  if (!hasParameters) {
    return (
      <div className="p-6 flex items-center justify-center">
        <span className="text-brand-light/60 text-[12px]">
          No inputs available
        </span>
      </div>
    );
  }

  return (
    <div className="p-6 flex flex-col gap-y-4 ">
      <div className="flex flex-col gap-y-5  rounded-md">
        {preprocessor.preprocessor.parameters!.map((param, index) => (
          <div key={index} className="flex flex-col gap-y-3 ">
            <div className="flex flex-col gap-y-1">
              <div className="flex flex-row items-center gap-x-2 justify-between">
                <span className="text-brand-lighter text-[11px] font-semibold">
                  {param.display_name || formatParameterName(param.name)}
                </span>
                <div className="flex items-center gap-x-2">
                  {param.required && (
                    <span className="text-red-400/80 text-[9px] px-1.5 py-0.5 bg-red-400/10 rounded-full border border-red-400/20">
                      Required
                    </span>
                  )}
                </div>
              </div>
              <ParamDescription
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

            <div className="mt-2">
              {param.type === "int" && (
                <PropertiesSlider
                  label=""
                  value={
                    paramValues[param.name] ?? param.default ?? param.min ?? 0
                  }
                  onChange={(value) => handleParamChange(param.name, value)}
                  min={param.min ?? 0}
                  max={param.max ?? 100}
                  step={1}
                  toFixed={0}
                />
              )}

              {param.type === "float" && (
                <PropertiesSlider
                  label=""
                  value={
                    paramValues[param.name] ?? param.default ?? param.min ?? 0
                  }
                  onChange={(value) => handleParamChange(param.name, value)}
                  min={param.min ?? 0}
                  max={param.max ?? 1}
                  step={0.01}
                  toFixed={2}
                />
              )}

              {param.type === "bool" && (
                <BooleanToggle
                  value={paramValues[param.name] ?? param.default ?? false}
                  onChange={(value) => handleParamChange(param.name, value)}
                />
              )}

              {param.type === "str" && (
                <Input
                  label=""
                  value={String(paramValues[param.name] ?? param.default ?? "")}
                  onChange={(value) => handleParamChange(param.name, value)}
                />
              )}

              {param.type === "category" &&
                param.options &&
                param.options.length > 0 && (
                  <CategorySelector
                    value={String(
                      paramValues[param.name] ??
                        param.default ??
                        param.options[0].value,
                    )}
                    onChange={(value) => handleParamChange(param.name, value)}
                    options={param.options}
                  />
                )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PreprocessorParametersPanel;

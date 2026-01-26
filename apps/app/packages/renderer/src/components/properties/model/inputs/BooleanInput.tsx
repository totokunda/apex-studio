import React, { useMemo } from "react";
import { cn } from "@/lib/utils";
import { LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface BooleanInputProps {
  label?: string;
  description?: string;
  value: boolean | string;
  defaultValue?: boolean | string;
  onChange: (value: boolean) => void;
}

const BooleanInput: React.FC<BooleanInputProps> = ({
  label,
  description,
  value,
  onChange,
}) => {
  const isOn = useMemo(() => String(value).toLowerCase() === "true", [value]);

  return (
    <div className="flex flex-row items-center w-full gap-x-2 min-w-0 relative mt-1">
      <div className="flex items-center gap-1.5">
        <label className="text-brand-light text-[10.5px] text-start font-medium">
          {label}
        </label>
        {description && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="text-brand-light/70 hover:text-brand-light focus:outline-none"
              >
                <LuInfo className="w-3 h-3" />
              </button>
            </TooltipTrigger>
            <TooltipContent
              sideOffset={6}
              className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
            >
              {description}
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      <button
        type="button"
        aria-pressed={isOn}
        aria-label={isOn ? "Disable" : "Enable"}
        onClick={() => onChange(!isOn)}
        className={cn(
          "ml-auto relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none",
          isOn
            ? "bg-blue-600"
            : "bg-brand-background border border-brand-light/10",
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 transform rounded-full bg-brand-light shadow transition-transform",
            isOn ? "translate-x-4.5" : "translate-x-0.5",
          )}
        />
      </button>
    </div>
  );
};

export default BooleanInput;

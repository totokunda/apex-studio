import React from "react";
import { cn } from "@/lib/utils";

interface BooleanToggleProps {
  value: boolean;
  onChange: (value: boolean) => void;
  label?: string;
}

const BooleanToggle: React.FC<BooleanToggleProps> = ({
  value,
  onChange,
  label,
}) => {
  return (
    <div className="flex flex-col gap-y-1.5">
      {label && (
        <span className="text-brand-light text-[10.5px] font-medium">
          {label}
        </span>
      )}
      <div className="flex flex-row gap-x-2">
        <button
          onClick={() => onChange(true)}
          className={cn(
            "flex-1 py-1.5 px-3 rounded-[6px] text-[10.5px] font-medium transition-all duration-200 border",
            value
              ? "bg-brand-background-light text-brand-lighter border-brand-light/20"
              : "bg-brand text-brand-light/60 border-brand-light/10 hover:bg-brand-light/5",
          )}
        >
          True
        </button>
        <button
          onClick={() => onChange(false)}
          className={cn(
            "flex-1 py-1.5 px-3 rounded-[6px] text-[10.5px] font-medium transition-all duration-200 border",
            !value
              ? "bg-brand-background-light text-brand-lighter border-brand-light/20"
              : "bg-brand text-brand-light/60 border-brand-light/10 hover:bg-brand-light/5",
          )}
        >
          False
        </button>
      </div>
    </div>
  );
};

export default BooleanToggle;

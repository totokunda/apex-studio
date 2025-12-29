import React from "react";
import { cn } from "@/lib/utils";

export const ProgressBar: React.FC<{
  percent: number; // 0..100
  className?: string;
  barClassName?: string;
}> = ({ percent, className, barClassName }) => {
  const safe = Math.max(
    0,
    Math.min(100, Number.isFinite(percent) ? percent : 0),
  );
  return (
    <div
      className={cn(
        "w-full h-2 bg-brand-background-dark rounded overflow-hidden border border-brand-light/10",
        className,
      )}
    >
      <div
        className={cn("h-full bg-brand-light/50 transition-all", barClassName)}
        style={{ width: `${safe}%` }}
      />
    </div>
  );
};

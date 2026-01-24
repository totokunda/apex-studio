import React from "react";

type SemiGaugeProps = {
  value: number; // 0-100
  label: string;
  size?: number; // diameter of full circle
};

function colorFor(value: number): string {
  if (value >= 85) return "#ef4444"; // red-500
  if (value >= 60) return "#f59e0b"; // amber-500
  return "#22c55e"; // green-500
}

export const SemiGauge: React.FC<SemiGaugeProps> = ({
  value,
  label,
  size = 84,
}) => {
  const clamped = Math.max(
    0,
    Math.min(100, Number.isFinite(value) ? value : 0),
  );
  const r = size / 2 - 6; // padding for stroke
  const cx = size / 2;
  const cy = size / 2;
  const arcLen = Math.PI * r; // semicircle length
  const progressLen = (clamped / 100) * arcLen;
  const stroke = 8;
  const trackColor = "rgba(255,255,255,0.12)";
  const activeColor = colorFor(clamped);

  // Path for top semicircle from left to right
  const d = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width={size} height={size / 2} viewBox={`0 0 ${size} ${size / 2}`}>
        <path
          d={d}
          fill="none"
          stroke={trackColor}
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={d}
          fill="none"
          stroke={activeColor}
          strokeWidth={stroke}
          strokeLinecap="round"
          style={{
            strokeDasharray: arcLen,
            strokeDashoffset: arcLen - progressLen,
            transition: "stroke-dashoffset 250ms ease, stroke 250ms ease",
          }}
        />
      </svg>
      <div className="-mt-2.5 text-center">
        <div className="text-[14px] font-bold text-brand-light">
          {clamped.toFixed(0)}%
        </div>
        <div className="text-[12px] text-brand-light font-semibold">
          {label}
        </div>
      </div>
    </div>
  );
};

export default SemiGauge;

import React from "react";
import ColorCorrectionProperties from "./ColorCorrectionProperties";
import EffectsProperties from "./EffectsProperties";

interface AdjustPropertiesProps {
  clipId: string;
}

const AdjustProperties: React.FC<AdjustPropertiesProps> = ({ clipId }) => {
  return (
    <div className="min-w-0 divide-y divide-brand-light/10">
      <ColorCorrectionProperties clipId={clipId} />
      <EffectsProperties clipId={clipId} />
    </div>
  );
};

export default AdjustProperties;

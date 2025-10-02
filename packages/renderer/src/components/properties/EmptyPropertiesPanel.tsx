import React from 'react'
import { CgFilm } from "react-icons/cg";

interface EmptyPropertiesPanelProps {
  
}

const EmptyPropertiesPanel:React.FC<EmptyPropertiesPanelProps> = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-y-3 opacity-70">
        <CgFilm className="text-brand-light h-7 w-7" />
        <h2 className="text-brand-light text-sm">Select a clip to view properties</h2>
    </div>
  )
}

export default EmptyPropertiesPanel
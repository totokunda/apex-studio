import React from 'react'
import { ResizablePanel } from '../ui/resizable'

interface PropertiesPanelProps { order?: number; defaultSize?: number; minSize?: number }

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ order, defaultSize = 25, minSize = 20 }) => {
  return (
    <ResizablePanel
    minSize={minSize}
    defaultSize={defaultSize}
    order={order}
    className="bg-brand-background flex flex-col  rounded-lg shadow w-full border-brand-light/10 flex-shrink-0"
  >
   <div className="bg-brand-background rounded-lg p-3">
    
   </div>
  </ResizablePanel>
  )
}

export default PropertiesPanel
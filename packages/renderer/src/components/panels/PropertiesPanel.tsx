import React from 'react'
import { ResizablePanel } from '../ui/resizable'
import EmptyPropertiesPanel from '../properties/EmptyPropertiesPanel'
import ClipPropertiesPanel from '../properties/ClipPropertiesPanel'
import { useControlsStore } from '@/lib/control'

interface PropertiesPanelProps { order?: number; defaultSize?: number; minSize?: number }

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ order, defaultSize = 25, minSize = 20 }) => {
  const { selectedClipIds } = useControlsStore();
  return (
    <ResizablePanel
    minSize={minSize}
    defaultSize={defaultSize}
    order={order}
    className="bg-brand-background flex flex-col  rounded-lg shadow w-full border-brand-light/10 flex-shrink-0"
  >

   <div className="bg-brand-background rounded-lg  h-full">
      {selectedClipIds.length > 0 ? <ClipPropertiesPanel /> : <EmptyPropertiesPanel />}
   </div>
  </ResizablePanel>
  )
}

export default PropertiesPanel
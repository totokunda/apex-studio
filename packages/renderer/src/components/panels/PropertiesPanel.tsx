import React from 'react'
import { ResizablePanel } from '../ui/resizable'

interface PropertiesPanelProps { order?: number; defaultSize?: number; minSize?: number }

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ order, defaultSize = 25, minSize = 20 }) => {
  return (
    <ResizablePanel
    minSize={minSize}
    defaultSize={defaultSize}
    order={order}
  >
   
  </ResizablePanel>
  )
}

export default PropertiesPanel
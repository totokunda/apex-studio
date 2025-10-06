
import React from 'react';
import {useDraggable} from '@dnd-kit/core';
// import {CSS} from '@dnd-kit/utilities';
import { MediaItem } from '../media/Item';


type GenericData = MediaItem

interface DraggableProps {
    id: string;
    data: GenericData;
    disabled?: boolean;
}

const Draggable:React.FC<React.PropsWithChildren<DraggableProps>> = (props) => {
  const {attributes, listeners, setNodeRef, isDragging} = useDraggable({
    id: props.id,
    data: props.data,
    disabled: props.disabled,
  });

  const style: React.CSSProperties = {
    transform: undefined,
    opacity: isDragging ? 0.5 : 1,
    transition: isDragging ? 'opacity 0.2s ease-in-out' : undefined,
  };

  return (
    <div ref={setNodeRef} {...listeners} {...attributes} style={style}>
      {props.children}
    </div>
  );
}

export default Draggable;

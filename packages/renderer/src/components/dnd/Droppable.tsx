import React, { useMemo } from "react";
import { useDroppable } from "@dnd-kit/core";

interface DroppableProps {
  id: string;
  accepts: string[];
  className?: string;
  highlight?: {
    borderColor: string;
    textColor: string;
    bgColor: string;
  };
}

const Droppable: React.FC<React.PropsWithChildren<DroppableProps>> = (
  props,
) => {
  const { setNodeRef, isOver } = useDroppable({
    id: props.id,
    data: {
      accepts: props.accepts,
    },
  });

  const style: React.CSSProperties = useMemo(
    () => ({
      border: isOver
        ? `1px solid ${props.highlight?.borderColor}`
        : "1px solid transparent",
      color: isOver
        ? (props.highlight?.textColor as React.CSSProperties["color"])
        : undefined,
      backgroundColor: isOver
        ? (props.highlight?.bgColor as React.CSSProperties["backgroundColor"])
        : undefined,
    }),
    [isOver, props.highlight],
  );

  return (
    <div ref={setNodeRef} style={style} className={props.className}>
      {props.children}
    </div>
  );
};

export default Droppable;

import { createContext, useContext } from "react";
import Konva from "konva";

export const KonvaImageContext = createContext<{
  image: HTMLCanvasElement | null;
  imageRef: React.RefObject<Konva.Image | null>;
  cornerRadius: number;
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  crop:
    | {
        x: number;
        y: number;
        width: number;
        height: number;
      }
    | undefined;
  onDragMove: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onDragStart: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onDragEnd: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  onClick: (e: Konva.KonvaEventObject<MouseEvent>) => void;
  draggable: boolean;
}>({
  image: null,
  cornerRadius: 0,
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  scaleX: 1,
  scaleY: 1,
  rotation: 0,
  crop: { x: 0, y: 0, width: 0, height: 0 },
  onDragMove: () => {},
  onDragStart: () => {},
  onDragEnd: () => {},
  onClick: () => {},
  imageRef: { current: null },
  draggable: false,
});

export const useKonvaImageContext = () => {
  return useContext(KonvaImageContext);
};

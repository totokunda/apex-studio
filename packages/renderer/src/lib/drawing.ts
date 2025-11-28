import { create } from "zustand";

export type DrawingTool = "brush" | "highlighter" | "eraser";

interface DrawingState {
  tool: DrawingTool;
  brushSize: number;
  highlighterSize: number;
  eraserSize: number;
  smoothing: number; // 0-1, where 0 is no smoothing (pen-like) and 1 is maximum smoothing (pencil-like)
  color: string;
  opacity: number;
  isDrawing: boolean;
  currentLineId: string | null;
  selectedLineId: string | null;
  setTool: (tool: DrawingTool) => void;
  setBrushSize: (size: number) => void;
  setHighlighterSize: (size: number) => void;
  setEraserSize: (size: number) => void;
  setSmoothing: (smoothing: number) => void;
  setColor: (color: string) => void;
  setOpacity: (opacity: number) => void;
  setIsDrawing: (isDrawing: boolean) => void;
  setCurrentLineId: (lineId: string | null) => void;
  setSelectedLineId: (lineId: string | null) => void;
  getCurrentSize: () => number;
}

export const useDrawingStore = create<DrawingState>((set, get) => ({
  tool: "brush",
  brushSize: 8,
  highlighterSize: 20,
  eraserSize: 10,
  smoothing: 0.5, // Default to medium smoothing
  color: "#FF0000",
  opacity: 100, // Default to fully opaque
  isDrawing: false,
  currentLineId: null,
  selectedLineId: null,
  setTool: (tool) => set({ tool }),
  setBrushSize: (size) => set({ brushSize: size }),
  setHighlighterSize: (size) => set({ highlighterSize: size }),
  setEraserSize: (size) => set({ eraserSize: size }),
  setSmoothing: (smoothing) => set({ smoothing }),
  setColor: (color) => set({ color }),
  setOpacity: (opacity) => set({ opacity }),
  setIsDrawing: (isDrawing) => set({ isDrawing }),
  setCurrentLineId: (lineId) => set({ currentLineId: lineId }),
  setSelectedLineId: (lineId) => set({ selectedLineId: lineId }),
  getCurrentSize: () => {
    const state = get();
    switch (state.tool) {
      case "brush":
        return state.brushSize;
      case "highlighter":
        return state.highlighterSize;
      case "eraser":
        return state.eraserSize;
      default:
        return state.brushSize;
    }
  },
}));

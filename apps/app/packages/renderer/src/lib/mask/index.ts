import { create } from "zustand";

export type MaskTool = "lasso" | "shape" | "draw" | "touch";
export type TouchDrawMode = "point" | "draw"; // For touch tool: point clicking or lasso drawing
export type MaskShapeTool = "rectangle" | "ellipse" | "polygon" | "star";
export type TouchLabel = "positive" | "negative";

interface MaskState {
  tool: MaskTool;
  shape: MaskShapeTool;
  brushSize: number;
  featherAmount: number;
  isTracking: boolean;
  trackingEnabled: boolean;
  activeMaskId: string | null;
  autoMask: boolean;
  touchLabel: TouchLabel;
  touchDrawMode: TouchDrawMode;
  isMaskDragging: boolean;
  isOverMask: boolean;
  touchMaskRefetchToken: number | null;
  setTouchMaskRefetchToken: (token: number | null) => void;
  setTool: (tool: MaskTool) => void;
  setShape: (shape: MaskShapeTool) => void;
  setBrushSize: (size: number) => void;
  setFeatherAmount: (amount: number) => void;
  setIsTracking: (isTracking: boolean) => void;
  setTrackingEnabled: (enabled: boolean) => void;
  setActiveMaskId: (id: string | null) => void;
  setAutoMask: (autoMask: boolean) => void;
  setTouchLabel: (label: TouchLabel) => void;
  setTouchDrawMode: (mode: TouchDrawMode) => void;
  setIsMaskDragging: (isDragging: boolean) => void;
  setIsOverMask: (isOverMask: boolean) => void;
}

export const useMaskStore = create<MaskState>((set) => ({
  tool: "touch",
  shape: "rectangle",
  brushSize: 10,
  featherAmount: 2,
  isTracking: false,
  trackingEnabled: false,
  activeMaskId: null,
  autoMask: false,
  touchLabel: "positive",
  touchDrawMode: "point",
  isMaskDragging: false,
  isOverMask: false,
  touchMaskRefetchToken: null,
  setTool: (tool) => set({ tool }),
  setShape: (shape) => set({ shape }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setFeatherAmount: (featherAmount) => set({ featherAmount }),
  setIsTracking: (isTracking) => set({ isTracking }),
  setTrackingEnabled: (trackingEnabled) => set({ trackingEnabled }),
  setActiveMaskId: (activeMaskId) => set({ activeMaskId }),
  setAutoMask: (autoMask) => set({ autoMask }),
  setTouchLabel: (touchLabel) => set({ touchLabel }),
  setTouchDrawMode: (touchDrawMode) => set({ touchDrawMode }),
  setIsMaskDragging: (isMaskDragging) => set({ isMaskDragging }),
  setIsOverMask: (isOverMask) => set({ isOverMask }),
  setTouchMaskRefetchToken: (touchMaskRefetchToken) =>
    set({ touchMaskRefetchToken }),
}));

// Export API functions and types
export {
  createMask,
  useMask,
  flatArrayToPoints,
  pointsToFlatArray,
  normalizePoints,
  normalizePointsWithLabels,
  normalizeBox,
  denormalizeContours,
  type MaskRequest,
  type MaskResponse,
  type ConfigResponse,
  type UseMaskOptions,
  type UseMaskResult,
} from "./api";

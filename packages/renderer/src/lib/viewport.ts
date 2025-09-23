import { create } from "zustand";
import { Point } from "./types";
type ViewTool = 'pointer' | 'hand';


interface ViewportState {
  scale: number; // 1 = 100%
  minScale: number;
  maxScale: number;
  position: Point; // stage position in screen space
  tool: ViewTool;
  clipPositions: Record<string, Point>;
  getClipPosition: (clipId: string) => Point | undefined;
  setClipPosition: (clipId: string, position: Point) => void;
  viewportSize: { width: number; height: number };
  setViewportSize: (size: { width: number; height: number }) => void;
  contentBounds: { x: number; y: number; width: number; height: number } | null;
  setContentBounds: (bounds: { x: number; y: number; width: number; height: number }) => void;
  aspectRatio: { width: number; height: number; id: string };
  setAspectRatio: (ratio: { width: number; height: number; id: string }) => void;
  setTool: (tool: ViewTool) => void;
  setScalePercent: (percent: number) => void;
  setScalePercentCentered: (percent: number) => void;
  setScale: (scale: number) => void;
  zoomAtScreenPoint: (nextScale: number, screenPoint: Point) => void;
  setPosition: (position: Point) => void;
  panBy: (deltaX: number, deltaY: number) => void;
  centerOnWorldPoint: (worldPoint: Point, viewportSize: { width: number; height: number }) => void;
  centerContentAt100: () => void;
}

export const useViewportStore = create<ViewportState>((set, get) => ({
  scale: 1,
  minScale: 0.1,
  maxScale: 4,
  position: { x: 0, y: 0 },
  tool: 'pointer',
  viewportSize: { width: 0, height: 0 },
  clipPositions: {},
  getClipPosition: (clipId) => {
    const clipPositions = get().clipPositions;
    return clipPositions[clipId];
  },
  setClipPosition: (clipId, position) => set({ clipPositions: { ...get().clipPositions, [clipId]: position } }),
  setViewportSize: (size) => set({ viewportSize: size }),
  contentBounds: null,
  setContentBounds: (bounds) => set({ contentBounds: bounds }),
  aspectRatio: { width: 16, height: 9, id: '16:9' },
  setAspectRatio: (ratio) => set({ aspectRatio: ratio }),
  setTool: (tool) => set({ tool }),
  setScalePercent: (percent) => {
    const { minScale, maxScale, viewportSize, position, scale } = get();
    const next = Math.max(minScale, Math.min(maxScale, (percent || 0) / 100));
    if (viewportSize.width === 0 || viewportSize.height === 0) {
      set({ scale: next });
      return;
    }
    const centerScreen = { x: viewportSize.width / 2, y: viewportSize.height / 2 };
    const worldPoint = {
      x: (centerScreen.x - position.x) / scale,
      y: (centerScreen.y - position.y) / scale,
    };
    const newPosition = {
      x: centerScreen.x - worldPoint.x * next,
      y: centerScreen.y - worldPoint.y * next,
    };
    set({ scale: next, position: newPosition });
  },
  setScale: (scale) => {
    const { minScale, maxScale } = get();
    const next = Math.max(minScale, Math.min(maxScale, scale));
    set({ scale: next });
  },
  setScalePercentCentered: (percent) => {
    const { minScale, maxScale, contentBounds, viewportSize } = get();
    const scale = Math.max(minScale, Math.min(maxScale, (percent || 0) / 100));
    if (!contentBounds || viewportSize.width === 0 || viewportSize.height === 0) {
      set({ scale });
      return;
    }
    const worldCenter = {
      x: contentBounds.x + contentBounds.width / 2,
      y: contentBounds.y + contentBounds.height / 2,
    };
    const centerScreen = { x: viewportSize.width / 2, y: viewportSize.height / 2 };
    const position = {
      x: centerScreen.x - worldCenter.x * scale,
      y: centerScreen.y - worldCenter.y * scale,
    };
    set({ scale, position });
  },
  zoomAtScreenPoint: (nextScale, screenPoint) => {
    const { position, scale, minScale, maxScale } = get();
    const clampedScale = Math.max(minScale, Math.min(maxScale, nextScale));
    if (clampedScale === scale) return;
    // compute world point under the cursor
    const worldPoint = {
      x: (screenPoint.x - position.x) / scale,
      y: (screenPoint.y - position.y) / scale,
    };
    // compute new position so the world point stays under the cursor
    const newPosition = {
      x: screenPoint.x - worldPoint.x * clampedScale,
      y: screenPoint.y - worldPoint.y * clampedScale,
    };
    set({ scale: clampedScale, position: newPosition });
  },
  setPosition: (position) => set({ position }),
  panBy: (deltaX, deltaY) => set((state) => ({ position: { x: state.position.x - deltaX, y: state.position.y - deltaY } })),
  centerOnWorldPoint: (worldPoint, viewportSize) => set((state) => {
    const { scale } = state;
    const centerScreen = { x: viewportSize.width / 2, y: viewportSize.height / 2 };
    const position = {
      x: centerScreen.x - worldPoint.x * scale,
      y: centerScreen.y - worldPoint.y * scale,
    };
    return { position };
  }),
  centerContentAt100: () => {
    const { contentBounds, viewportSize } = get();
    if (!contentBounds) return;
    const worldCenter = {
      x: contentBounds.x + contentBounds.width / 2,
      y: contentBounds.y + contentBounds.height / 2,
    };
    const scale = 1;
    const centerScreen = { x: viewportSize.width / 2, y: viewportSize.height / 2 };
    const position = {
      x: centerScreen.x - worldCenter.x * scale,
      y: centerScreen.y - worldCenter.y * scale,
    };
    set({ scale, position });
  },
}));

export type { ViewTool };


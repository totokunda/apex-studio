import { create } from "zustand";

interface ContainerRect {
  top: number;
  height: number;
}

interface LayoutState {
  bottomHeight: number;
  minBottom: number;
  minTop: number;
  handleHeight: number;
  containerRect: ContainerRect | null;
  isDragging: boolean;
  setConstraints: (
    opts: Partial<Pick<LayoutState, "minBottom" | "minTop" | "handleHeight">>,
  ) => void;
  setContainerRect: (rect: ContainerRect) => void;
  setBottomHeight: (height: number) => void;
  startDragging: () => void;
  stopDragging: () => void;
  dragTo: (clientY: number) => void;
}

export const useLayoutStore = create<LayoutState>((set, get) => ({
  bottomHeight: 300,
  minBottom: 240,
  minTop: 360,
  handleHeight: 1,
  containerRect: null,
  isDragging: false,
  setConstraints: (opts) => set((state) => ({ ...state, ...opts })),
  setContainerRect: (rect) => {
    set({ containerRect: rect });
    // Clamp current bottom height when container size changes
    const { minBottom, minTop, handleHeight } = get();
    const maxBottom = Math.max(minBottom, rect.height - handleHeight - minTop);
    const current = get().bottomHeight;
    const clamped = Math.min(Math.max(current, minBottom), maxBottom);
    set({ bottomHeight: clamped });
  },
  setBottomHeight: (height) => {
    const { containerRect, minBottom, minTop, handleHeight } = get();
    if (!containerRect) {
      set({ bottomHeight: Math.max(minBottom, Math.round(height)) });
      return;
    }
    const maxBottom = Math.max(
      minBottom,
      containerRect.height - handleHeight - minTop,
    );
    const clamped = Math.min(
      Math.max(Math.round(height), minBottom),
      maxBottom,
    );
    set({ bottomHeight: clamped });
  },
  startDragging: () => set({ isDragging: true }),
  stopDragging: () => set({ isDragging: false }),
  dragTo: (clientY) => {
    const { containerRect, handleHeight, minTop, minBottom } = get();
    if (!containerRect) return;
    const topHeight = clientY - containerRect.top;
    const maxTop = containerRect.height - handleHeight - minBottom;
    const clampedTop = Math.max(minTop, Math.min(topHeight, maxTop));
    const newBottom = containerRect.height - handleHeight - clampedTop;
    get().setBottomHeight(newBottom);
  },
}));

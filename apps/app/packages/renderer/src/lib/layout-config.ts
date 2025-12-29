import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AppLayoutId = "default" | "media" | "properties";

interface LayoutConfigState {
  layout: AppLayoutId;
  setLayout: (layout: AppLayoutId) => void;
}

export const useLayoutConfigStore = create<LayoutConfigState>()(
  persist(
    (set) => ({
      layout: "default",
      setLayout: (layout) => set({ layout }),
    }),
    { name: "apex-studio:layout-config" },
  ),
);

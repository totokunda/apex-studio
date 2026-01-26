import { create } from "zustand";
import { SidebarSection } from "./types";

interface SidebarStore {
  open: boolean;
  section: SidebarSection | null;
  _setOpen: (open: boolean) => void;
  _setSection: (section: SidebarSection | null) => void;
  openSection: (section: SidebarSection, callback?: () => void) => void;
  closeSection: (callback?: () => void) => void;
}

export const useSidebarStore = create<SidebarStore>((set) => ({
  open: false,
  _setOpen: (open) => set({ open }),
  _setSection: (section) => set({ section }),
  section: "media",
  closeSection: (callback) => {
    callback?.();
    set({ section: null, open: false });
  },
  openSection: (section, callback) => {
    callback?.();
    set({ section, open: true });
  },
}));

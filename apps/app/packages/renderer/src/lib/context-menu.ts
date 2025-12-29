import { create } from "zustand";

export type ContextMenuAction =
  | "copy"
  | "cut"
  | "paste"
  | "delete"
  | "split"
  | "separateAudio"
  | "convertToMedia"
  | "export"
  | "group"
  | "ungroup"
  | "extend"
  | "stabilize"
  | "animate"
  | "inpaint"
  | "outpaint"
  | "control"
  | "editImage"
  | "editVideo";

export type ContextTarget =
  | { type: "clip"; clipIds: string[]; primaryClipId: string; isVideo: boolean }
  | { type: "timeline"; timelineId: string }
  | { type: "textSelection" };

export interface ContextMenuItem {
  id: string;
  label: string;
  action: ContextMenuAction;
  shortcut?: string;
  disabled?: boolean;
}

export interface ContextMenuGroup {
  id: string;
  label?: string;
  items: ContextMenuItem[];
}

interface ContextMenuState {
  open: boolean;
  position: { x: number; y: number };
  items: ContextMenuItem[];
  groups: ContextMenuGroup[] | null;
  target: ContextTarget | null;
  openMenu: (args: {
    position: { x: number; y: number };
    items?: ContextMenuItem[];
    groups?: ContextMenuGroup[];
    target: ContextTarget;
  }) => void;
  closeMenu: () => void;
  setPosition: (pos: { x: number; y: number }) => void;
}

export const useContextMenuStore = create<ContextMenuState>((set) => ({
  open: false,
  position: { x: 0, y: 0 },
  items: [],
  groups: null,
  target: null,
  openMenu: ({ position, items = [], groups, target }) =>
    set({ open: true, position, items, groups: groups ?? null, target }),
  closeMenu: () => set({ open: false, items: [], groups: null, target: null }),
  setPosition: (position) => set({ position }),
}));

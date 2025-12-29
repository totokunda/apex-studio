import { create, useStore } from "zustand";

interface MediaLibraryVersionStore {
  version: number;
  bump: () => void;
}

export const MediaLibraryEvents = create<MediaLibraryVersionStore>(
  (set, get) => ({
    version: 0,
    bump: () => set({ version: get().version + 1 }),
  }),
);

export const useMediaLibraryVersion = (): number =>
  useStore(MediaLibraryEvents).version;

export const bumpMediaLibraryVersion = (): void => {
  try {
    MediaLibraryEvents.getState().bump();
  } catch {}
};

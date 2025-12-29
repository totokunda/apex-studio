import { create } from "zustand";

export interface ManifestStoreState {
  selectedManifestId: string | null;
  setSelectedManifestId: (manifestId: string) => void;
  clearSelectedManifestId: () => void;
}

export const useManifestStore = create<ManifestStoreState>((set, get) => ({
  modelTypes: null,
  manifests: null,
  selectedManifestId: null,
  setSelectedManifestId: (manifestId: string) =>
    set({ selectedManifestId: manifestId }),
  clearSelectedManifestId: () => set({ selectedManifestId: null }),

}));

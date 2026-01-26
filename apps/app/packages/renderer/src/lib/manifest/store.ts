import { create } from "zustand";
import type { ManifestDocument, ModelTypeInfo } from "./api";
import { getManifest, listManifests, listModelTypes } from "./api";

export interface ManifestStoreState {
  // Cached data
  modelTypes: ModelTypeInfo[] | null;
  manifests: ManifestDocument[] | null;
  manifestsById: Record<string, ManifestDocument>;

  // Async state
  loading: {
    modelTypes: boolean;
    manifests: boolean;
    byId: Record<string, boolean>;
  };
  error: {
    modelTypes: string | null;
    manifests: string | null;
    byId: Record<string, string | null>;
  };

  // Actions
  loadModelTypes: (force?: boolean) => Promise<void>;
  loadManifests: (force?: boolean) => Promise<void>;
  loadManifest: (manifestId: string, force?: boolean) => Promise<void>;
  getLoadedManifest: (manifestId: string) => ManifestDocument | null;

  selectedManifestId: string | null;
  setSelectedManifestId: (manifestId: string) => void;
  clearSelectedManifestId: () => void;
}

export const useManifestStore = create<ManifestStoreState>((set, get) => ({
  // Data
  modelTypes: null,
  manifests: null,
  manifestsById: {},

  // Async state
  loading: {
    modelTypes: false,
    manifests: false,
    byId: {},
  },
  error: {
    modelTypes: null,
    manifests: null,
    byId: {},
  },

  // Actions
  loadModelTypes: async (force: boolean = false) => {
    const state = get();
    if (!force && state.modelTypes != null) return;
    if (state.loading.modelTypes) return;
    set((s) => ({
      loading: { ...s.loading, modelTypes: true },
      error: { ...s.error, modelTypes: null },
    }));
    try {
      const res = await listModelTypes();
      if (!res.success) {
        throw new Error(res.error || "Failed to load model types");
      }
      set((s) => ({
        modelTypes: res.data ?? [],
        loading: { ...s.loading, modelTypes: false },
      }));
    } catch (e) {
      set((s) => ({
        loading: { ...s.loading, modelTypes: false },
        error: {
          ...s.error,
          modelTypes: e instanceof Error ? e.message : String(e),
        },
      }));
    }
  },
  loadManifests: async (force: boolean = false) => {
    const state = get();
    if (!force && state.manifests != null) return;
    if (state.loading.manifests) return;
    set((s) => ({
      loading: { ...s.loading, manifests: true },
      error: { ...s.error, manifests: null },
    }));
    try {
      const res = await listManifests();
      if (!res.success) {
        throw new Error(res.error || "Failed to load manifests");
      }
      set((s) => ({
        manifests: res.data ?? [],
        loading: { ...s.loading, manifests: false },
      }));
    } catch (e) {
      set((s) => ({
        loading: { ...s.loading, manifests: false },
        error: {
          ...s.error,
          manifests: e instanceof Error ? e.message : String(e),
        },
      }));
    }
  },
  loadManifest: async (manifestId: string, force: boolean = false) => {
    if (!manifestId) return;
    const state = get();
    if (!force && state.manifestsById[manifestId]) return;
    if (state.loading.byId[manifestId]) return;
    set((s) => ({
      loading: {
        ...s.loading,
        byId: { ...s.loading.byId, [manifestId]: true },
      },
      error: {
        ...s.error,
        byId: { ...s.error.byId, [manifestId]: null },
      },
    }));
    try {
      const res = await getManifest(manifestId);
      if (!res.success || !res.data) {
        throw new Error(res.error || "Failed to load manifest");
      }
      set((s) => ({
        manifestsById: { ...s.manifestsById, [manifestId]: res.data! },
        loading: {
          ...s.loading,
          byId: { ...s.loading.byId, [manifestId]: false },
        },
      }));
    } catch (e) {
      set((s) => ({
        loading: {
          ...s.loading,
          byId: { ...s.loading.byId, [manifestId]: false },
        },
        error: {
          ...s.error,
          byId: {
            ...s.error.byId,
            [manifestId]: e instanceof Error ? e.message : String(e),
          },
        },
      }));
    }
  },
  getLoadedManifest: (manifestId: string) => {
    if (!manifestId) return null;
    return get().manifestsById[manifestId] ?? null;
  },

  selectedManifestId: null,
  setSelectedManifestId: (manifestId: string) =>
    set({ selectedManifestId: manifestId }),
  clearSelectedManifestId: () => set({ selectedManifestId: null }),

}));

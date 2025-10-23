import { create } from 'zustand';
import {
  listModelTypes,
  listManifests,
  listManifestsByModel,
  listManifestsByType,
  listManifestsByModelAndType,
  getManifest,
  type ManifestInfo,
  type ModelTypeInfo,
} from './api';

type ManifestByModelKey = string; // model
type ManifestByTypeKey = string; // modelType

export interface ManifestStoreState {
  modelTypes: ModelTypeInfo[] | null;
  manifests: ManifestInfo[] | null;
  manifestsByModel: Record<ManifestByModelKey, ManifestInfo[]>;
  manifestsByType: Record<ManifestByTypeKey, ManifestInfo[]>;
  manifestsByModelAndType: Record<string, Record<string, ManifestInfo[]>>; // model -> type -> manifests
  manifestById: Record<string, any>;
  selectedManifestId: string | null;
  setSelectedManifestId: (manifestId: string) => void;
  clearSelectedManifestId: () => void;
  getLoadedManifest: (manifestId: string) => ManifestInfo | null;
  loading: {
    modelTypes: boolean;
    manifests: boolean;
    byModel: Record<ManifestByModelKey, boolean>;
    byType: Record<ManifestByTypeKey, boolean>;
    byModelAndType: Record<string, Record<string, boolean>>; // model -> type -> loading
    byId: Record<string, boolean>;
  };

  error: {
    modelTypes: string | null;
    manifests: string | null;
    byModel: Record<ManifestByModelKey, string | null>;
    byType: Record<ManifestByTypeKey, string | null>;
    byModelAndType: Record<string, Record<string, string | null>>;
    byId: Record<string, string | null>;
  };

  // Loaders (idempotent; will not refetch if cached unless force=true)
  loadModelTypes: (force?: boolean) => Promise<void>;
  loadManifests: (force?: boolean) => Promise<void>;
  loadManifestsByModel: (model: string, force?: boolean) => Promise<void>;
  loadManifestsByType: (modelType: string, force?: boolean) => Promise<void>;
  loadManifestsByModelAndType: (model: string, modelType: string, force?: boolean) => Promise<void>;
  loadManifest: (manifestId: string, force?: boolean) => Promise<void>;
}

export const useManifestStore = create<ManifestStoreState>((set, get) => ({
  modelTypes: null,
  manifests: null,
  manifestsByModel: {},
  manifestsByType: {},
  manifestsByModelAndType: {},
  manifestById: {},
  selectedManifestId: null,
  setSelectedManifestId: (manifestId: string) => set({ selectedManifestId: manifestId }),
  clearSelectedManifestId: () => set({ selectedManifestId: null }),
  getLoadedManifest: (manifestId: string) => {
    const state = get();
    const manifests = state.manifests || [];
    const manifest = manifests.find((manifest) => manifest.id === manifestId);
    if (manifest) return manifest;
    return state.manifestById[manifestId] || null;
  },
  loading: {
    modelTypes: false,
    manifests: false,
    byModel: {},
    byType: {},
    byModelAndType: {},
    byId: {},
  },

  error: {
    modelTypes: null,
    manifests: null,
    byModel: {},
    byType: {},
    byModelAndType: {},
    byId: {},
  },


  loadModelTypes: async (force = false) => {
    const state = get();
    if (!force && (state.modelTypes !== null || state.loading.modelTypes)) return;
    set((s) => ({ loading: { ...s.loading, modelTypes: true }, error: { ...s.error, modelTypes: null } }));
    const res = await listModelTypes();
    if (res.success) set((s) => ({ modelTypes: res.data || [], loading: { ...s.loading, modelTypes: false } }));
    else set((s) => ({ loading: { ...s.loading, modelTypes: false }, error: { ...s.error, modelTypes: res.error || 'Failed to load model types' } }));
  },

  loadManifests: async (force = false) => {
    const state = get();
    if (!force && (state.manifests !== null || state.loading.manifests)) return;
    set((s) => ({ loading: { ...s.loading, manifests: true }, error: { ...s.error, manifests: null } }));
    const res = await listManifests();
    if (res.success) set((s) => ({ manifests: res.data || [], loading: { ...s.loading, manifests: false } }));
    else set((s) => ({ loading: { ...s.loading, manifests: false }, error: { ...s.error, manifests: res.error || 'Failed to load manifests' } }));
  },

  loadManifestsByModel: async (model: string, force = false) => {
    const state = get();
    if (!model) return;
    if (!force && (state.manifestsByModel[model] || state.loading.byModel[model])) return;
    set((s) => ({ loading: { ...s.loading, byModel: { ...s.loading.byModel, [model]: true } }, error: { ...s.error, byModel: { ...s.error.byModel, [model]: null } } }));
    const res = await listManifestsByModel(model);
    if (res.success) set((s) => ({ manifestsByModel: { ...s.manifestsByModel, [model]: res.data || [] }, loading: { ...s.loading, byModel: { ...s.loading.byModel, [model]: false } } }));
    else set((s) => ({ loading: { ...s.loading, byModel: { ...s.loading.byModel, [model]: false } }, error: { ...s.error, byModel: { ...s.error.byModel, [model]: res.error || 'Failed to load' } } }));
  },

  loadManifestsByType: async (modelType: string, force = false) => {
    const state = get();
    if (!modelType) return;
    if (!force && (state.manifestsByType[modelType] || state.loading.byType[modelType])) return;
    set((s) => ({ loading: { ...s.loading, byType: { ...s.loading.byType, [modelType]: true } }, error: { ...s.error, byType: { ...s.error.byType, [modelType]: null } } }));
    const res = await listManifestsByType(modelType);
    if (res.success) set((s) => ({ manifestsByType: { ...s.manifestsByType, [modelType]: res.data || [] }, loading: { ...s.loading, byType: { ...s.loading.byType, [modelType]: false } } }));
    else set((s) => ({ loading: { ...s.loading, byType: { ...s.loading.byType, [modelType]: false } }, error: { ...s.error, byType: { ...s.error.byType, [modelType]: res.error || 'Failed to load' } } }));
  },

  loadManifestsByModelAndType: async (model: string, modelType: string, force = false) => {
    const state = get();
    if (!model || !modelType) return;
    const cache = state.manifestsByModelAndType[model]?.[modelType];
    const isLoading = !!state.loading.byModelAndType[model]?.[modelType];
    if (!force && (cache || isLoading)) return;
    set((s) => ({
      loading: {
        ...s.loading,
        byModelAndType: {
          ...s.loading.byModelAndType,
          [model]: { ...(s.loading.byModelAndType[model] || {}), [modelType]: true },
        },
      },
      error: {
        ...s.error,
        byModelAndType: {
          ...s.error.byModelAndType,
          [model]: { ...(s.error.byModelAndType[model] || {}), [modelType]: null },
        },
      },
    }));
    const res = await listManifestsByModelAndType(model, modelType);
    if (res.success) set((s) => ({
      manifestsByModelAndType: {
        ...s.manifestsByModelAndType,
        [model]: { ...(s.manifestsByModelAndType[model] || {}), [modelType]: res.data || [] },
      },
      loading: {
        ...s.loading,
        byModelAndType: {
          ...s.loading.byModelAndType,
          [model]: { ...(s.loading.byModelAndType[model] || {}), [modelType]: false },
        },
      },
    }));
    else set((s) => ({
      loading: {
        ...s.loading,
        byModelAndType: {
          ...s.loading.byModelAndType,
          [model]: { ...(s.loading.byModelAndType[model] || {}), [modelType]: false },
        },
      },
      error: {
        ...s.error,
        byModelAndType: {
          ...s.error.byModelAndType,
          [model]: { ...(s.error.byModelAndType[model] || {}), [modelType]: res.error || 'Failed to load' },
        },
      },
    }));
  },

  loadManifest: async (manifestId: string, force = false) => {
    const state = get();
    if (!manifestId) return;
    if (!force && (state.manifestById[manifestId] || state.loading.byId[manifestId])) return;
    set((s) => ({ loading: { ...s.loading, byId: { ...s.loading.byId, [manifestId]: true } }, error: { ...s.error, byId: { ...s.error.byId, [manifestId]: null } } }));
    const res = await getManifest(manifestId);
    if (res.success) set((s) => ({ manifestById: { ...s.manifestById, [manifestId]: res.data }, loading: { ...s.loading, byId: { ...s.loading.byId, [manifestId]: false } } }));
    else set((s) => ({ loading: { ...s.loading, byId: { ...s.loading.byId, [manifestId]: false } }, error: { ...s.error, byId: { ...s.error.byId, [manifestId]: res.error || 'Failed to load' } } }));
  },
}));



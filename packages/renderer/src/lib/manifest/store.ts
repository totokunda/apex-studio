import { create } from "zustand";
import {
  listModelTypes,
  listManifests,
  getManifest,
  getManifestPart,
  type ModelTypeInfo,
  ManifestDocument,
} from "./api";

export interface ManifestStoreState {
  modelTypes: ModelTypeInfo[] | null;
  manifests: ManifestDocument[] | null;
  selectedManifestId: string | null;
  setSelectedManifestId: (manifestId: string) => void;
  clearSelectedManifestId: () => void;
  getLoadedManifest: (manifestId: string) => ManifestDocument | null;
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

  // Loaders (idempotent; will not refetch if cached unless force=true)
  loadModelTypes: (force?: boolean) => Promise<void>;
  loadManifests: (force?: boolean) => Promise<void>;
  loadManifest: (manifestId: string, force?: boolean) => Promise<void>;
  refreshManifestPart: (manifestId: string, pathDot: string) => Promise<void>;
}

export const useManifestStore = create<ManifestStoreState>((set, get) => ({
  modelTypes: null,
  manifests: null,
  selectedManifestId: null,
  setSelectedManifestId: (manifestId: string) =>
    set({ selectedManifestId: manifestId }),
  clearSelectedManifestId: () => set({ selectedManifestId: null }),
  getLoadedManifest: (manifestId: string) => {
    const state = get();
    const manifests = state.manifests || [];
    const manifest = manifests.find(
      (manifest) => manifest.metadata?.id === manifestId,
    );
    if (manifest) return manifest as ManifestDocument;
    return null;
  },
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

  loadModelTypes: async (force = false) => {
    const state = get();
    if (!force && (state.modelTypes !== null || state.loading.modelTypes))
      return;
    set((s) => ({
      loading: { ...s.loading, modelTypes: true },
      error: { ...s.error, modelTypes: null },
    }));
    const res = await listModelTypes();
    if (res.success)
      set((s) => ({
        modelTypes: res.data || [],
        loading: { ...s.loading, modelTypes: false },
      }));
    else
      set((s) => ({
        loading: { ...s.loading, modelTypes: false },
        error: {
          ...s.error,
          modelTypes: res.error || "Failed to load model types",
        },
      }));
  },

  loadManifests: async (force = false) => {
    const state = get();
    if (!force && (state.manifests !== null || state.loading.manifests)) return;
    set((s) => ({
      loading: { ...s.loading, manifests: true },
      error: { ...s.error, manifests: null },
    }));
    const res = await listManifests();
    if (res.success)
      set((s) => ({
        manifests: res.data || [],
        loading: { ...s.loading, manifests: false },
      }));
    else
      set((s) => ({
        loading: { ...s.loading, manifests: false },
        error: {
          ...s.error,
          manifests: res.error || "Failed to load manifests",
        },
      }));
  },

  loadManifest: async (manifestId: string, force = false) => {
    const state = get();
    if (!manifestId) return;
    const already = state.getLoadedManifest(manifestId);
    if (!force && (already || state.loading.byId[manifestId])) return;
    set((s) => ({
      loading: {
        ...s.loading,
        byId: { ...s.loading.byId, [manifestId]: true },
      },
      error: { ...s.error, byId: { ...s.error.byId, [manifestId]: null } },
    }));
    const res = await getManifest(manifestId);
    if (res.success)
      set((s) => {
        const incoming = res.data as ManifestDocument | undefined;
        if (!incoming) {
          return {
            loading: {
              ...s.loading,
              byId: { ...s.loading.byId, [manifestId]: false },
            },
          };
        }
        const current = s.manifests || [];
        const idx = current.findIndex(
          (m) => (m.metadata?.id || m.id) === manifestId,
        );
        const next = [...current];
        if (idx >= 0) next[idx] = incoming;
        else next.push(incoming);
        return {
          manifests: next,
          loading: {
            ...s.loading,
            byId: { ...s.loading.byId, [manifestId]: false },
          },
        };
      });
    else
      set((s) => ({
        loading: {
          ...s.loading,
          byId: { ...s.loading.byId, [manifestId]: false },
        },
        error: {
          ...s.error,
          byId: {
            ...s.error.byId,
            [manifestId]: res.error || "Failed to load",
          },
        },
      }));
  },

  refreshManifestPart: async (manifestId: string, pathDot: string) => {
    const state = get();
    if (!manifestId || !pathDot) return;

    // Ensure base manifest is present; load if missing
    let current = state.getLoadedManifest(manifestId) as any;
    if (!current) {
      const resFull = await getManifest(manifestId);
      if (resFull.success && resFull.data) {
        current = resFull.data;
        set((s) => {
          const list = s.manifests || [];
          const idx = list.findIndex(
            (m) => (m.metadata?.id || (m as any).id) === manifestId,
          );
          const next = [...list];
          if (idx >= 0) next[idx] = current as ManifestDocument;
          else next.push(current as ManifestDocument);
          return { manifests: next };
        });
      } else {
        // If full fetch failed, abort
        return;
      }
    }

    // Fetch the part
    const res = await getManifestPart<any>(manifestId, pathDot);
    if (!res.success) return;
    const partValue = res.data;

    // Apply replacement at dot path in a cloned manifest object
    const clone: any = JSON.parse(JSON.stringify(current));
    const tokens = pathDot.split(".").filter((t) => t.length > 0);
    let cursor: any = clone;
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      const isLast = i === tokens.length - 1;
      const idx = Number.isFinite(Number(t)) ? Number(t) : null;
      if (isLast) {
        if (idx !== null && Array.isArray(cursor)) {
          cursor[idx] = partValue;
        } else if (idx !== null) {
          // Create array if needed
          if (!Array.isArray(cursor)) {
            // If parent expects array, replace with array
            cursor = [];
          }
          (cursor as any[])[idx] = partValue;
        } else {
          cursor[t] = partValue;
        }
      } else {
        if (idx !== null) {
          if (!Array.isArray(cursor)) {
            // Create array if missing
            const arr: any[] = [];
            // Assign to parent appropriately by looking back one token
            const prevToken = tokens[i - 1];
            if (i === 0) {
              // top-level replacement (unlikely for array)
              // no-op here; will assign at end
            } else {
              const prevIdx = Number.isFinite(Number(prevToken))
                ? Number(prevToken)
                : null;
              if (prevIdx !== null) {
                // parent was array
                // Not easily re-bindable; bail out to avoid corrupting structure
              } else {
                // parent was object
                try {
                  (clone as any)[prevToken] = arr;
                } catch {}
              }
            }
            cursor = arr;
          }
          if (!cursor[idx]) cursor[idx] = {};
          cursor = cursor[idx];
        } else {
          if (typeof cursor[t] !== "object" || cursor[t] == null) {
            cursor[t] = {};
          }
          cursor = cursor[t];
        }
      }
    }

    // update manifests list with the modified manifest
    set((s) => {
      const list = s.manifests || [];
      const idx = list.findIndex(
        (m) => (m.metadata?.id || (m as any).id) === manifestId,
      );
      const next = [...list];
      if (idx >= 0) next[idx] = clone as ManifestDocument;
      else next.push(clone as ManifestDocument);
      return { manifests: next };
    });
  },
}));

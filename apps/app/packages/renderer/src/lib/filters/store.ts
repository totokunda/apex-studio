import { create } from "zustand";
import { fetchFilters } from "@app/preload";
import { Filter } from "@/lib/types";

interface FiltersStoreState {
  filters: Filter[] | null;
  loading: boolean;
  error: string | null;
  loadedAt?: number;
  load: (force?: boolean) => Promise<void>;
}

export const useFiltersStore = create<FiltersStoreState>((set, get) => ({
  filters: null,
  loading: false,
  error: null,

  load: async (force?: boolean) => {
    const { filters, loading } = get();
    if (loading) return; // prevent concurrent
    if (!force && Array.isArray(filters) && filters.length > 0) return; // cached

    set({ loading: true, error: null });
    try {
      const data = await fetchFilters();
      set({ filters: data ?? [], loading: false, loadedAt: Date.now() });
    } catch (e: any) {
      set({ error: e?.message || "Failed to load filters", loading: false });
    }
  },
}));

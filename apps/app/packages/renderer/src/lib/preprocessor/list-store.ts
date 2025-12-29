import { create } from "zustand";
import { listPreprocessors, type Preprocessor } from "./api";

interface PreprocessorsListState {
  preprocessors: Preprocessor[] | null;
  loading: boolean;
  error: string | null;
  loadedAt?: number;
  load: (force?: boolean) => Promise<void>;
  setSelectedPreprocessorId: (preprocessorId: string | null) => void;
  selectedPreprocessorId: string | null;
}

export const usePreprocessorsListStore = create<PreprocessorsListState>(
  (set, get) => ({
    preprocessors: null,
    loading: false,
    error: null,
    selectedPreprocessorId: null,
    setSelectedPreprocessorId: (preprocessorId: string | null) =>
      set({ selectedPreprocessorId: preprocessorId }),
    load: async (force?: boolean) => {
      const { preprocessors, loading } = get();
      if (loading) return; // prevent concurrent
      if (!force && Array.isArray(preprocessors) && preprocessors.length > 0)
        return; // cached

      set({ loading: true, error: null });
      try {
        const res = await listPreprocessors();
        const list = res.data?.preprocessors ?? [];
        set({ preprocessors: list, loading: false, loadedAt: Date.now() });
      } catch (e: any) {
        set({
          error: e?.message || "Failed to load preprocessors",
          loading: false,
        });
      }
    },
  }),
);

import { create } from "zustand";

interface PrepareStoreState {
    isPreparingGenerationState: Record<string, boolean>;
    setIsPreparingGeneration: (clipId: string, isPreparingGeneration: boolean) => void;
    getIsPreparingGeneration: (clipId: string) => boolean;
}

export const usePrepareStore = create<PrepareStoreState>((set, get) => ({
    isPreparingGenerationState: {},
    setIsPreparingGeneration: (clipId: string, isPreparingGeneration: boolean) => {
        set((state) => ({
            isPreparingGenerationState: { ...state.isPreparingGenerationState, [clipId]: isPreparingGeneration },
        }));
    },
    getIsPreparingGeneration: (clipId: string) => {
        return get().isPreparingGenerationState[clipId] ?? false;
    },
}));
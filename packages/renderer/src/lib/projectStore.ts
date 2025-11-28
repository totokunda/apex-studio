import { create } from "zustand";
import type { ProjectSummary } from "./persistence/types";

type ProjectStore = {
  projects: ProjectSummary[];
  currentProjectId: string | null;
  currentProjectName: string | null;
  loading: boolean;
  saving: boolean;
  lastSavedAt: number | null;
  setProjects: (projects: ProjectSummary[]) => void;
  upsertProject: (project: ProjectSummary) => void;
  setCurrentProject: (id: string | null, name?: string | null) => void;
  markSaving: (saving: boolean) => void;
  markSavedAt: (timestamp: number) => void;
  setLoading: (loading: boolean) => void;
  createProject: (name?: string) => Promise<ProjectSummary | null>;
  renameProject: (id: string, name: string) => Promise<ProjectSummary | null>;
  switchProject: (id: string) => Promise<void>;
};

const noopAsyncProject = async () => null;
const noopAsyncVoid = async () => {};

export const useProjectStore = create<ProjectStore>((set, get) => ({
  projects: [],
  currentProjectId: null,
  currentProjectName: null,
  loading: true,
  saving: false,
  lastSavedAt: null,
  setProjects: (projects) => set({ projects }),
  upsertProject: (project) =>
    set((state) => {
      const existingIdx = state.projects.findIndex((p) => p.id === project.id);
      const next = [...state.projects];
      if (existingIdx >= 0) {
        next[existingIdx] = project;
      } else {
        next.push(project);
      }
      next.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
      const update: Partial<ProjectStore> = { projects: next };
      if (state.currentProjectId === project.id) {
        update.currentProjectName = project.name;
      }
      return update as Partial<ProjectStore>;
    }),
  setCurrentProject: (id, name) =>
    set(() => {
      const fallbackName = id
        ? (get().projects.find((p) => p.id === id)?.name ?? null)
        : null;
      return {
        currentProjectId: id,
        currentProjectName: name ?? fallbackName,
      };
    }),
  markSaving: (saving) => set({ saving }),
  markSavedAt: (timestamp) => set({ lastSavedAt: timestamp, saving: false }),
  setLoading: (loading) => set({ loading }),
  createProject: noopAsyncProject,
  renameProject: noopAsyncProject,
  switchProject: noopAsyncVoid,
}));

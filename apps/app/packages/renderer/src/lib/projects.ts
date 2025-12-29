import { create } from "zustand";
import { withJsonProjectPersistence } from "./middlewares/json-persistence";

export interface ProjectSettings {
  id: number;
  name: string;
  fps: number;
  folderUuid: string;
  aspectRatio: { width: number; height: number; id: string };
}

interface ProjectsStoreState {
  projects: ProjectSettings[];
  activeProjectId: string | number | null;
  // True once we've attempted an initial load (and ensured at least one project exists)
  projectsLoaded: boolean;
  setProjects: (projects: ProjectSettings[]) => void;
  addProject: (project: ProjectSettings) => void;
  updateProject: (
    id: string | number,
    payload: Partial<Pick<ProjectSettings, "name" | "fps">>,
  ) => void;
  removeProject: (id: string | number) => void;
  setActiveProjectId: (id: string | number) => void;
  getActiveProject: () => ProjectSettings | null;
}

export const useProjectsStore = create<ProjectsStoreState>(
  withJsonProjectPersistence((set, get) => ({
    projects: [],
    activeProjectId: null,
    projectsLoaded: false,

    setProjects: (projects) => set({ projects }),

    addProject: (project) =>
      set((state) => ({
        projects: [...state.projects, project],
        activeProjectId: state.activeProjectId ?? project.id,
      })),

    updateProject: (id, payload) =>
      set((state) => ({
        projects: state.projects.map((p) =>
          p.id === id ? { ...p, ...payload } : p,
        ),
      })),

    removeProject: (id) =>
      set((state) => {
        const remaining = state.projects.filter((p) => p.id !== id);
        let active = state.activeProjectId;
        if (active === id) {
          active = remaining.length > 0 ? remaining[0].id : null;
        }
        return { projects: remaining, activeProjectId: active };
      }),

    setActiveProjectId: (id) => set({ activeProjectId: id }),

    getActiveProject: () => {
      const state = get();
      if (state.activeProjectId == null) return null;
      return (
        state.projects.find((p) => p.id === state.activeProjectId) ?? null
      );
    },
  })),
);

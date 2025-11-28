import {
  listProjects as ipcListProjects,
  createProject as ipcCreateProject,
  renameProject as ipcRenameProject,
  loadProjectState as ipcLoadProjectState,
  saveProjectState as ipcSaveProjectState,
} from "@app/preload";
import type {
  PersistedProjectSnapshot,
  ProjectStateEnvelope,
  ProjectSummary,
} from "./types";

const normalizeResponse = <T = any>(raw: any): ProjectStateEnvelope => {
  if (raw && typeof raw === "object" && "success" in raw) {
    return raw as ProjectStateEnvelope;
  }
  return { success: true, data: raw };
};

export async function fetchProjects(): Promise<
  ProjectStateEnvelope<ProjectSummary[]>
> {
  const res = normalizeResponse<ProjectSummary[]>(await ipcListProjects());
  if (res.success) {
    const data = (res.data as any)?.projects ?? res.data;
    if (Array.isArray(data)) {
      return { success: true, data };
    }
    return { success: true, data: [] };
  }
  return res;
}

export async function createProject(
  name?: string,
): Promise<ProjectStateEnvelope<ProjectSummary>> {
  return normalizeResponse<ProjectSummary>(await ipcCreateProject(name));
}

export async function renameProject(
  id: string,
  name: string,
): Promise<ProjectStateEnvelope<ProjectSummary>> {
  return normalizeResponse<ProjectSummary>(await ipcRenameProject(id, name));
}

export async function loadProjectSnapshot(
  projectId: string,
): Promise<ProjectStateEnvelope<PersistedProjectSnapshot>> {
  return normalizeResponse<PersistedProjectSnapshot>(
    await ipcLoadProjectState(projectId),
  );
}

export async function saveProjectSnapshot(
  projectId: string,
  snapshot: PersistedProjectSnapshot,
): Promise<ProjectStateEnvelope<{ updatedAt: number }>> {
  return normalizeResponse(
    await ipcSaveProjectState(projectId, snapshot as any),
  );
}

import {
  listManifestModelTypes as listManifestModelTypesPreload,
  listManifests as listManifestsPreload,
  listManifestsByModel as listManifestsByModelPreload,
  listManifestsByType as listManifestsByTypePreload,
  listManifestsByModelAndType as listManifestsByModelAndTypePreload,
  getManifest as getManifestPreload,
} from '@app/preload';

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type ModelTypeInfo = {
  key: string;
  label: string;
  description: string;
};

export type ManifestInfo = {
  id: string;
  name: string;
  model: string;
  model_type: string[] | string;
  full_path: string;
  version: string;
  description: string;
  tags: string[];
  author: string;
  license: string;
  demo_path: string;
};

export async function listModelTypes(): Promise<ConfigResponse<ModelTypeInfo[]>> {
  return await listManifestModelTypesPreload();
}

export async function listManifests(): Promise<ConfigResponse<ManifestInfo[]>> {
  return await listManifestsPreload();
}

export async function listManifestsByModel(model: string): Promise<ConfigResponse<ManifestInfo[]>> {
  return await listManifestsByModelPreload(model);
}

export async function listManifestsByType(modelType: string): Promise<ConfigResponse<ManifestInfo[]>> {
  return await listManifestsByTypePreload(modelType);
}

export async function listManifestsByModelAndType(model: string, modelType: string): Promise<ConfigResponse<ManifestInfo[]>> {
  return await listManifestsByModelAndTypePreload(model, modelType);
}

export async function getManifest(manifestId: string): Promise<ConfigResponse<any>> {
  return await getManifestPreload(manifestId);
}



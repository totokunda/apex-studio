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
  downloaded?: boolean;
};

// Manifest v1 Types (aligned with backend schema_v1 and manifest_updated YAMLs)
export type ManifestSchedulerOption = {
  name: string;
  label?: string;
  description?: string;
  base?: string;
  config_path?: string;
  [key: string]: any;
};

export type ManifestResourceRequirements = {
  min_vram_gb?: number;
  recommended_vram_gb?: number;
  compute_capability?: string;
  [key: string]: any;
};

export type ManifestComponentModelPathItem = {
  path: string;
  variant?: string;
  precision?: string;
  type?: string;
  resource_requirements?: ManifestResourceRequirements;
  [key: string]: any;
};

export type ManifestComponent = {
  type: 'scheduler' | 'vae' | 'text_encoder' | 'transformer' | 'helper' | string;
  name?: string;
  label?: string;
  base?: string;
  model_path?: string | ManifestComponentModelPathItem[];
  config_path?: string;
  file_pattern?: string;
  tag?: string;
  key_map?: Record<string, any>;
  extra_kwargs?: Record<string, any>;
  save_path?: string;
  converter_kwargs?: Record<string, any>;
  model_key?: string;
  extra_model_paths?: string[];
  converted_model_path?: string;
  scheduler_options?: ManifestSchedulerOption[];
  gguf_files?: { type: string; path: string }[];
  deprecated?: boolean;
  downloaded?: boolean;
  // Common text encoder extras seen in manifests
  tokenizer_class?: string;
  tokenizer_name?: string;
  tokenizer_kwargs?: Record<string, any>;
  [key: string]: any;
};

export type ManifestExamplesItem = {
  name?: string;
  description?: string;
  parameters?: Record<string, any>;
};

export type ManifestMetadata = {
  id?: string;
  model?: string;
  name: string;
  version?: string;
  description?: string;
  tags?: string[];
  author?: string;
  license?: string;
  homepage?: string;
  registry?: string;
  demo_path?: string;
  annotations?: Record<string, any>;
  examples?: ManifestExamplesItem[];
  [key: string]: any;
};

export type ManifestSpec = {
  engine?: string;
  model_type?: string | string[];
  model_types?: string[];
  engine_type?: 'torch' | 'mlx' | string;
  denoise_type?: string;
  shared?: string[];
  components?: ManifestComponent[];
  preprocessors?: any[];
  postprocessors?: any[];
  defaults?: Record<string, any>;
  loras?: Array<string | Record<string, any>>;
  save?: Record<string, any>;
  resource_requirements?: ManifestResourceRequirements;
  ui?: any; // UI schema is large; typed loosely here
  [key: string]: any;
};

export type ManifestDocument = {
  api_version: string;
  kind: 'Model' | 'Pipeline' | string;
  metadata: ManifestMetadata;
  spec: ManifestSpec;
  ui?: any; // allow top-level UI per loader normalization
  [key: string]: any;
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

export async function getManifest(manifestId: string): Promise<ConfigResponse<ManifestDocument>> {
  return (await getManifestPreload(manifestId)) as ConfigResponse<ManifestDocument>;
}




import {
  listManifestModelTypes as listManifestModelTypesPreload,
  listManifests as listManifestsPreload,
  listManifestsByModel as listManifestsByModelPreload,
  listManifestsByType as listManifestsByTypePreload,
  listManifestsByModelAndType as listManifestsByModelAndTypePreload,
  getManifest as getManifestPreload,
  getManifestPart as getManifestPartPreload,
  validateAndRegisterCustomModelPath as validateAndRegisterCustomModelPathPreload,
  deleteCustomModelPath as deleteCustomModelPathPreload,
  updateManifestLoraScale as updateManifestLoraScalePreload,
  updateManifestLoraName as updateManifestLoraNamePreload,
  deleteManifestLora as deleteManifestLoraPreload,
} from "@app/preload";
import { ClipType } from "../types";

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
  file_size?: number;
  resource_requirements?: ManifestResourceRequirements;
  [key: string]: any;
  custom?: boolean;
  is_downloaded?: boolean;
};

export type ManifestComponent = {
  type:
    | "scheduler"
    | "vae"
    | "text_encoder"
    | "transformer"
    | "helper"
    | "extra_model_path"
    | string;
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
  extra_model_paths?: (string | ManifestComponentModelPathItem)[];
  converted_model_path?: string;
  scheduler_options?: ManifestSchedulerOption[];
  gguf_files?: { type: string; path: string }[];
  deprecated?: boolean;
  is_downloaded?: boolean;
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
  categories?: string[];
};

// UI Schema (derived from manifest_updated YAML structure)
export type UILayoutFlow = "row" | "column";

export type UIFloatingRegion = {
  inputs: string[];
  flow?: UILayoutFlow;
};

export type UIFloatingPanel = {
  regions: Record<string, UIFloatingRegion>;
};

export type UIPanelLayout = {
  flow?: UILayoutFlow;
  rows: string[][];
};

export type UIPanel = {
  name: string;
  label?: string;
  icon?: string;
  collapsible?: boolean;
  default_open?: boolean;
  layout: UIPanelLayout;
};

export type UIInputBase = {
  id: string;
  value?: string;
  label?: string;
  description?: string;
  panel?: string;
  required?: boolean;
  default?: any;
  floating_panel?: boolean;
};

export type UIInputText = UIInputBase & {
  type: "text";
  placeholder?: string;
};

export type UIInputNumber = UIInputBase & {
  type: "number";
  value_type?: "integer" | "float" | string;
  min?: number;
  max?: number;
  step?: number;
};

export type UIInputNumberSlider = UIInputBase & {
  type: "number+slider";
  value_type?: "integer" | "float" | string;
  min?: number;
  max?: number;
  step?: number;
};

export type UIInputNumberList = UIInputBase & {
  type: "number_list";
  value_type?: "integer" | "float" | string;
  items?: UIInputNumber[];
};

export type UIInputRandom = UIInputBase & {
  type: "random";
  min?: number;
  max?: number;
  step?: number;
};

export type UIInputBoolean = UIInputBase & {
  type: "boolean";
};

export type UIInputMapDimensions = {
  map_h?: string; // the id of the input for height mapping
  map_w?: string; // the id of the input for width mapping
  scale_by: string;
};

export type UIInputVideo = UIInputBase &
  UIInputMapDimensions & {
    type: "video";
    max_duration_secs?: number;
  };


export type UIInputVideoMask = UIInputBase &
  UIInputMapDimensions & {
    type: "video+mask";
    map_to?: string;
    max_duration_secs?: number;
  };

export type UIInputImagePreprocessor = UIInputBase &
  UIInputMapDimensions & {
    type: "image+preprocessor";
    preprocessor_ref?: string;
    preprocessor_kwargs?: Record<string, any>;
  };

export type UIInputVideoPreprocessor = UIInputBase &
  UIInputMapDimensions & {
    type: "video+preprocessor";
    preprocessor_ref?: string;
    max_duration_secs?: number;
    preprocessor_kwargs?: Record<string, any>;
  };

export type UIInputImage = UIInputBase &
  UIInputMapDimensions & {
    type: "image";
  };

export type UIInputImageMask = UIInputBase &
  UIInputMapDimensions & {
    type: "image+mask";
    map_to?: string;
  };

export type UIInputImageList = UIInputBase &
  UIInputMapDimensions & {
    type: "image_list";
    items?: UIInputImage[];
    min?: number;
    max?: number;
  };

export type UIInputAudio = UIInputBase & {
  type: "audio";
};

export type UIInputSelect = UIInputBase & {
  type: "select";
  options?: { name: string; value: string }[];
};

export type UIInputOther = UIInputBase & {
  type: string;
  [key: string]: any;
};

export type UIInput =
  | UIInputText
  | UIInputNumber
  | UIInputNumberSlider
  | UIInputRandom
  | UIInputBoolean
  | UIInputVideo
  | UIInputVideoMask
  | UIInputImagePreprocessor
  | UIInputVideoPreprocessor
  | UIInputImage
  | UIInputImageList
  | UIInputImageMask
  | UIInputAudio
  | UIInputSelect
  | UIInputNumberList
  | UIInputOther;

export type UISchema = {
  floating_panel?: UIFloatingPanel;
  panels?: UIPanel[];
  inputs: UIInput[];
  [key: string]: any;
};

export type ManifestSpec = {
  engine?: string;
  model_type?: string | string[];
  engine_type?: "torch" | "mlx" | string;
  fps?: number;
  min_duration_secs?: number;
  default_duration_secs?: number;
  max_duration_secs?: number;
  attention_types: string[];
  attention_types_detail: {
    name: string;
    label: string;
    description: string;
  }[];
  denoise_type?: string;
  shared?: string[];
  components?: ManifestComponent[];
  defaults?: Record<string, any>;
  save?: Record<string, any>;
  resource_requirements?: ManifestResourceRequirements;
  ui?: UISchema; // Typed UI schema
  loras?: LoraType[];
  [key: string]: any;
};

export type ManifestWithType = ManifestDocument & {
  type: ClipType;
  category: string;
};

export type LoraType = {
      source?: string;
      remote_source?: string;
      verified?: boolean;
      scale?: number;
      name?: string;
      label?: string;
      is_downloaded?: boolean;
      required?: boolean;
      component?: string;
    }

export type ManifestDocument = {
  api_version: string;
  kind: "Model" | "Pipeline" | string;
  metadata: ManifestMetadata;
  spec: ManifestSpec;
  ui?: UISchema; // allow top-level UI per loader normalization
  id: string;
  name: string;
  model: string;
  model_type: string[];
  version: string;
  description: string;
  tags: string[];
  author: string;
  license: string;
  demo_path: string;
  downloaded: boolean;
};

export async function listModelTypes(): Promise<
  ConfigResponse<ModelTypeInfo[]>
> {
  return await listManifestModelTypesPreload();
}

export async function listManifests(): Promise<
  ConfigResponse<ManifestDocument[]>
> {
  return await listManifestsPreload();
}

export async function listManifestsByModel(
  model: string,
): Promise<ConfigResponse<ManifestDocument[]>> {
  return await listManifestsByModelPreload(model);
}

export async function listManifestsByType(
  modelType: string,
): Promise<ConfigResponse<ManifestDocument[]>> {
  return await listManifestsByTypePreload(modelType);
}

export async function listManifestsByModelAndType(
  model: string,
  modelType: string,
): Promise<ConfigResponse<ManifestDocument[]>> {
  return await listManifestsByModelAndTypePreload(model, modelType);
}

export async function getManifest(
  manifestId: string,
): Promise<ConfigResponse<ManifestDocument>> {
  return (await getManifestPreload(
    manifestId,
  )) as ConfigResponse<ManifestDocument>;
}

export async function getManifestPart<T = any>(
  manifestId: string,
  pathDot?: string,
): Promise<ConfigResponse<T>> {
  return (await getManifestPartPreload(
    manifestId,
    pathDot,
  )) as ConfigResponse<T>;
}

export async function validateAndRegisterCustomModelPath(
  manifestId: string,
  componentIndex: number,
  name: string | undefined,
  path: string,
): Promise<ConfigResponse<any>> {
  return (await validateAndRegisterCustomModelPathPreload(
    manifestId,
    componentIndex,
    name,
    path,
  )) as ConfigResponse<any>;
}

export async function deleteCustomModelPath(
  manifestId: string,
  componentIndex: number,
  path: string,
): Promise<ConfigResponse<any>> {
  return (await deleteCustomModelPathPreload(
    manifestId,
    componentIndex,
    path,
  )) as ConfigResponse<any>;
}

export async function updateManifestLoraScale(
  manifestId: string,
  loraIndex: number,
  scale: number,
): Promise<ConfigResponse<any>> {
  return (await updateManifestLoraScalePreload(
    manifestId,
    loraIndex,
    scale,
  )) as ConfigResponse<any>;
}

export async function updateManifestLoraName(
  manifestId: string,
  loraIndex: number,
  name: string,
): Promise<ConfigResponse<any>> {
  return (await updateManifestLoraNamePreload(
    manifestId,
    loraIndex,
    name,
  )) as ConfigResponse<any>;
}

export async function deleteManifestLora(
  manifestId: string,
  loraIndex: number,
): Promise<ConfigResponse<any>> {
  return (await deleteManifestLoraPreload(
    manifestId,
    loraIndex,
  )) as ConfigResponse<any>;
}


export const extractAllDownloadableDefaultPaths = (manifest: ManifestDocument) => {
  const spec = manifest.spec;
  const components = spec.components ?? [];
  const loras = spec.loras ?? [];
  const allDownloadablePaths = new Set<{type: string, path: string | string[], index?: number}>();
  for (const [index, component] of components.entries()) {
    const componentDownloadablePath = [];
    if (component.is_downloaded) continue;
    if (component.model_path) {
      if (typeof component.model_path === "string") {
        componentDownloadablePath.push({type: "component", path: component.model_path, index});
      } else {
        for (const pathItem of component.model_path) {
          if (pathItem.path && !pathItem.is_downloaded && !pathItem.custom && (pathItem.variant ?? 'default').toLowerCase() == "default") {
            componentDownloadablePath.push({type: "component", path: pathItem.path, index});
          }
        }
      }
      if (component.config_path) {
        componentDownloadablePath.push({type: "component", path: component.config_path, index});
      }
    }
    if (componentDownloadablePath.length > 0) {
      allDownloadablePaths.add({type: "component", path: componentDownloadablePath.map((path) => path.path), index});
    }
  }
  for (const [index, lora] of loras.entries()) {
    if (lora.source && !lora.is_downloaded) {
      allDownloadablePaths.add({type: "lora", path: lora.source, index: index});
    }
  }
  return Array.from(allDownloadablePaths);
}

export const extractAllDownloadablePaths = (manifest: ManifestDocument) => {
  const spec = manifest.spec;
  const components = spec.components ?? [];
  const loras = spec.loras ?? [];
  const allDownloadablePaths = new Set<{type: string, path: string}>();
  for (const component of components) {
    if (component.model_path) {
      if (typeof component.model_path === "string") {
        allDownloadablePaths.add({type: "component", path: component.model_path});
      } else {
        for (const pathItem of component.model_path) {
          if (pathItem.path) {
            allDownloadablePaths.add({type: "component", path: pathItem.path});
          }
        }
      }
      if (component.config_path) {
        allDownloadablePaths.add({type: "component", path: component.config_path});
      }
    }
  }
  for (const lora of loras) {
    if (lora.source) {
      allDownloadablePaths.add({type: "lora", path: lora.source});
    }
  }
  return Array.from(allDownloadablePaths);
}

export const extractAllComponentDownloadingPaths = (component: ManifestComponent) => {
  const modelPath = component.model_path;
  const configPath = component.config_path;
  const allDownloadingPaths = new Set<{type: string, path: string}>();
  if (modelPath) {
    if (typeof modelPath === "string") {
      allDownloadingPaths.add({type: "component", path: modelPath});
    } else {
      for (const pathItem of modelPath) {
        if (pathItem.path && !pathItem.is_downloaded) {
          allDownloadingPaths.add({type: "component", path: pathItem.path});
        }
      }
    }
  }
  if (configPath) {
    allDownloadingPaths.add({type: "component", path: configPath});
  }
  return Array.from(allDownloadingPaths);
}

export const extractAllLoraDownloadingPaths = (lora: LoraType) => {
  const source = lora.source;
  const allDownloadingPaths = new Set<{type: string, path: string}>();
  if (source && !lora.is_downloaded) {
    allDownloadingPaths.add({type: "lora", path: source});
  }
  return Array.from(allDownloadingPaths);
}
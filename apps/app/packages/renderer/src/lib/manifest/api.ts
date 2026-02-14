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
  listManifestGroups as listManifestGroupsPreload,
  getManifestGroup as getManifestGroupPreload,
} from "@app/preload";
import { ClipType } from "../types";
import {
  normalizeModelDownloadProfile,
  selectPreferredModelPathItem,
} from "./model-variant-selection.js";

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
  config_id?: string;
  config?: Record<string, any>;
  [key: string]: any;
};

export type ManifestSchedulerField = {
  label?: string;
  description?: string;
  type?: "number" | "select" | "boolean" | "text" | string;
  value_type?: "integer" | "float" | string;
  min?: number;
  max?: number;
  step?: number;
  default?: any;
  options?: { name: string; value: any }[];
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
  scheduler_fields?: Record<string, ManifestSchedulerField>;
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
      file_size?: number;
      size_bytes?: number;
      filesize?: number;
      size?: number;
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

// ----------------------------- Model Group Types ----------------------------- //

export type ManifestGroupVariant = {
  id: string;
  label: string;
  description?: string;
  manifest_ref: string;
  default?: boolean;
  /** The fully resolved and enriched manifest for this variant.
   *  Populated by the backend when the manifest_ref can be resolved.
   *  May be null/undefined if the referenced manifest could not be loaded. */
  manifest?: ManifestDocument | null;
};

export type ManifestGroupMetadata = {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
  author?: string;
  license?: string;
  demo_path?: string;
  categories?: string[];
};

export type ManifestGroup = {
  api_version: string;
  kind: "ModelGroup";
  type: string;
  metadata: ManifestGroupMetadata;
  variants: ManifestGroupVariant[];
  // Top-level convenience fields (normalized by backend)
  id: string;
  name: string;
  description: string;
  tags: string[];
  categories: string[];
  author: string;
  license: string;
  demo_path: string;
  group_type: string;
  full_path: string;
};

// ----------------------------- Manifest API Functions ----------------------------- //

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

// ----------------------------- Model Group API Functions ----------------------------- //

/**
 * List all manifest groups.
 *
 * Backward compatibility: if the backend does not support groups (older API
 * versions), this gracefully returns an empty list instead of throwing, so
 * callers can fall back to the flat manifest list.
 */
export async function listManifestGroups(): Promise<
  ConfigResponse<ManifestGroup[]>
> {
  try {
    const response = await listManifestGroupsPreload();
    // Older backends may return an error shape rather than throwing
    if (response && response.success === false) {
      return { success: true, data: [] };
    }
    return response as ConfigResponse<ManifestGroup[]>;
  } catch {
    // Backend does not support groups endpoint – degrade gracefully
    return { success: true, data: [] };
  }
}

/**
 * Get a specific manifest group by its id.
 *
 * Backward compatibility: returns a not-found style response when the backend
 * does not support groups.
 */
export async function getManifestGroup(
  groupId: string,
): Promise<ConfigResponse<ManifestGroup>> {
  try {
    const response = await getManifestGroupPreload(groupId);
    if (response && response.success === false) {
      return { success: false, error: response.error ?? "Group not found" };
    }
    return response as ConfigResponse<ManifestGroup>;
  } catch {
    return { success: false, error: "Groups not supported by this API version" };
  }
}

/**
 * List manifest groups filtered by group type (e.g. "video", "image", "audio").
 *
 * This is a client-side convenience filter over listManifestGroups().
 */
export async function listManifestGroupsByType(
  groupType: string,
): Promise<ConfigResponse<ManifestGroup[]>> {
  const response = await listManifestGroups();
  if (!response.success || !response.data) {
    return response;
  }
  const filtered = response.data.filter(
    (g) => g.group_type === groupType || g.type === groupType,
  );
  return { success: true, data: filtered };
}

/**
 * List manifest groups filtered by category (e.g. "text-to-video", "inpaint").
 *
 * This is a client-side convenience filter over listManifestGroups().
 */
export async function listManifestGroupsByCategory(
  category: string,
): Promise<ConfigResponse<ManifestGroup[]>> {
  const response = await listManifestGroups();
  if (!response.success || !response.data) {
    return response;
  }
  const filtered = response.data.filter(
    (g) => g.categories && g.categories.includes(category),
  );
  return { success: true, data: filtered };
}

/**
 * Resolve a group's variant to its full manifest document.
 *
 * Given a group and variant id, looks up the variant's manifest_ref and
 * returns the corresponding manifest. Falls back to the default variant
 * if variantId is not provided.
 *
 * The backend pre-resolves each variant's manifest_ref into a full enriched
 * manifest embedded in `variant.manifest`. If that data is already present
 * we return it directly (no extra fetch). Otherwise (backward compatibility
 * with older backends that don't resolve refs) we fall back to fetching the
 * manifest by its ref via getManifest().
 */
export async function resolveGroupVariantManifest(
  group: ManifestGroup,
  variantId?: string,
): Promise<ConfigResponse<ManifestDocument>> {
  const variants = group.variants ?? [];
  let variant: ManifestGroupVariant | undefined;

  if (variantId) {
    variant = variants.find((v) => v.id === variantId);
  }
  if (!variant) {
    variant = variants.find((v) => v.default === true) ?? variants[0];
  }
  if (!variant) {
    return { success: false, error: "No variants available in this group" };
  }

  // If the backend already resolved the manifest, use it directly
  if (variant.manifest) {
    return { success: true, data: variant.manifest as ManifestDocument };
  }

  // Fallback: fetch by manifest_ref (works with older backends that don't
  // pre-resolve refs, or when the ref could not be resolved server-side)
  const manifestId = variant.manifest_ref;
  return await getManifest(manifestId);
}


export const extractAllDownloadableDefaultPaths = (
  manifest: ManifestDocument,
  options?: { modelDownloadProfile?: string },
) => {
  const spec = manifest.spec;
  const components = spec.components ?? [];
  const loras = spec.loras ?? [];
  const allDownloadablePaths = new Set<{type: string, path: string | string[], index?: number}>();
  const profile = normalizeModelDownloadProfile(
    options?.modelDownloadProfile ?? "auto",
  );
  for (const [index, component] of components.entries()) {
    const componentDownloadablePath = [];
    if (component.is_downloaded) continue;
    if (component.model_path) {
      if (typeof component.model_path === "string") {
        componentDownloadablePath.push({type: "component", path: component.model_path, index});
      } else {
        const candidates = component.model_path.filter(
          (pathItem) =>
            Boolean(pathItem?.path) &&
            !pathItem.is_downloaded &&
            !pathItem.custom,
        );
        const selected = selectPreferredModelPathItem(
          candidates as ManifestComponentModelPathItem[],
          {
            profile,
            componentType: component.type,
            manifestMetadata: manifest.metadata,
          },
        );
        if (selected?.path) {
          componentDownloadablePath.push({
            type: "component",
            path: selected.path,
            index,
          });
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

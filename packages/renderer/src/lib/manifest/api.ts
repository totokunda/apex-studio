import { getBackendApiUrl } from '@/lib/config';

export interface ManifestInfo {
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
}

export interface ModelTypeInfo {
  key: string;
  label: string;
  description: string;
}

/**
 * Get all available manifests from the API
 */
export async function getAllManifests(): Promise<ManifestInfo[]> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return [];
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/list`);
    
    if (!response.ok) {
      console.error('Failed to fetch manifests:', response.statusText);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching manifests:', error);
    return [];
  }
}

/**
 * Get manifests filtered by model type
 */
export async function getManifestsByType(modelType: string): Promise<ManifestInfo[]> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return [];
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/list/type/${modelType}`);
    
    if (!response.ok) {
      console.error('Failed to fetch manifests by type:', response.statusText);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching manifests by type:', error);
    return [];
  }
}

/**
 * Get manifests filtered by model name
 */
export async function getManifestsByModel(model: string): Promise<ManifestInfo[]> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return [];
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/list/model/${model}`);
    
    if (!response.ok) {
      console.error('Failed to fetch manifests by model:', response.statusText);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching manifests by model:', error);
    return [];
  }
}

/**
 * Get all available model types
 */
export async function getModelTypes(): Promise<ModelTypeInfo[]> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return [];
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/types`);
    
    if (!response.ok) {
      console.error('Failed to fetch model types:', response.statusText);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching model types:', error);
    return [];
  }
}

/**
 * Get full manifest content by ID
 */
export async function getManifestContent(manifestId: string): Promise<any> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return null;
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/${manifestId}`);
    
    if (!response.ok) {
      console.error('Failed to fetch manifest content:', response.statusText);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching manifest content:', error);
    return null;
  }
}

export interface FloatingPanelRegion {
  inputs: string[];
}

export interface FloatingPanelConfig {
  regions: {
    media?: FloatingPanelRegion;
    text?: FloatingPanelRegion;
    shortcuts?: FloatingPanelRegion;
  };
}

export interface InputSpec {
  id: string;
  label: string;
  type: string;
  panel: string;
  required?: boolean;
  floating_panel?: boolean;
  description?: string;
  default?: any;
  [key: string]: any;
}

export interface FloatingPanelUIData {
  floating_panel: FloatingPanelConfig;
  inputs: InputSpec[];
}

/**
 * Get floating panel UI configuration for a specific manifest
 */
export async function getFloatingPanelUI(manifestId: string): Promise<FloatingPanelUIData | null> {
  try {
    const backendConfig = await getBackendApiUrl();
    if (!backendConfig.success || !backendConfig.data) {
      console.error('Failed to get backend URL');
      return null;
    }

    const baseUrl = backendConfig.data.url;
    const response = await fetch(`${baseUrl}/manifest/${manifestId}/ui/floating_panel`);
    
    if (!response.ok) {
      console.error('Failed to fetch floating panel UI:', response.statusText);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching floating panel UI:', error);
    return null;
  }
}


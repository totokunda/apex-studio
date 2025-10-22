import { useEffect, useState } from 'react';
import {
  listModelTypes,
  listManifests,
  listManifestsByModel,
  listManifestsByType,
  listManifestsByModelAndType,
  getManifest,
  type ManifestInfo,
  type ModelTypeInfo,
  type ConfigResponse,
} from './api';

type AsyncState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function useAsync<T>(fn: () => Promise<ConfigResponse<T>>, deps: any[] = []): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: null, loading: true, error: null });
  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    fn().then((res) => {
      if (cancelled) return;
      if (res.success) setState({ data: res.data as T, loading: false, error: null });
      else setState({ data: null, loading: false, error: res.error || 'Request failed' });
    }).catch((err: any) => {
      if (cancelled) return;
      setState({ data: null, loading: false, error: err?.message || 'Request error' });
    });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function useManifestTypes() {
  return useAsync<ModelTypeInfo[]>(() => listModelTypes(), []);
}

export function useManifests() {
  return useAsync<ManifestInfo[]>(() => listManifests(), []);
}

export function useManifestsByModel(model: string | null) {
  return useAsync<ManifestInfo[]>(() => model ? listManifestsByModel(model) : Promise.resolve({ success: true, data: [] as any }), [model]);
}

export function useManifestsByType(modelType: string | null) {
  return useAsync<ManifestInfo[]>(() => modelType ? listManifestsByType(modelType) : Promise.resolve({ success: true, data: [] as any }), [modelType]);
}

export function useManifestsByModelAndType(model: string | null, modelType: string | null) {
  return useAsync<ManifestInfo[]>(() => (model && modelType) ? listManifestsByModelAndType(model, modelType) : Promise.resolve({ success: true, data: [] as any }), [model, modelType]);
}

export function useManifest(manifestId: string | null) {
  return useAsync<any>(() => manifestId ? getManifest(manifestId) : Promise.resolve({ success: true, data: null as any }), [manifestId]);
}



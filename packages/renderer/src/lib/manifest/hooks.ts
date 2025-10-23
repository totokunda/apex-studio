import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { ManifestDocument, type ManifestInfo, type ModelTypeInfo } from './api';
import { useManifestStore } from './store';

type AsyncState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
}

// legacy async hook removed; Zustand-based hooks below return AsyncState shape

export function useManifestTypes() {
  const { modelTypes, loading, error } = useManifestStore(useShallow((s) => ({
    modelTypes: s.modelTypes,
    loading: s.loading.modelTypes,
    error: s.error.modelTypes,
  })));
  useEffect(() => {
    if (modelTypes == null && !loading) {
      useManifestStore.getState().loadModelTypes(false);
    }
  }, [modelTypes, loading]);
  return { data: modelTypes, loading, error } as AsyncState<ModelTypeInfo[]>;
}

export function useManifests() {
  const { manifests, loading, error } = useManifestStore(useShallow((s) => ({
    manifests: s.manifests,
    loading: s.loading.manifests,
    error: s.error.manifests,
  })));
  useEffect(() => {
    if (manifests == null && !loading) {
      useManifestStore.getState().loadManifests(false);
    }
  }, [manifests, loading]);
  return { data: manifests, loading, error } as AsyncState<ManifestInfo[]>;
}

export function useManifestsByModel(model: string | null) {
  const { manifestsByModel, loadingMap, errorMap } = useManifestStore(useShallow((s) => ({
    manifestsByModel: s.manifestsByModel,
    loadingMap: s.loading.byModel,
    errorMap: s.error.byModel,
  })));
  const data = model ? (manifestsByModel[model] || undefined) : undefined;
  const isLoading = !!(model && loadingMap[model]);
  useEffect(() => {
    if (model && !data && !isLoading) {
      useManifestStore.getState().loadManifestsByModel(model, false);
    }
  }, [model, data, isLoading]);
  return { data: data || [], loading: isLoading, error: model ? (errorMap[model] || null) : null } as AsyncState<ManifestInfo[]>;
}

export function useManifestsByType(modelType: string | null) {
  const { manifestsByType, loadingMap, errorMap } = useManifestStore(useShallow((s) => ({
    manifestsByType: s.manifestsByType,
    loadingMap: s.loading.byType,
    errorMap: s.error.byType,
  })));
  const data = modelType ? (manifestsByType[modelType] || undefined) : undefined;
  const isLoading = !!(modelType && loadingMap[modelType]);
  useEffect(() => {
    if (modelType && !data && !isLoading) {
      useManifestStore.getState().loadManifestsByType(modelType, false);
    }
  }, [modelType, data, isLoading]);
  return { data: data || [], loading: isLoading, error: modelType ? (errorMap[modelType] || null) : null } as AsyncState<ManifestInfo[]>;
}

export function useManifestsByModelAndType(model: string | null, modelType: string | null) {
  const { dataMap, loadingMap, errorMap } = useManifestStore(useShallow((s) => ({
    dataMap: s.manifestsByModelAndType,
    loadingMap: s.loading.byModelAndType,
    errorMap: s.error.byModelAndType,
  })));
  const data = model && modelType ? (dataMap[model]?.[modelType] || undefined) : undefined;
  const isLoading = !!(model && modelType && loadingMap[model]?.[modelType]);
  useEffect(() => {
    if (model && modelType && !data && !isLoading) {
      useManifestStore.getState().loadManifestsByModelAndType(model, modelType, false);
    }
  }, [model, modelType, data, isLoading]);
  const error = model && modelType ? (errorMap[model]?.[modelType] || null) : null;
  return { data: data || [], loading: isLoading, error } as AsyncState<ManifestInfo[]>;
}

export function useManifest(manifestId: string | null) {
  const { dataMap, loadingMap, errorMap } = useManifestStore(useShallow((s) => ({
    dataMap: s.manifestById,
    loadingMap: s.loading.byId,
    errorMap: s.error.byId,
  })));
  const data = manifestId ? (dataMap[manifestId] ?? undefined) : undefined;
  const isLoading = !!(manifestId && loadingMap[manifestId]);
  useEffect(() => {
    if (manifestId && !data && !isLoading) {
      useManifestStore.getState().loadManifest(manifestId, false);
    }
  }, [manifestId, data, isLoading]);
  const error = manifestId ? (errorMap[manifestId] || null) : null;
  return { data: data ?? null, loading: isLoading, error } as AsyncState<ManifestDocument>;
}



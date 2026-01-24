import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { ManifestDocument, type ModelTypeInfo } from "./api";
import { useManifestStore } from "./store";

type AsyncState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
};

// legacy async hook removed; Zustand-based hooks below return AsyncState shape

export function useManifestTypes() {
  const { modelTypes, loading, error } = useManifestStore(
    useShallow((s) => ({
      modelTypes: s.modelTypes,
      loading: s.loading.modelTypes,
      error: s.error.modelTypes,
    })),
  );
  useEffect(() => {
    if (modelTypes == null && !loading) {
      useManifestStore.getState().loadModelTypes(false);
    }
  }, [modelTypes, loading]);
  return { data: modelTypes, loading, error } as AsyncState<ModelTypeInfo[]>;
}

export function useManifests() {
  const { manifests, loading, error } = useManifestStore(
    useShallow((s) => ({
      manifests: s.manifests,
      loading: s.loading.manifests,
      error: s.error.manifests,
    })),
  );
  useEffect(() => {
    if (manifests == null && !loading) {
      useManifestStore.getState().loadManifests(false);
    }
  }, [manifests, loading]);
  return { data: manifests, loading, error } as AsyncState<ManifestDocument[]>;
}

export function useManifestsByModel(model: string | null) {
  const { manifests, loading, error } = useManifestStore(
    useShallow((s) => ({
      manifests: s.manifests,
      loading: s.loading.manifests,
      error: s.error.manifests,
    })),
  );
  const data = model ? (manifests || [])?.filter((m) => m.model === model) : [];
  const isLoading = loading || manifests == null;
  useEffect(() => {
    // Ensure manifests are fetched when missing.
    if (manifests == null && !loading) {
      useManifestStore.getState().loadManifests(false);
    }
  }, [manifests, loading]);
  return { data, loading: isLoading, error } as AsyncState<ManifestDocument[]>;
}

export function useManifestsByType(modelType: string | null) {
  const { manifests, loading, error } = useManifestStore(
    useShallow((s) => ({
      manifests: s.manifests,
      loading: s.loading.manifests,
      error: s.error.manifests,
    })),
  );
  const data = modelType
    ? (manifests || [])?.filter(
        (m) => Array.isArray(m.model_type) && m.model_type.includes(modelType),
      )
    : [];
  const isLoading = loading || manifests == null;
  useEffect(() => {
    // Ensure manifests are fetched when missing.
    if (manifests == null && !loading) {
      useManifestStore.getState().loadManifests(false);
    }
  }, [manifests, loading]);
  return { data, loading: isLoading, error } as AsyncState<ManifestDocument[]>;
}

export function useManifestsByModelAndType(
  model: string | null,
  modelType: string | null,
) {
  const { manifests, loading, error } = useManifestStore(
    useShallow((s) => ({
      manifests: s.manifests,
      loading: s.loading.manifests,
      error: s.error.manifests,
    })),
  );
  const data =
    model && modelType
      ? (manifests || [])?.filter(
          (m) =>
            m.model === model &&
            Array.isArray(m.model_type) &&
            m.model_type.includes(modelType),
        )
      : [];
  const isLoading = loading || manifests == null;
  useEffect(() => {
    // Ensure manifests are fetched when missing.
    if (manifests == null && !loading) {
      useManifestStore.getState().loadManifests(false);
    }
  }, [manifests, loading]);
  return { data, loading: isLoading, error } as AsyncState<ManifestDocument[]>;
}

export function useManifest(manifestId: string | null) {
  const { manifests, loadingById, errorById } = useManifestStore(
    useShallow((s) => ({
      manifests: s.manifests,
      loadingById: s.loading.byId,
      errorById: s.error.byId,
    })),
  );
  const data = manifestId
    ? (manifests || []).find((m) => (m.metadata?.id || m.id) === manifestId)
    : undefined;
  const isLoading = !!(manifestId && loadingById[manifestId]);
  useEffect(() => {
    if (manifestId && !data && !isLoading) {
      useManifestStore.getState().loadManifest(manifestId, false);
    }
  }, [manifestId, data, isLoading]);
  const error = manifestId ? errorById[manifestId] || null : null;
  return {
    data: (data as ManifestDocument) ?? null,
    loading: isLoading,
    error,
  } as AsyncState<ManifestDocument>;
}

import { QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { getManifest, getManifestPart, listManifests, listManifestGroups, listModelTypes, type ManifestGroup, type ModelTypeInfo } from "./api";
import { ManifestDocument } from "@/lib/manifest";
import _ from "lodash";

export async function fetchManifestsAndPrimeCache(
  queryClient: QueryClient,
): Promise<ManifestDocument[]> {
  const response = await listManifests();
  if (!response.success) {
    throw new Error(
      response.error || "Backend is unavailable (failed to load manifests).",
    );
  }
  const manifests = (response.data ?? []) as ManifestDocument[];
  manifests.forEach((manifest) => {
    const id = manifest?.metadata?.id;
    if (id) {
      queryClient.setQueryData(["manifest", id], manifest);
    }
  });
  return manifests;
}

export async function fetchModelTypes(): Promise<ModelTypeInfo[]> {
  const response = await listModelTypes();
  if (!response.success) {
    throw new Error(
      response.error || "Backend is unavailable (failed to load model types).",
    );
  }
  const data = response.data;
  return (Array.isArray(data) ? data : []) as ModelTypeInfo[];
}

/**
 * Fetch all manifest groups from the backend and prime the per-manifest cache
 * for every resolved variant.  Returns an empty array (rather than throwing)
 * when the backend does not support the groups endpoint, so callers can
 * safely fall back to the flat manifest list.
 */
export async function fetchManifestGroups(
  queryClient: QueryClient,
): Promise<ManifestGroup[]> {
  try {
    const response = await listManifestGroups();
    if (!response.success || !response.data?.length) {
      return [];
    }
    
    const groups = response.data;
    // Prime individual manifest caches from resolved group variants
    for (const group of groups) {
      for (const variant of group.variants ?? []) {
        if (variant.manifest) {
          const id = variant.manifest.metadata?.id;
          if (id) {
            queryClient.setQueryData(["manifest", id], variant.manifest);
          }
        }
      }
    }
    return groups;
  } catch {
    // Backend does not support groups – degrade gracefully
    return [];
  }
}

/**
 * Fetch manifests with group support.
 *
 * Strategy:
 *   1. Fetch groups and flat manifests in parallel.
 *   2. If groups are available, derive the display list from each group's
 *      default variant manifest (one card per group, deduplicating variants).
 *   3. If groups are unavailable (old backend), return the flat manifest list.
 *
 * The flat manifest query ALWAYS runs so that the per-manifest cache stays
 * warm for downstream consumers (ModelPage, useManifestQuery, etc.).
 */
export async function fetchManifestsWithGroupSupport(
  queryClient: QueryClient,
): Promise<{ manifests: ManifestDocument[]; groups: ManifestGroup[] }> {
  const [groups, flatManifests] = await Promise.all([
    fetchManifestGroups(queryClient),
    fetchManifestsAndPrimeCache(queryClient),
  ]);

  if (groups.length > 0) {
    // Build a de-duplicated manifest list from group default variants.
    // Groups that couldn't resolve their manifest_ref fall back to the
    // matching flat manifest (by variant id) if available.
    const groupManifests: ManifestDocument[] = [];
    const seenIds = new Set<string>();

    for (const group of groups) {
      const variants = group.variants ?? [];
      const defaultVariant = variants.find((v) => v.default) ?? variants[0];
      if (!defaultVariant) continue;

      let manifest = defaultVariant.manifest as ManifestDocument | null | undefined;

      // Fallback: try to find the manifest in the flat list by variant id
      if (!manifest && defaultVariant.id) {
        manifest = flatManifests.find(
          (m) => m.metadata?.id === defaultVariant.id,
        );
      }

      if (manifest) {
        const id = manifest.metadata?.id;
        if (id && !seenIds.has(id)) {
          seenIds.add(id);
          groupManifests.push(manifest);
        }
      }
    }

    if (groupManifests.length > 0) {
      return { manifests: groupManifests, groups };
    }
  }

  // Fallback: no groups available, return flat manifests
  return { manifests: flatManifests, groups: [] };
}

export async function prefetchModelMenuQueries(
  queryClient: QueryClient,
): Promise<void> {
  // Prefetch model types and groups in parallel.  The flat manifest list is
  // only fetched as a fallback when groups returns empty (old backend).
  const [, groupsResult] = await Promise.allSettled([
    queryClient.prefetchQuery({
      queryKey: ["modelTypes"],
      queryFn: fetchModelTypes,
      staleTime: 30_000,
    }),
    queryClient.prefetchQuery({
      queryKey: ["manifestGroups"],
      queryFn: () => fetchManifestGroups(queryClient),
      staleTime: 30_000,
    }),
  ]);

  // Only prefetch the flat manifest list when groups didn't return data.
  const groups =
    groupsResult.status === "fulfilled"
      ? queryClient.getQueryData<ManifestGroup[]>(["manifestGroups"])
      : undefined;

  if (!groups || groups.length === 0) {
    await queryClient.prefetchQuery({
      queryKey: ["manifest"],
      queryFn: () => fetchManifestsAndPrimeCache(queryClient),
      staleTime: 30_000,
    });
  }
}

export const useManifestQuery = (manifestId: string | null, forceRefresh: boolean = false) => {
    const queryClient = useQueryClient();   
    return useQuery({
    queryKey: ["manifest", manifestId],
    queryFn: async () => {
      if (!manifestId) return null;
      // check cache as opposed to making a request
      // check if the manifest is already in the cache
      const manifest = queryClient.getQueryData<ManifestDocument>(["manifest", manifestId]);
      if (manifest && !forceRefresh) {
        return manifest;
      }
      const manifests = queryClient.getQueryData<ManifestDocument[]>(["manifest"])
      if (manifests && !forceRefresh) {
        const manifest = manifests.find((m) => m.metadata?.id === manifestId);
        if (manifest) return manifest;
      }
      const response = await getManifest(manifestId);
      if (!response.success) {
        throw new Error(
          response.error || "Backend is unavailable (failed to load manifest).",
        );
      }
      // update the manifest in the cache 
      const manifestIndex = manifests?.findIndex((m) => m.metadata?.id === manifestId);
      if (manifestIndex !== undefined && manifests) {
        const updatedManifests = [...manifests];
        updatedManifests[manifestIndex] = response.data as ManifestDocument;
        queryClient.setQueryData(["manifest"], updatedManifests);

      }
      return response.data ?? null;
    },
    initialData: () => {
      const manifests = queryClient.getQueryData<ManifestDocument[]>(["manifest"]);
      return manifests?.find((m) => m.metadata?.id === manifestId) ?? null;
    },
    placeholderData: null,
    retry: false,
    refetchOnWindowFocus: false,
    enabled: !!manifestId,
  });
};

export const refreshManifest = async (
  manifestId: string | null,
  queryClient: QueryClient,
  invalidateManifestList: boolean = false,
) => {
  if (!manifestId) return null;

  const response = await getManifest(manifestId);
  if (!response.success) {
    throw new Error(
      response.error || "Backend is unavailable (failed to refresh manifest).",
    );
  }

  const manifest = (response.data ?? null) as ManifestDocument | null;
  queryClient.setQueryData(["manifest", manifestId], manifest);

  const manifests = queryClient.getQueryData<ManifestDocument[]>(["manifest"]);
  if (manifests && manifest) {
    const manifestIdx = manifests.findIndex((m) => m.metadata?.id === manifestId);
    if (manifestIdx !== -1) {
      const updated = [...manifests];
      updated[manifestIdx] = manifest;
      queryClient.setQueryData(["manifest"], updated);
    }
    if (invalidateManifestList) {
      await queryClient.invalidateQueries({ queryKey: ["manifest"] });
    }
  }

  await queryClient.invalidateQueries({ queryKey: ["manifest", manifestId] });
  return manifest;
};

export const refreshManifestPart = async (manifestId: string | null, part:string, queryClient: QueryClient, invalidateManifest: boolean = false) => {
      if (!manifestId) return;
      const response = await getManifestPart(manifestId, part);
      // get the 
      if (!response.success) {
        throw new Error(response.error || "Backend is unavailable (failed to refresh manifest part).");
      }
      // update the manifest part in the cache
      
      const manifest = queryClient.getQueryData<ManifestDocument>(["manifest", manifestId]);
      let updatedManifest: ManifestDocument | null = null;

      if (manifest) {
        updatedManifest = _.cloneDeep(manifest);
        _.set(updatedManifest, part, response.data);
        queryClient.setQueryData(["manifest", manifestId], updatedManifest);
        await queryClient.invalidateQueries({ queryKey: ["manifest", manifestId] });
      }
      const manifests = queryClient.getQueryData<ManifestDocument[]>(["manifest"]);
      if (manifests) {
        const manifestIdx = manifests.findIndex((m) => m.metadata?.id === manifestId);
        if (manifestIdx !== -1) {
          updatedManifest = _.cloneDeep(manifests[manifestIdx]);
          _.set(updatedManifest, part, response.data);
          manifests[manifestIdx] = updatedManifest;
          queryClient.setQueryData(["manifest"], manifests);
          if (invalidateManifest) {
            await queryClient.invalidateQueries({ queryKey: ["manifest"] });
          }
        }
      }
      return updatedManifest;
    }

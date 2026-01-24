import { QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { getManifest, getManifestPart, listManifests, listModelTypes, type ModelTypeInfo } from "./api";
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

export async function prefetchModelMenuQueries(
  queryClient: QueryClient,
): Promise<void> {
  await Promise.allSettled([
    queryClient.prefetchQuery({
      queryKey: ["manifest"],
      queryFn: () => fetchManifestsAndPrimeCache(queryClient),
      staleTime: 30_000,
    }),
    queryClient.prefetchQuery({
      queryKey: ["modelTypes"],
      queryFn: fetchModelTypes,
      staleTime: 30_000,
    }),
  ]);
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

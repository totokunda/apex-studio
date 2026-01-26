import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getPreprocessor, listPreprocessors, type Preprocessor } from "./api";

export const PREPROCESSORS_LIST_QUERY_KEY = ["preprocessors", "list"] as const;
export const PREPROCESSOR_QUERY_KEY = (id: string) =>
  ["preprocessors", "get", id] as const;

export async function fetchPreprocessorsList(): Promise<Preprocessor[]> {
  const res = await listPreprocessors(true);
  if (!res.success) {
    throw new Error(res.error || "Failed to load preprocessors");
  }
  return res.data?.preprocessors ?? [];
}

export function usePreprocessorsListQuery(opts?: { enabled?: boolean }) {
  return useQuery({
    queryKey: PREPROCESSORS_LIST_QUERY_KEY,
    queryFn: fetchPreprocessorsList,
    enabled: opts?.enabled ?? true,
    placeholderData: (prev) => prev ?? [],
    retry: true,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });
}

export async function fetchPreprocessor(id: string): Promise<Preprocessor> {
  const res = await getPreprocessor(id);
  if (!res.success) {
    throw new Error(res.error || "Failed to load preprocessor");
  }
  if (!res.data) {
    throw new Error("Failed to load preprocessor");
  }
  return res.data;
}

export function usePreprocessorQuery(
  id: string,
  opts?: { enabled?: boolean; forceFetch?: boolean },
) {
  const queryClient = useQueryClient();
  const cachedSingle = queryClient.getQueryData(PREPROCESSOR_QUERY_KEY(id)) as
    | Preprocessor
    | undefined;
  const cachedList = queryClient.getQueryData(PREPROCESSORS_LIST_QUERY_KEY) as
    | Preprocessor[]
    | undefined;
  const cachedFromList = cachedList?.find((p) => p.id === id);
  const cached = cachedSingle ?? cachedFromList;
  const shouldForceFetch = !!opts?.forceFetch;
  const shouldFetch = !cached || shouldForceFetch;
  return useQuery({
    queryKey: PREPROCESSOR_QUERY_KEY(id),
    queryFn: async () => {
      const data = await fetchPreprocessor(id);
      // Write-through into the list cache so callers relying on the list
      // immediately reflect the latest fetched preprocessor details.
      try {
        queryClient.setQueryData(
          PREPROCESSORS_LIST_QUERY_KEY,
          (prev: Preprocessor[] | undefined) => {
            if (!Array.isArray(prev) || prev.length === 0) return prev;
            let found = false;
            const next = prev.map((p) => {
              if (p.id !== id) return p;
              found = true;
              return { ...p, ...data };
            });
            return found ? next : prev;
          },
        );
      } catch {}
      return data;
    },
    enabled: (opts?.enabled ?? true) && !!id && shouldFetch,
    initialData: cached,
    placeholderData: () => {
      return cached;
    },
    retry: true,
    refetchOnWindowFocus: false,
    staleTime: cached && !shouldForceFetch ? Infinity : 30_000,
  });
}



import { useMutation, type UseMutationOptions } from "@tanstack/react-query";
import { startUnifiedDownload } from "./api";

export type StartUnifiedDownloadRequest = Parameters<typeof startUnifiedDownload>[0];
export type StartUnifiedDownloadResponse = Awaited<
  ReturnType<typeof startUnifiedDownload>
>;

export type StartUnifiedDownloadResult = NonNullable<
  StartUnifiedDownloadResponse["data"]
>;

export const START_UNIFIED_DOWNLOAD_MUTATION_KEY = [
  "download",
  "startUnified",
] as const;

export function useStartUnifiedDownloadMutation(
  opts?: Omit<
    UseMutationOptions<
      StartUnifiedDownloadResult,
      Error,
      StartUnifiedDownloadRequest,
      unknown
    >,
    "mutationKey" | "mutationFn"
  >,
) {
  return useMutation({
    ...(opts ?? {}),
    mutationKey: START_UNIFIED_DOWNLOAD_MUTATION_KEY,
    mutationFn: async (request: StartUnifiedDownloadRequest) => {
      const res = await startUnifiedDownload(request);
      if (!res.success) {
        throw new Error(res.error || "Failed to start download");
      }
      if (!res.data?.job_id) {
        throw new Error("Failed to start download (missing job_id)");
      }
      return res.data;
    },
  });
}



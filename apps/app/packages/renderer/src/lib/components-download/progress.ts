export type NormalizedStatus =
  | "processing"
  | "completed"
  | "complete"
  | "success"
  | "error"
  | "failed"
  | "canceled"
  | "cancelled"
  | "";

export function extractStatus(payload: any): NormalizedStatus {
  try {
    const s = (
      payload?.status ||
      payload?.state ||
      payload?.metadata?.status ||
      ""
    )
      .toString()
      .toLowerCase();
    return (s as NormalizedStatus) || "";
  } catch {
    return "";
  }
}

export function extractPercent(payload: any): number | null {
  try {
    if (!payload || typeof payload !== "object") return null;
    const p =
      typeof payload.progress === "number"
        ? payload.progress
        : typeof payload?.metadata?.progress === "number"
          ? payload.metadata.progress
          : typeof payload.percent === "number"
            ? payload.percent
            : null;
    if (typeof p === "number") {
      const pct = p <= 1 ? p * 100 : p;
      return Math.max(0, Math.min(100, pct));
    }

    const downloaded =
      payload.downloaded ??
      payload.bytes_downloaded ??
      payload.downloaded_bytes ??
      payload.current_bytes ??
      payload?.metadata?.downloaded;
    const total =
      payload.total ??
      payload.bytes_total ??
      payload.total_bytes ??
      payload?.metadata?.total;
    if (
      typeof downloaded === "number" &&
      typeof total === "number" &&
      total > 0
    ) {
      return Math.max(0, Math.min(100, Math.floor((downloaded / total) * 100)));
    }

    const done =
      payload.files_downloaded ??
      payload.completed ??
      payload.done ??
      payload?.metadata?.completed;
    const files =
      payload.total_files ??
      payload.files ??
      payload.total ??
      payload?.metadata?.total_files;
    if (typeof done === "number" && typeof files === "number" && files > 0) {
      return Math.max(0, Math.min(100, Math.floor((done / files) * 100)));
    }
    return null;
  } catch {
    return null;
  }
}

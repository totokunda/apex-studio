export function formatBytes(
  bytes: number | null | undefined,
  decimals = 1,
): string {
  if (bytes == null || bytes === 0) return "0 B";
  if (bytes < 0) return "0 B";

  const k = 1000;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["B", "KB", "MB", "GB", "TB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const size = parseFloat((bytes / Math.pow(k, i)).toFixed(dm));

  return `${size} ${sizes[i]}`;
}

export function formatSpeed(
  bytesPerSecond: number | null | undefined,
): string | null {
  if (bytesPerSecond == null || bytesPerSecond <= 0) return null;
  const formatted = formatBytes(bytesPerSecond, 1);
  if (formatted.includes("undefined")) return null;
  return `${formatted}/s`;
}

export function formatDownloadProgress(
  downloadedBytes: number | null | undefined,
  totalBytes: number | null | undefined,
): string {
  if (downloadedBytes == null && totalBytes == null) return "";
  if (totalBytes == null) return formatBytes(downloadedBytes);
  if (downloadedBytes == null) return `0 B / ${formatBytes(totalBytes)}`;

  return `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}`;
}

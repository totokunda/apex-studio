const MAX_DEFAULT_LEN = 180;

function truncateOneLine(s: string, maxLen: number): string {
  const oneLine = s.replace(/\s+/g, " ").trim();
  if (oneLine.length <= maxLen) return oneLine;
  return `${oneLine.slice(0, Math.max(0, maxLen - 1)).trimEnd()}…`;
}

function shortenPathLike(s: string): string {
  // Shrink very long quoted paths/URIs to the last few segments for readability.
  // Example: "/Users/me/projects/app/node_modules/pkg/file.js" -> "…/pkg/file.js"
  const parts = s.split("/").filter(Boolean);
  if (parts.length <= 3) return s;
  const tail = parts.slice(-3).join("/");
  return `…/${tail}`;
}

export function formatErrMessage(msg: unknown, opts?: { maxLen?: number }): string {
  const maxLen = typeof opts?.maxLen === "number" ? opts.maxLen : MAX_DEFAULT_LEN;
  const raw = typeof msg === "string" ? msg : msg instanceof Error ? msg.message : "";
  const trimmed = raw.trim();
  if (!trimmed) return "Something went wrong.";

  // Normalize newlines, then keep only the meaningful “headline” line(s).
  const normalized = trimmed.replace(/\r\n/g, "\n");
  const lines = normalized
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  // Drop obvious stack-trace lines (e.g. "at foo (file:line)").
  const nonStack = lines.filter((l) => !/^at\s+\S+/i.test(l));
  const headline = (nonStack[0] ?? lines[0] ?? trimmed).replace(
    /^(\w*Error|Error|UnhandledPromiseRejection\w*|Uncaught)\s*:\s*/i,
    "",
  );

  const withShortPaths = headline.replace(/(['"])(\/[^'"]+)\1/g, (_m, q: string, p: string) => {
    const shortened = p.length >= 48 ? shortenPathLike(p) : p;
    return `${q}${shortened}${q}`;
  });

  return truncateOneLine(withShortPaths, maxLen) || "Something went wrong.";
}

export function looksLikeSslCertError(msg: unknown): boolean {
  const s = typeof msg === "string" ? msg : msg instanceof Error ? msg.message : "";
  const t = (s || "").toLowerCase();
  return (
    t.includes("certificate_verify_failed") ||
    t.includes("certificateverificationerror") ||
    t.includes("unable to get local issuer certificate") ||
    t.includes("self signed certificate") ||
    t.includes("self-signed certificate")
  );
}


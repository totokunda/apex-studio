import { useCallback, useEffect, useMemo, useState } from "react";
import { listServerMediaPage } from "@app/preload";

type ServerMediaType = "generations" | "processors";
type MediaType = "image" | "video" | "audio";

export function useServerMediaHasAny(opts: {
  folderUuid?: string;
  type: ServerMediaType;
  allowedTypes: MediaType[];
  enabled?: boolean;
}) {
  const { folderUuid, type, allowedTypes, enabled = true } = opts;
  const [hasAny, setHasAny] = useState<boolean | null>(null);

  const allowed = useMemo(() => new Set(allowedTypes), [allowedTypes]);

  const refresh = useCallback(async () => {
    if (!enabled) return;
    setHasAny(null);
    try {
      const page = await listServerMediaPage({
        folderUuid,
        cursor: null,
        limit: 10,
        type,
        sortKey: "date",
        sortOrder: "desc",
      });
      const items = (page?.items ?? []).filter((it) =>
        allowed.has(it.type as any),
      );
      setHasAny(items.length > 0);
    } catch {
      setHasAny(false);
    }
  }, [enabled, folderUuid, type, allowed]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!enabled) return;
      try {
        const page = await listServerMediaPage({
          folderUuid,
          cursor: null,
          limit: 10,
          type,
          sortKey: "date",
          sortOrder: "desc",
        });
        if (cancelled) return;
        const items = (page?.items ?? []).filter((it) =>
          allowed.has(it.type as any),
        );
        setHasAny(items.length > 0);
      } catch {
        if (!cancelled) setHasAny(false);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [enabled, folderUuid, type, allowed]);

  // Keep in sync with engine/job refresh events (same event GenerationsMenu listens to).
  useEffect(() => {
    if (!enabled) return;
    const handler = () => {
      void refresh();
    };
    try {
      window.addEventListener("generations-menu-reload", handler as any);
    } catch {
      // ignore
    }
    return () => {
      try {
        window.removeEventListener("generations-menu-reload", handler as any);
      } catch {
        // ignore
      }
    };
  }, [enabled, refresh]);

  return { hasAny, refresh };
}


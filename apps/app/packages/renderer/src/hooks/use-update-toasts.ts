import { useEffect, useRef } from "react";
import { toast } from "sonner";
import {
  type AppUpdateEvent,
  type AppUpdateState,
  checkForAppUpdates,
  downloadAppUpdate,
  getAppUpdateState,
  installAppUpdate,
  onAppUpdateEvent,
  type ApiUpdateEvent,
  type ApiUpdateState,
  applyApiUpdate,
  checkForApiUpdates,
  getApiUpdateState,
  onApiUpdateEvent,
} from "@app/preload";

const REMIND_EVERY_MS = 24 * 60 * 60 * 1000;

function isAppUpdateRelevant(st: AppUpdateState | null | undefined): boolean {
  return st?.status === "available" || st?.status === "downloaded";
}

function isApiUpdateRelevant(st: ApiUpdateState | null | undefined): boolean {
  return st?.status === "available";
}

function formatErrMessage(msg: unknown): string {
  const s = typeof msg === "string" ? msg : msg instanceof Error ? msg.message : "";
  return s.trim() || "Something went wrong.";
}

function safeSetLocalStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export function useUpdateToasts(): void {
  const mountedRef = useRef(false);
  const lastShownRef = useRef<{ appKey: string | null; apiKey: string | null }>({
    appKey: null,
    apiKey: null,
  });

  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;

    let cancelled = false;

    const showAppToastFromState = (st: AppUpdateState, origin: "startup" | "event" | "reminder") => {
      if (!isAppUpdateRelevant(st)) return;
      const key = `${st.status}`;
      // Avoid spamming identical state changes except on startup/reminder.
      if (origin === "event" && lastShownRef.current.appKey === key) return;
      lastShownRef.current.appKey = key;
      safeSetLocalStorage("apex:lastShown:app-update", String(Date.now()));

      if (st.status === "downloaded") {
        toast.success("App update ready", {
          id: "app-update-toast",
          description: "Restart Apex Studio to apply the update.",
          action: {
            label: "Restart",
            onClick: () => {
              void installAppUpdate();
            },
          },
          duration: Infinity,
        });
        return;
      }

      toast.info("App update available", {
        id: "app-update-toast",
        description: "Download the update, then restart Apex Studio to apply it.",
        action: {
          label: "Download",
          onClick: () => {
            void downloadAppUpdate();
          },
        },
        duration: Infinity,
      });
    };

    const showApiToastFromState = (st: ApiUpdateState, origin: "startup" | "event" | "reminder") => {
      if (!isApiUpdateRelevant(st)) return;
      const key = `${st.status}`;
      if (origin === "event" && lastShownRef.current.apiKey === key) return;
      lastShownRef.current.apiKey = key;
      safeSetLocalStorage("apex:lastShown:api-update", String(Date.now()));

      toast.info("Engine update available", {
        id: "api-update-toast",
        description: "Update and restart the engine to apply the latest backend.",
        action: {
          label: "Update & restart",
          onClick: () => {
            void applyApiUpdate();
          },
        },
        duration: Infinity,
      });
    };

    const refreshAndMaybeToast = async (origin: "startup" | "reminder") => {
      // Fetch current snapshots first (fast), then run checks (slower) in the background.
      try {
        const st = await getAppUpdateState();
        if (!cancelled) showAppToastFromState(st, origin);
      } catch {}

      try {
        const st = await getApiUpdateState();
        if (!cancelled) showApiToastFromState(st, origin);
      } catch {}

      // Kick checks to ensure we show on startup when updates exist.
      void (async () => {
        try {
          await checkForAppUpdates();
          const st = await getAppUpdateState();
          if (!cancelled) showAppToastFromState(st, origin);
        } catch {}
      })();

      void (async () => {
        try {
          await checkForApiUpdates();
          const st = await getApiUpdateState();
          if (!cancelled) showApiToastFromState(st, origin);
        } catch {}
      })();
    };

    // Startup behavior: always check and show if update is available.
    void refreshAndMaybeToast("startup");

    const offApp = onAppUpdateEvent((ev: AppUpdateEvent) => {
      if (cancelled) return;
      if (ev.type === "available" || ev.type === "downloaded" || ev.type === "not-available") {
        void (async () => {
          try {
            const st = await getAppUpdateState();
            showAppToastFromState(st, "event");
          } catch {}
        })();
        return;
      }
      if (ev.type === "error") {
        toast.error("App update error", {
          id: "app-update-toast-error",
          description: formatErrMessage(ev.message),
        });
      }
    });

    const offApi = onApiUpdateEvent((ev: ApiUpdateEvent) => {
      if (cancelled) return;
      if (ev.type === "available" || ev.type === "not-available") {
        void (async () => {
          try {
            const st = await getApiUpdateState();
            showApiToastFromState(st, "event");
          } catch {}
        })();
        return;
      }
      if (ev.type === "updated") {
        toast.success("Engine updated", {
          id: "api-update-toast",
          description: "The engine was updated and restarted.",
        });
        return;
      }
      if (ev.type === "error") {
        toast.error("Engine update error", {
          id: "api-update-toast-error",
          description: formatErrMessage(ev.message),
        });
      }
    });

    const reminderId = setInterval(() => {
      if (cancelled) return;
      void refreshAndMaybeToast("reminder");
    }, REMIND_EVERY_MS);

    return () => {
      cancelled = true;
      try {
        offApp();
      } catch {}
      try {
        offApi();
      } catch {}
      clearInterval(reminderId);
    };
  }, []);
}


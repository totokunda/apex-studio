import { createElement, useEffect, useRef } from "react";
import { toast } from "sonner";
import {
  LuArrowRight,
  LuCheck,
  LuDownload,
  LuRefreshCcw,
  LuSparkles,
  LuX,
} from "react-icons/lu";
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
  suppressApiUpdateToast,
} from "@app/preload";

const REMIND_EVERY_MS = 24 * 60 * 60 * 1000;
const SUPPRESS_API_TOAST_MS = 12 * 60 * 60 * 1000;

function isAppUpdateRelevant(st: AppUpdateState | null | undefined): boolean {
  return st?.status === "available" || st?.status === "downloaded";
}

function isApiUpdateRelevant(st: ApiUpdateState | null | undefined): boolean {
  if (st?.status !== "available") return false;
  return (st.toastSuppressedUntil ?? 0) <= Date.now();
}

const MAX_TOAST_ERR_LEN = 180;

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

function formatErrMessage(msg: unknown): string {
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

  return truncateOneLine(withShortPaths, MAX_TOAST_ERR_LEN) || "Something went wrong.";
}

const API_UPDATE_TOAST_ID = "api-update-toast";

function safeSetLocalStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export function useUpdateToasts(): void {
  const lastShownRef = useRef<{ appKey: string | null; apiKey: string | null }>({
    appKey: null,
    apiKey: null,
  });
  const apiModeRef = useRef<"idle" | "available" | "updating">("idle");
  const apiProgressRef = useRef<{
    stage?: "stopping" | "downloading" | "applying" | "restarting";
    percent?: number;
    message?: string;
  }>({});

  useEffect(() => {
    let cancelled = false;
    const timers = new Set<ReturnType<typeof setTimeout>>();

    const showUpdateToast = (opts: {
      id: string;
      title: string;
      description: string;
      icon: "sparkles" | "download" | "refresh";
      actionLabel: string;
      onAction: () => void;
      onDismissClick?: () => void;
    }) => {
      toast.custom(
        () => {
          const iconNode =
            opts.icon === "download"
              ? createElement(LuDownload, { className: "h-[18px] w-[18px]" })
              : opts.icon === "refresh"
                ? createElement(LuRefreshCcw, { className: "h-[18px] w-[18px]" })
                : createElement(LuSparkles, { className: "h-[18px] w-[18px]" });

          return createElement(
            "div",
            { className: "w-full max-w-[420px] p-4 " },
            createElement(
              "div",
              { className: "flex items-start gap-3 " },
              createElement(
                "div",
                {
                  className:
                    "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-brand-accent-shade/30 via-brand-accent-shade/10 to-transparent p-2 text-brand-light",
                },
                iconNode,
              ),
              createElement(
                "div",
                { className: "min-w-0 flex-1" },
                createElement(
                  "div",
                  {
                    className:
                      "text-[12.5px] font-semibold tracking-tight text-brand-light",
                  },
                  opts.title,
                ),
                createElement(
                  "div",
                  {
                    className:
                      "mt-1 text-[10.5px] leading-snug text-brand-light/70",
                  },
                  opts.description,
                ),
                createElement(
                  "div",
                  { className: "mt-3 flex items-center justify-end gap-2" },
                  createElement(
                    "button",
                    {
                      type: "button",
                      className:
                        "h-8 rounded-[6px] border border-white/10 bg-white/5 px-3 text-[10.5px] font-medium text-brand-light/80 hover:bg-white/10",
                      onClick: () => {
                        try {
                          opts.onDismissClick?.();
                        } catch {
                          // ignore
                        }
                        toast.dismiss(opts.id);
                      },
                    },
                    "Dismiss",
                  ),
                  createElement(
                    "button",
                    {
                      type: "button",
                      className:
                        "h-8 rounded-[6px] bg-brand-accent-shade px-3 text-[10.5px] font-semibold text-brand-light hover:bg-brand-accent",
                      onClick: () => opts.onAction(),
                    },
                    createElement(
                      "span",
                      { className: "inline-flex items-center gap-1.5" },
                      opts.actionLabel,
                      createElement(LuArrowRight, { className: "h-3.5 w-3.5" }),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
        ({
          id: opts.id,
          duration: Infinity,
          important: true,
          dismissible: false,
          className:
            "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
        } as any),
      );
    };

    const showApiUpdatingToast = () => {
      apiModeRef.current = "updating";
      toast.custom(
        () => {
          const p = apiProgressRef.current;
          const pct = typeof p.percent === "number" ? Math.max(0, Math.min(100, p.percent)) : null;
          const desc =
            p.message ||
            (p.stage === "stopping"
              ? "Stopping engine…"
              : p.stage === "downloading"
                ? pct != null
                  ? `Downloading update… ${pct.toFixed(1)}%`
                  : "Downloading update…"
                : p.stage === "applying"
                  ? "Applying update…"
                  : "Restarting engine…");
          const iconNode = createElement(LuRefreshCcw, {
            className: "h-[18px] w-[18px] animate-spin",
          });

          const progressWidth = pct != null ? `${pct}%` : "35%";
          const progressClass =
            pct != null
              ? "h-full rounded-[999px] bg-linear-to-r from-brand-accent-shade via-brand-accent to-brand-accent-shade transition-[width] duration-200"
              : "h-full w-[35%] rounded-[999px] bg-linear-to-r from-brand-accent-shade via-brand-accent to-brand-accent-shade animate-pulse";

          return createElement(
            "div",
            { className: "w-full max-w-[420px] p-4 " },
            createElement(
              "div",
              { className: "flex items-start gap-3 " },
              createElement(
                "div",
                {
                  className:
                    "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-brand-accent-shade/30 via-brand-accent-shade/10 to-transparent p-2 text-brand-light",
                },
                iconNode,
              ),
              createElement(
                "div",
                { className: "min-w-0 flex-1" },
                createElement(
                  "div",
                  { className: "text-[12.5px] font-semibold tracking-tight text-brand-light" },
                  "Updating engine…",
                ),
                createElement(
                  "div",
                  { className: "mt-1 text-[10.5px] leading-snug text-brand-light/70" },
                  desc,
                ),
                createElement(
                  "div",
                  { className: "mt-3" },
                  createElement(
                    "div",
                    {
                      className:
                        "h-2 w-full overflow-hidden rounded-[999px] border border-white/10 bg-white/5",
                    },
                    createElement("div", {
                      className: progressClass,
                      style: { width: progressWidth },
                    }),
                  ),
                  createElement(
                    "div",
                    { className: "mt-2 text-[10px] text-brand-light/50" },
                    "Please keep Apex Studio open while the engine updates.",
                  ),
                ),
              ),
            ),
          );
        },
        ({
          id: API_UPDATE_TOAST_ID,
          duration: Infinity,
          dismissible: false,
          className:
            "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
        } as any),
      );
    };

    const showAppToastFromState = (st: AppUpdateState, origin: "startup" | "event" | "reminder") => {
      if (!isAppUpdateRelevant(st)) return;
      const key = `${st.status}`;
      // Avoid spamming identical state changes except on startup/reminder.
      if (origin === "event" && lastShownRef.current.appKey === key) return;
      lastShownRef.current.appKey = key;
      safeSetLocalStorage("apex:lastShown:app-update", String(Date.now()));

      if (st.status === "downloaded") {
        showUpdateToast({
          id: "app-update-toast",
          title: "App update ready",
          description: "Restart Apex Studio to apply the update.",
          icon: "sparkles",
          actionLabel: "Restart",
          onAction: () => {
            void installAppUpdate();
          },
        });
        return;
      }

      showUpdateToast({
        id: "app-update-toast",
        title: "App update available",
        description: "Download the update, then restart Apex Studio to apply it.",
        icon: "download",
        actionLabel: "Download",
        onAction: () => {
          void downloadAppUpdate();
        },
      });
    };

    const showApiToastFromState = (st: ApiUpdateState, origin: "startup" | "event" | "reminder") => {
      // If we're actively updating, never show the "available" prompt again.
      if (apiModeRef.current === "updating") return;
      if (!isApiUpdateRelevant(st)) return;
      const key = `${st.status}`;
      if (origin === "event" && lastShownRef.current.apiKey === key) return;
      lastShownRef.current.apiKey = key;
      safeSetLocalStorage("apex:lastShown:api-update", String(Date.now()));

      apiModeRef.current = "available";
      toast.custom(
        () => {
          const iconNode = createElement(LuRefreshCcw, { className: "h-[18px] w-[18px]" });
          return createElement(
            "div",
            { className: "w-full max-w-[420px] p-4 " },
            createElement(
              "div",
              { className: "flex items-start gap-3 " },
              createElement(
                "div",
                {
                  className:
                    "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-brand-accent-shade/30 via-brand-accent-shade/10 to-transparent p-2 text-brand-light",
                },
                iconNode,
              ),
              createElement(
                "div",
                { className: "min-w-0 flex-1" },
                createElement(
                  "div",
                  { className: "text-[12.5px] font-semibold tracking-tight text-brand-light" },
                  "Engine update available",
                ),
                createElement(
                  "div",
                  { className: "mt-1 text-[10.5px] leading-snug text-brand-light/70" },
                  "Update and restart the engine to apply the latest backend.",
                ),
                createElement(
                  "div",
                  { className: "mt-3 flex items-center justify-end gap-2" },
                  createElement(
                    "button",
                    {
                      type: "button",
                      className:
                        "h-8 rounded-[6px] border border-white/10 bg-white/5 px-3 text-[10.5px] font-medium text-brand-light/80 hover:bg-white/10",
                      onClick: () => {
                        try {
                          void suppressApiUpdateToast(SUPPRESS_API_TOAST_MS);
                        } catch {
                          // ignore
                        }
                        toast.dismiss(API_UPDATE_TOAST_ID);
                      },
                    },
                    "Dismiss",
                  ),
                  createElement(
                    "button",
                    {
                      type: "button",
                      className:
                        "h-8 rounded-[6px] bg-brand-accent-shade px-3 text-[10.5px] font-semibold text-brand-light hover:bg-brand-accent",
                      onClick: () => {
                        // Flip to updating UI immediately (single toast id = single state).
                        apiProgressRef.current = { stage: "stopping", percent: 0, message: "Stopping engine…" };
                        showApiUpdatingToast();
                        // Start update.
                        void applyApiUpdate().catch((e) => {
                          apiModeRef.current = "idle";
                          toast.custom(
                            () => {
                              const iconErr = createElement(LuX, { className: "h-[18px] w-[18px]" });
                              return createElement(
                                "div",
                                { className: "w-full max-w-[420px] p-4 " },
                                createElement(
                                  "div",
                                  { className: "flex items-start gap-3 " },
                                  createElement(
                                    "div",
                                    {
                                      className:
                                        "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-red-500/20 via-red-500/10 to-transparent p-2 text-brand-light",
                                    },
                                    iconErr,
                                  ),
                                  createElement(
                                    "div",
                                    { className: "min-w-0 flex-1" },
                                    createElement(
                                      "div",
                                      { className: "text-[12.5px] font-semibold tracking-tight text-brand-light" },
                                      "Engine update error",
                                    ),
                                    createElement(
                                      "div",
                                      { className: "mt-1 text-[10.5px] leading-snug text-brand-light/70" },
                                      formatErrMessage(e),
                                    ),
                                  ),
                                ),
                              );
                            },
                            {
                              id: API_UPDATE_TOAST_ID,
                              duration: 6000,
                              dismissible: true,
                              className:
                                "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
                            } as any,
                          );
                        });
                      },
                    },
                    createElement(
                      "span",
                      { className: "inline-flex items-center gap-1.5" },
                      "Update & restart",
                      createElement(LuArrowRight, { className: "h-3.5 w-3.5" }),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
        ({
          id: API_UPDATE_TOAST_ID,
          duration: Infinity,
          dismissible: false,
          className:
            "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
        } as any),
      );
    };

    const refreshAndMaybeToast = async (origin: "startup" | "reminder") => {
      // Fetch current snapshots first (fast), then run checks (slower) in the background.
      try {
        const st = await getAppUpdateState();
        if (!cancelled) showAppToastFromState(st, origin);
      } catch (e) {
      }

      try {
        const st = await getApiUpdateState();
        if (cancelled) return;
        // If another window already started updating, show the updating toast immediately.
        if (st?.status === "updating") {
          apiProgressRef.current = st.updateProgress ?? {
            stage: "stopping",
            percent: 0,
            message: "Stopping engine…",
          };
          // Render the updating toast (single toast id) for continuity across windows.
          showApiUpdatingToast();
        } else {
          showApiToastFromState(st, origin);
        }
      } catch (e) {
      }

      // Kick checks to ensure we show on startup when updates exist.
      void (async () => {
        try {
          await checkForAppUpdates();
          const st = await getAppUpdateState();
          if (!cancelled) showAppToastFromState(st, origin);
        } catch (e) {
        }
      })();

      void (async () => {
        try {
          await checkForApiUpdates();
          const st = await getApiUpdateState();
          if (!cancelled) showApiToastFromState(st, origin);
        } catch (e) {
        }
      })();
    };

    refreshAndMaybeToast("startup");

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
        // While updating, ignore "available" refreshes (main may re-check in background).
        if (apiModeRef.current === "updating") return;
        void (async () => {
          try {
            const st = await getApiUpdateState();
            showApiToastFromState(st, "event");
          } catch {}
        })();
        return;
      }
      if (ev.type === "updating") {
        apiProgressRef.current = { stage: "stopping", percent: 0, message: "Stopping engine…" };
        showApiUpdatingToast();
        return;
      }
      if (ev.type === "progress") {
        apiModeRef.current = "updating";
        apiProgressRef.current = { stage: ev.stage, percent: ev.percent, message: ev.message };
        // Re-render the updating toast (single id, single state).
        showApiUpdatingToast();
        return;
      }
      if (ev.type === "updated") {
        apiModeRef.current = "idle";
        toast.custom(
          () => {
            const iconOk = createElement(LuCheck, { className: "h-[18px] w-[18px]" });
            return createElement(
              "div",
              { className: "w-full max-w-[420px] p-4 " },
              createElement(
                "div",
                { className: "flex items-start gap-3 " },
                createElement(
                  "div",
                  {
                    className:
                      "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-emerald-500/20 via-emerald-500/10 to-transparent p-2 text-brand-light",
                  },
                  iconOk,
                ),
                createElement(
                  "div",
                  { className: "min-w-0 flex-1" },
                  createElement(
                    "div",
                    { className: "text-[12.5px] font-semibold tracking-tight text-brand-light" },
                    "Engine updated",
                  ),
                  createElement(
                    "div",
                    { className: "mt-1 text-[10.5px] leading-snug text-brand-light/70" },
                    "The engine was updated and restarted.",
                  ),
                ),
              ),
            );
          },
          ({
            id: API_UPDATE_TOAST_ID,
            duration: 4500,
            dismissible: true,
            className:
              "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
          } as any),
        );
        return;
      }
      if (ev.type === "error") {
        apiModeRef.current = "idle";
        toast.custom(
          () => {
            const iconErr = createElement(LuX, { className: "h-[18px] w-[18px]" });
            return createElement(
              "div",
              { className: "w-full max-w-[420px] p-4 " },
              createElement(
                "div",
                { className: "flex items-start gap-3 " },
                createElement(
                  "div",
                  {
                    className:
                      "mt-0.5 shrink-0 rounded-md border border-white/10 bg-linear-to-br from-red-500/20 via-red-500/10 to-transparent p-2 text-brand-light",
                  },
                  iconErr,
                ),
                createElement(
                  "div",
                  { className: "min-w-0 flex-1" },
                  createElement(
                    "div",
                    { className: "text-[12.5px] font-semibold tracking-tight text-brand-light" },
                    "Engine update error",
                  ),
                  createElement(
                    "div",
                    { className: "mt-1 text-[10.5px] leading-snug text-brand-light/70" },
                    formatErrMessage(ev.message),
                  ),
                ),
              ),
            );
          },
          ({
            id: API_UPDATE_TOAST_ID,
            duration: 6000,
            dismissible: true,
            className:
              "!p-0 !text-start !shadow-xl !border !border-white/10 !bg-linear-to-br from-slate-950 via-black to-slate-900",
          } as any),
        );
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
      for (const t of timers) clearTimeout(t);
      timers.clear();
      clearInterval(reminderId);
    };
  }, []);
}


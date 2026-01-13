import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PiRocketLaunchFill } from "react-icons/pi";
import {
  launchMainWindow,
  listProjects,
  setActiveProjectId,
  getUserDataPath,
  readFileBuffer,
  createProject,
  deleteProject,
  getLauncherStatus,
  onLauncherStatusChange,
  startLauncherStatusWatch,
  stopLauncherStatusWatch,
} from "@app/preload";
import Installer from "@/components/Installer";
import { ProjectSettings, useProjectsStore } from "@/lib/projects";
import {
  LuFolder,
  LuClock,
  LuTrash,
  LuPlus,
  LuRotateCcw,
  LuCheck,
  LuX,
  LuChevronDown,
  LuChevronUp,
  LuCopy,
} from "react-icons/lu";
import { toast } from "sonner";
import { DEFAULT_FPS } from "@/lib/settings";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

let USER_DATA_DIR_CACHE: string | null = null;
let USER_DATA_DIR_IN_FLIGHT: Promise<string | null> | null = null;

async function getUserDataDirCached(): Promise<string | null> {
  if (USER_DATA_DIR_CACHE) return USER_DATA_DIR_CACHE;
  if (USER_DATA_DIR_IN_FLIGHT) return await USER_DATA_DIR_IN_FLIGHT;
  USER_DATA_DIR_IN_FLIGHT = (async () => {
    try {
      const res = await getUserDataPath();
      const dir =
        res?.success && res.data?.user_data ? String(res.data.user_data) : null;
      USER_DATA_DIR_CACHE = dir;
      return dir;
    } catch {
      USER_DATA_DIR_CACHE = null;
      return null;
    } finally {
      USER_DATA_DIR_IN_FLIGHT = null;
    }
  })();
  return await USER_DATA_DIR_IN_FLIGHT;
}

function coverPathToAppUserDataUrl(
  absPath: string,
  userDataDir: string,
  version?: string | number,
): string | null {
  const rawAbs = String(absPath || "");
  const rawBase = String(userDataDir || "");
  if (!rawAbs || !rawBase) return null;

  const isWindowsDrivePath = /^[a-zA-Z]:[\\/]/.test(rawAbs) || /^[a-zA-Z]:[\\/]/.test(rawBase);
  const normalizeForUrlPath = (p: string) => p.replace(/\\/g, "/");
  const stripTrailingSlashes = (p: string) => p.replace(/[\\/]+$/g, "");
  const normalizeForCompare = (p: string) => {
    const stripped = stripTrailingSlashes(p);
    const slashed = normalizeForUrlPath(stripped);
    // Windows paths are case-insensitive; normalize drive letter and rest to lower for comparison.
    return isWindowsDrivePath ? slashed.toLowerCase() : slashed;
  };

  const absForCompare = normalizeForCompare(rawAbs);
  const baseForCompare = normalizeForCompare(rawBase);

  // Make sure we only accept paths inside the user data directory.
  if (
    absForCompare !== baseForCompare &&
    !absForCompare.startsWith(`${baseForCompare}/`)
  ) {
    return null;
  }

  // Compute relative path using the *normalized slash* version so URL paths are correct on Windows.
  const absForUrl = normalizeForUrlPath(stripTrailingSlashes(rawAbs));
  const baseForUrl = normalizeForUrlPath(stripTrailingSlashes(rawBase));
  const rel = absForUrl.startsWith(baseForUrl) ? absForUrl.slice(baseForUrl.length) : "";
  const pathPart = rel.startsWith("/") ? rel : `/${rel}`;
  const v =
    version != null && String(version).length > 0
      ? `?v=${encodeURIComponent(String(version))}`
      : "";
  return `app://user-data${encodeURI(pathPart)}${v}`;
}


function coerceEpochMs(value: unknown): number | null {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return null;
  // Heuristic: treat 10-digit values as seconds.
  return n < 1_000_000_000_000 ? Math.round(n * 1000) : Math.round(n);
}

function formatDateShort(epochMs: number): string {
  const d = new Date(epochMs);
  if (Number.isNaN(d.getTime())) {
    // Absolute fallback; should never happen once coercion works.
    return new Date(Date.now()).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}


const ProjectCard: React.FC<{ project: ProjectSettings }> = ({ project }) => {
  const [coverSrc, setCoverSrc] = useState<string | null>(null);

  // Convert coverPath (absolute filesystem path) into a renderer-loadable URL.
  // We prefer app://user-data/... (served by AppDirProtocol) to avoid Electron blocking file://.
  useEffect(() => {
    let cancelled = false;
    let previousBlobUrl: string | null = null;

    const run = async () => {
      const coverPath = project.coverPath;
      if (!coverPath) {
        setCoverSrc(null);
        return;
      }

      const version =
        coerceEpochMs(project.lastModified) ??
        coerceEpochMs(project.createdAt) ??
        Date.now();

      // If already an app:// URL (future-proof), just use it.
      if (coverPath.startsWith("app://")) {
        const glue = coverPath.includes("?") ? "&" : "?";
        setCoverSrc(`${coverPath}${glue}v=${encodeURIComponent(String(version))}`);
        return;
      }

      const userDataDir = await getUserDataDirCached();
      if (cancelled) return;

      if (userDataDir) {
        const appUrl = coverPathToAppUserDataUrl(coverPath, userDataDir, version);
        if (appUrl) {
          setCoverSrc(appUrl);
          return;
        }
      }

      // Fallback: read bytes and create a blob URL.
      try {
        const buf = await readFileBuffer(coverPath);
        if (cancelled) return;
        const blob = new Blob([new Uint8Array(buf)], { type: "image/jpeg" });
        const blobUrl = URL.createObjectURL(blob);
        previousBlobUrl = blobUrl;
        setCoverSrc(blobUrl);
      } catch {
        setCoverSrc(null);
      }
    };

    void run();
    return () => {
      cancelled = true;
      if (previousBlobUrl) {
        try {
          URL.revokeObjectURL(previousBlobUrl);
        } catch {
          // ignore
        }
      }
    };
  }, [project.coverPath, project.lastModified, project.createdAt]);

  const dateEpoch =
    coerceEpochMs(project.lastModified) ??
    coerceEpochMs(project.createdAt) ??
    Date.now();
  const dateStr = formatDateShort(dateEpoch);

  return (
    <div 
      className="group relative flex shadow flex-col w-full h-fit bg-brand-background/80 hover:bg-brand-background/70 rounded-lg p-3 border border-brand-light/5 hover:border-brand-light/20 transition-all duration-200 cursor-pointer"
      onClick={() => {
        // When clicked, set as active project and launch
        setActiveProjectId(project.id);
        launchMainWindow();
      }}
    >
      <div className="w-full aspect-video bg-black/40 rounded-md overflow-hidden relative mb-3">
        {coverSrc ? (
          <img src={coverSrc} alt={project.name} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-brand-light/10 group-hover:text-brand-light/20 transition-colors">
            <LuFolder className="w-12 h-12" />
          </div>
        )}
      </div>
      <div className="flex flex-col px-1 gap-1">
        <span className="text-[13px] text-brand-light truncate text-start">
          {project.name}
        </span>
        <div className="flex items-center gap-1.5 text-brand-light/40">
           <LuClock className="w-3 h-3" />
           <span className="text-[10px]">{dateStr}</span>
        </div>
        
      </div>
    </div>
  );
};

const Launcher: React.FC = () => {
  const [isLaunching, setIsLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  const [backendConnected, setBackendConnected] = useState(false);
  const [runtimeAvailable, setRuntimeAvailable] = useState(false);
  const [backendStarting, setBackendStarting] = useState(false);
  const initialCheckDoneRef = useRef(false);
  const [showInstaller, setShowInstaller] = useState(false);
  const [showBlockingDetails, setShowBlockingDetails] = useState(false);
  const projects = useProjectsStore((s) => s.projects);
  const setProjects = useProjectsStore((s) => s.setProjects);
  const removeProject = useProjectsStore((s) => s.removeProject);
  const addProject = useProjectsStore((s) => s.addProject);

  const [createOpen, setCreateOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  const [deleteOpen, setDeleteOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<ProjectSettings | null>(
    null,
  );
  const [isDeletingProject, setIsDeletingProject] = useState(false);

  const getNextDefaultProjectName = () => {
    const usedNumbers = projects
      .map((p) => {
        const match = /^Project\s+(\d+)$/i.exec(p.name || "");
        return match ? Number(match[1]) : null;
      })
      .filter((n): n is number => n !== null);

    let candidate = 1;
    while (usedNumbers.includes(candidate)) {
      candidate += 1;
    }
    return `Project ${candidate}`;
  };

  const hasBackend = useMemo(() => {
    return backendConnected || runtimeAvailable || backendStarting;
  }, [backendConnected, runtimeAvailable, backendStarting]);

  const canLaunch = useMemo(() => {
    const hasProjects = projects.length > 0;
    return hasBackend && hasProjects;
  }, [hasBackend, projects.length]);

  const applyStatus = useCallback((st: any) => {
    const hasBackend = Boolean(st?.hasBackend);
    const canStartLocal = Boolean(st?.canStartLocal);
    const starting = Boolean(st?.backendStarting);
    setBackendConnected(hasBackend);
    setRuntimeAvailable(canStartLocal);
    setBackendStarting(starting);
    setError(
      st?.lastError
        ? String(st.lastError)
        : st?.runtimeVerified?.ok === false && st?.runtimeVerified?.reason
          ? String(st.runtimeVerified.reason)
          : null,
    );
  }, []);

  const refreshLauncherStatus = useCallback(
    async ({ showBlocking = false }: { showBlocking?: boolean } = {}) => {
      if (showBlocking) setIsChecking(true);
      try {
        const res = await getLauncherStatus();
        if (res.success) {
          applyStatus(res.data);
        } else {
          setError(res.error || "Failed to check launch prerequisites");
          setBackendConnected(false);
          setRuntimeAvailable(false);
          setBackendStarting(false);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to check launch prerequisites");
        setBackendConnected(false);
        setRuntimeAvailable(false);
        setBackendStarting(false);
      } finally {
        if (showBlocking) {
          initialCheckDoneRef.current = true;
          setIsChecking(false);
        }
      }
    },
    [applyStatus],
  );

  useEffect(() => {
    let cancelled = false;

    const runInitial = async () => {
      // Only show the blocking "Checking installation…" screen on first load.
      if (!initialCheckDoneRef.current) setIsChecking(true);
      try {
        const res = await getLauncherStatus();
        if (cancelled) return;
        if (res.success) {
          applyStatus(res.data);
        } else {
          setError(res.error || "Failed to check launch prerequisites");
          setBackendConnected(false);
          setRuntimeAvailable(false);
          setBackendStarting(false);
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to check launch prerequisites");
        setBackendConnected(false);
        setRuntimeAvailable(false);
        setBackendStarting(false);
      } finally {
        if (!cancelled) {
          initialCheckDoneRef.current = true;
          setIsChecking(false);
        }
      }
    };

    void runInitial();

    const unsubscribe = onLauncherStatusChange((st) => {
      if (cancelled) return;
      applyStatus(st);
    });

    void startLauncherStatusWatch({ intervalMs: 3000 });

    return () => {
      cancelled = true;
      unsubscribe();
      void stopLauncherStatusWatch();
    };
  }, [applyStatus]);

  // Auto-start logic is handled in the check loop above to keep decisions in one place.

  // Refresh projects list when (re)entering the launcher so covers/timestamps stay current
  // after actions performed in the main window (delete, cover regeneration, rename, etc.).
  useEffect(() => {
    let cancelled = false;

    const refreshProjects = async () => {
      try {
        const res = await listProjects<ProjectSettings>();
        if (cancelled) return;
        if (res?.success && Array.isArray(res.data)) {
          setProjects(res.data);
        }
      } catch {
        // ignore; launcher can still render with last known list
      }
    };

    const onFocus = () => {
      void refreshProjects();
    };
    const onVisibilityChange = () => {
      if (!document.hidden) {
        void refreshProjects();
      }
    };

    void refreshProjects();
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [setProjects]);


  const onLaunch = async () => {
    if (projects.length === 0) {
      setError("Create a project to continue.");
      return;
    }
    setIsLaunching(true);
    setError(null);
    const res = await launchMainWindow();
    if (!res.ok) {
      setError(res.error || "Failed to launch");
      setIsLaunching(false);
    } else {
     setIsLaunching(false);
    }
    // On success, main process will close this window shortly.
  };

  const handleCreateProject = async () => {
    const trimmed = newProjectName.trim();
    if (!trimmed) {
      toast("Project name is required");
      return;
    }

    setIsCreatingProject(true);
    try {
      const res = await createProject({
        name: trimmed,
        fps: DEFAULT_FPS,
      });
      if (!res?.success || !res.data) {
        toast("Failed to create project", {
          description: res?.error || "Unknown error",
        });
        return;
      }

      const created = res.data as ProjectSettings;
      const alreadyInList = projects.some((p) => p.id === created.id);
      if (!alreadyInList) {
        addProject(created);
      }
      await setActiveProjectId(created.id);

      setCreateOpen(false);
      toast("Project created", { description: created.name });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Launcher: failed to create project", e);
      toast("Failed to create project");
    } finally {
      setIsCreatingProject(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!projectToDelete) return;
    setIsDeletingProject(true);
    try {
      const res = await deleteProject(projectToDelete.id);
      if (!res?.success) {
        toast("Failed to delete project", {
          description: res?.error || "Unknown error",
        });
        return;
      }

      removeProject(projectToDelete.id);

      // Keep the persisted "active project id" in sync with what's left.
      const remaining = projects.filter((p) => p.id !== projectToDelete.id);
      await setActiveProjectId(remaining.length > 0 ? remaining[0].id : null);

      setDeleteOpen(false);
      toast("Project deleted", { description: projectToDelete.name });
      setProjectToDelete(null);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Launcher: failed to delete project", e);
      toast("Failed to delete project");
    } finally {
      setIsDeletingProject(false);
    }
  };


  const showBlockingCheck =
    !initialCheckDoneRef.current && (isChecking || backendStarting);

  if (showBlockingCheck) {
    const title = backendStarting ? "Starting backend" : "Checking installation";
    const subtitle = backendStarting
      ? "Warming things up. This usually takes a few seconds."
      : "Verifying your runtime and connecting to the backend.";

    const StatusPill = ({
      label,
      state,
    }: {
      label: string;
      state: "ok" | "bad" | "pending";
    }) => {
      const base =
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium";
      if (state === "ok") {
        return (
          <span className={`${base} border-emerald-400/20 bg-emerald-400/10 text-emerald-200`}>
            <LuCheck className="h-3.5 w-3.5" />
            {label}
          </span>
        );
      }
      if (state === "bad") {
        return (
          <span className={`${base} border-red-400/20 bg-red-400/10 text-red-200`}>
            <LuX className="h-3.5 w-3.5" />
            {label}
          </span>
        );
      }
      return (
        <span className={`${base} border-brand-light/10 bg-brand-light/5 text-brand-light/70`}>
          <span className="h-2 w-2 rounded-full bg-brand-light/30 animate-pulse" />
          {label}
        </span>
      );
    };

    return (
      <main className="w-full h-screen flex flex-col bg-black text-center font-poppins relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-24 left-1/2 h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-brand-accent-two-shade/15 blur-3xl" />
          <div className="absolute -bottom-32 right-[-120px] h-[520px] w-[520px] rounded-full bg-brand-accent-shade/10 blur-3xl" />
          <div className="absolute inset-0 bg-gradient-to-b from-black via-black to-brand-background-dark/60" />
        </div>

        <div className="relative w-full max-w-[720px] px-6 flex-1 flex flex-col justify-center">
          <div className="rounded-2xl border border-brand-light/10 bg-brand-background/30 backdrop-blur-xl shadow-[0_0_0_1px_rgba(255,255,255,0.02),0_25px_80px_rgba(0,0,0,0.55)] overflow-hidden">
            <div className="px-8 pt-8 pb-6">
              <div className="flex items-start justify-between gap-6">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <div className="absolute inset-0 rounded-2xl bg-brand-accent-two-shade/25 blur-xl" />
                    <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl border border-brand-light/10 bg-black/40">
                      <PiRocketLaunchFill className="h-5 w-5 text-brand-light/90" />
                    </div>
                  </div>
                  <div className="text-left">
                    <div className="text-[11px] uppercase tracking-[0.35em] text-brand-light/45">
                      Apex Studio
                    </div>
                    <div className="mt-1 text-xl font-semibold tracking-tight text-brand-light">
                      {title}
                    </div>
                    <div className="mt-1 text-[13px] text-brand-light/60">{subtitle}</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowBlockingDetails((v: boolean) => !v)}
                    className="inline-flex items-center gap-2 rounded-md border border-brand-light/10 bg-brand-light/5 px-3 py-2 text-[12px] font-medium text-brand-light/75 hover:bg-brand-light/10 hover:text-brand-light transition-colors"
                  >
                    {showBlockingDetails ? (
                      <>
                        <LuChevronUp className="h-4 w-4" />
                        Hide details
                      </>
                    ) : (
                      <>
                        <LuChevronDown className="h-4 w-4" />
                        Show details
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="mt-6 flex items-center gap-3 justify-between">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusPill
                    label={runtimeAvailable ? "Runtime ready" : "Runtime"}
                    state={error && !runtimeAvailable ? "bad" : runtimeAvailable ? "ok" : "pending"}
                  />
                  <StatusPill
                    label={backendConnected ? "Backend connected" : backendStarting ? "Backend starting" : "Backend"}
                    state={
                      error && !backendConnected && !backendStarting
                        ? "bad"
                        : backendConnected
                          ? "ok"
                          : "pending"
                    }
                  />
                  <StatusPill
                    label={projects.length > 0 ? `${projects.length} project${projects.length === 1 ? "" : "s"}` : "Projects"}
                    state={projects.length > 0 ? "ok" : "pending"}
                  />
                </div>

                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 rounded-full border-2 border-brand-light/15 border-t-brand-accent-two-shade animate-spin" />
                  <div className="text-[12px] text-brand-light/55">
                    {backendStarting ? "Booting…" : "Checking…"}
                  </div>
                </div>
              </div>

              {error ? (
                <div className="mt-5 rounded-xl border border-red-400/20 bg-red-500/10 px-4 py-3 text-left">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[12px] font-semibold text-red-200">Something needs attention</div>
                      <div className="mt-1 text-[12px] leading-relaxed text-red-200/80 break-words">
                        {error}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          if (error) {
                            await navigator.clipboard.writeText(error);
                            toast("Copied", { description: "Error copied to clipboard" });
                          }
                        } catch {
                          toast("Copy failed", { description: "Could not access clipboard" });
                        }
                      }}
                      className="shrink-0 inline-flex items-center gap-2 rounded-md border border-red-400/20 bg-red-500/10 px-3 py-2 text-[12px] font-medium text-red-100/90 hover:bg-red-500/15 transition-colors"
                    >
                      <LuCopy className="h-4 w-4" />
                      Copy
                    </button>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        void refreshLauncherStatus({ showBlocking: true });
                      }}
                      className="inline-flex items-center gap-2 rounded-md bg-brand-accent-two-shade text-brand-light px-4 py-2 text-[12px] font-semibold hover:bg-brand-accent-shade transition-colors"
                    >
                      <LuRotateCcw className="h-4 w-4" />
                      Retry
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        setShowInstaller(true);
                      }}
                      className="inline-flex items-center gap-2 rounded-md border border-brand-light/15 bg-brand-background-light px-4 py-2 text-[12px] font-semibold text-brand-light hover:bg-brand-light/10 transition-colors"
                    >
                      <LuRotateCcw className="h-4 w-4" />
                      Reinstall
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            {showBlockingDetails ? (
              <div className="border-t border-brand-light/10 bg-black/20 px-8 py-6 text-left">
                <div className="text-[11px] uppercase tracking-[0.3em] text-brand-light/45">
                  Live details
                </div>
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-[12px] text-brand-light/70">
                  <div className="rounded-lg border border-brand-light/10 bg-brand-light/5 px-3 py-2">
                    <div className="text-brand-light/50">Runtime available</div>
                    <div className="mt-1 text-brand-light">
                      {runtimeAvailable ? "Yes" : "Not yet"}
                    </div>
                  </div>
                  <div className="rounded-lg border border-brand-light/10 bg-brand-light/5 px-3 py-2">
                    <div className="text-brand-light/50">Backend connected</div>
                    <div className="mt-1 text-brand-light">
                      {backendConnected ? "Yes" : backendStarting ? "Starting" : "No"}
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void refreshLauncherStatus()}
                    className="inline-flex items-center gap-2 rounded-md border border-brand-light/10 bg-brand-light/5 px-3 py-2 text-[12px] font-medium text-brand-light/80 hover:bg-brand-light/10 hover:text-brand-light transition-colors"
                  >
                    <LuRotateCcw className="h-4 w-4" />
                    Refresh status
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </main>
    );
  }

  if (!hasBackend || showInstaller) {
    return <Installer hasBackend={hasBackend} setShowInstaller={setShowInstaller} />;
  }


  return (
    <main className="w-full h-screen flex flex-col bg-black text-center font-poppins">
      <div className="flex items-start justify-between gap-4">
        <div className="px-12 py-8 gap-2.5 flex flex-col">
          <div className="text-sm uppercase tracking-[0.35em] text-brand-light/50 text-start">
            <h4>Launcher</h4>
          </div>
          <div className="flex items-start justify-between gap-4">
            <h2 className="text-3xl font-semibold tracking-tight text-brand-light text-start">
              Apex Studio
            </h2>
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            setError(null);
            setNewProjectName(getNextDefaultProjectName());
            setCreateOpen(true);
          }}
          className="flex items-center gap-2 w-fit bg-brand-background border mt-8 mr-12 border-brand-light/10 text-brand-light rounded-md px-5 py-2 hover:bg-brand/60 hover:border-brand-light/20 transition-colors duration-200"
        >
          <LuPlus className="w-4 h-4" />
          <span className="text-[12.5px] font-medium">Create</span>
        </button>
      </div>
      <div className="absolute bottom-0 left-0 right-0 px-12 py-6 border-t bg-brand-background-dark border-brand-light/10 flex items-center justify-between">
        <button
          type="button"
          onClick={() => {
            setError(null);
            setShowInstaller(true);
          }}
          className="flex items-center gap-2 w-fit dark border bg-brand-background-light border-brand-light/15 text-brand-light hover:bg-brand-light/10 text-[11.5px] font-medium px-5 h-8 rounded-[6px] duration-200"
        >
          <span className="flex items-center gap-2">
            <LuRotateCcw className="w-3.5 h-3.5" />
            <span className="text-[13px] font-medium text-brand-light">Reinstall</span>
          </span>
        </button>
        <button 
          onClick={() => void onLaunch()}
          disabled={isChecking || isLaunching || !canLaunch}
          className="flex items-center gap-2 w-fit bg-brand-accent-two-shade text-brand-light rounded-md px-6 py-2 hover:bg-brand-accent-shade transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-brand-accent-two-shade"
        >
          <PiRocketLaunchFill className="w-3.5 h-3.5" />
          <span className="text-[13px] font-medium text-brand-light">
            {isLaunching ? "Launching..." : "Launch"}
          </span>
        </button>
      </div>
      <div className="px-12 py-4 flex-1 overflow-y-auto">
        {projects.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-brand-light/60 text-sm">
              No projects yet. Click <span className="text-brand-light font-medium">Create</span> to get started.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {projects.map((project) => (
              <div key={project.id} className="relative">
                <ProjectCard project={project} />
                <button
                  type="button"
                  aria-label={`Delete ${project.name}`}
                  className="absolute bottom-2.5 right-2.5 p-1 rounded hover:bg-brand-light/5"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setProjectToDelete(project);
                    setDeleteOpen(true);
                  }}
                >
                  <LuTrash className="w-3.5 h-3.5 text-brand-light/40 hover:text-brand-light/60 transition-colors duration-200" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <Dialog
        open={createOpen}
        onOpenChange={(open) => {
          setCreateOpen(open);
          if (!open) {
            setIsCreatingProject(false);
            setNewProjectName("");
          }
        }}
      >
        <DialogContent className="dark bg-brand-background/95 font-poppins backdrop-blur-md border border-brand-light/10 py-4 ">
          <DialogHeader>
            <DialogTitle className="text-brand-light text-base font-medium">Create project</DialogTitle>
            <DialogDescription className="text-brand-light/70 text-xs ">
              Choose a name for your new project.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <div className="text-brand-light text-[11px] font-medium">
              Project name
            </div>
            <Input
              autoFocus
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="e.g. My Project"
              className="dark text-brand-light placeholder:text-brand-light/40 border-brand-light/15 text-xs! rounded-[6px]"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleCreateProject();
                }
              }}
              disabled={isCreatingProject}
            />
          </div>
          <DialogFooter className="gap-2 sm:gap-0">
            <button
              type="button"
              disabled={isCreatingProject}
              onClick={() => setCreateOpen(false)}
              className="dark border border-brand-light/15 text-brand-light hover:bg-brand-light/10 text-[11.5px] font-medium px-5 h-8 rounded-[6px] duration-200"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={isCreatingProject || newProjectName.trim().length === 0}
              onClick={() => void handleCreateProject()}
              className="bg-brand-accent-two-shade hover:bg-brand-accent-shade text-brand-light text-[11.5px] font-medium px-5 h-8 rounded-[6px] disabled:opacity-50 disabled:cursor-not-allowed duration-200"
            >
              {isCreatingProject ? "Creating..." : "Create"}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={deleteOpen}
        onOpenChange={(open) => {
          setDeleteOpen(open);
          if (!open) {
            setProjectToDelete(null);
            setIsDeletingProject(false);
          }
        }}
      >
        <AlertDialogContent className="dark bg-brand-background/95 font-poppins backdrop-blur-md border border-brand-light/10">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-brand-light text-base font-medium">
              Delete project?
            </AlertDialogTitle>
            <AlertDialogDescription className="text-brand-light/70 text-xs">
              This will permanently delete{" "}
              <span className="text-brand-light font-medium">
                {projectToDelete?.name ?? "this project"}
              </span>{" "}
              and its saved JSON.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel
              disabled={isDeletingProject}
              className="dark border-brand-light/15 text-brand-light hover:bg-brand-light/10 text-[11.5px] font-medium px-5 h-8 rounded-[6px] duration-200"
            >
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={!projectToDelete || isDeletingProject}
              onClick={(e) => {
                e.preventDefault();
                void handleConfirmDelete();
              }}
              className="bg-red-500/80 hover:bg-red-500/70 text-brand-light text-[11.5px] font-medium px-5 h-8 rounded-[6px] duration-200"
            >
              {isDeletingProject ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
};

export default Launcher;



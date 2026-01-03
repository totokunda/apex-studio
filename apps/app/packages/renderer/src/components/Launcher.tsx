import React, { useEffect, useMemo, useState } from "react";
import { PiRocketLaunchFill } from "react-icons/pi";
import {
  checkPythonHealth,
  getApiPathSetting,
  getBackendUrl,
  getPathExists,
  launchMainWindow,
  listProjects,
  setActiveProjectId,
  getUserDataPath,
  readFileBuffer,
  createProject,
  deleteProject,
} from "@app/preload";
import Installer from "@/components/Installer";
import { ProjectSettings, useProjectsStore } from "@/lib/projects";
import { LuFolder, LuClock, LuTrash, LuPlus } from "react-icons/lu";
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
  const normalizedAbs = String(absPath || "");
  const base = String(userDataDir || "");
  if (!normalizedAbs || !base) return null;
  if (!normalizedAbs.startsWith(base)) return null;
  const rel = normalizedAbs.slice(base.length);
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
  const [apiPathExists, setApiPathExists] = useState(false);
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
    return backendConnected || apiPathExists;
  }, [backendConnected, apiPathExists]);

  const canLaunch = useMemo(() => {
    const hasProjects = projects.length > 0;
    return hasBackend && hasProjects;
  }, [hasBackend, projects.length]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setIsChecking(true);
      setError(null);


      

      try {
        const apiPath = await getApiPathSetting();
        const apiPathTrimmed = (apiPath ?? "").trim();

        const [pythonHealthRes, backendUrlRes] = await Promise.all([
          checkPythonHealth().catch(() => ({ success: false } as any)),
          getBackendUrl().catch(() => ({ success: false } as any)),
        ]);

        // 1) Connected to a running bundled python API?
        const pythonHealthy = Boolean(
          pythonHealthRes?.success && pythonHealthRes?.data?.healthy,
        );

        // 2) Connected to configured backend URL?
        let backendHealthy = false;
        const backendUrl =
          backendUrlRes?.success && backendUrlRes?.data?.url
            ? String(backendUrlRes.data.url)
            : null;
        if (backendUrl) {
          try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 1500);
            const res = await fetch(`${backendUrl}/health`, {
              signal: controller.signal,
            }).finally(() => clearTimeout(timeout));
            backendHealthy = res.ok;
          } catch {
            backendHealthy = false;
          }
        }

        // 3) apiPath exists on disk?
        let apiExists = false;
        if (apiPathTrimmed) {
          const existsRes = await getPathExists(apiPathTrimmed).catch(
            () => null,
          );
          apiExists = Boolean(existsRes?.success && existsRes?.data?.exists);
        }

        if (cancelled) return;
        setBackendConnected(Boolean(pythonHealthy || backendHealthy));
        setApiPathExists(apiExists);
      } catch (e) {
        if (cancelled) return;
        setBackendConnected(false);
        setApiPathExists(false);
        setError(
          e instanceof Error ? e.message : "Failed to check launch prerequisites",
        );
      } finally {
        if (!cancelled) setIsChecking(false);
      }
    };

    void run();
    const interval = setInterval(() => {
      void run();
    }, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

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

  if (!hasBackend) {
    return <Installer />;
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
      <div className="absolute bottom-0 left-0 right-0 px-12 py-6 border-t bg-brand-background-dark border-brand-light/10 flex justify-end">
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



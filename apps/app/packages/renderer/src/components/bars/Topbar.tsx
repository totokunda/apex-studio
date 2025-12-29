import React, { useMemo, useState } from "react";
import {
  LuChevronDown,
  LuChevronUp,
  LuCheck,
  LuFolder,
  LuFolderPlus,
  LuSearch,
  LuTrash2,
} from "react-icons/lu";
import { useProjectsStore, type ProjectSettings } from "@/lib/projects";
import { createProject, deleteProject } from "@app/preload";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import { useLayoutConfigStore } from "@/lib/layout-config";
import { useViewportStore } from "@/lib/viewport";
import { PiResize } from "react-icons/pi";
import { LayoutIcon } from "./LayoutIcon";
import { TbPackageExport } from "react-icons/tb";
import SystemMemoryMenu from "@/components/system/SystemMemoryMenu";
import JobsMenu from "@/components/system/JobsMenu";
import ExportModal, { ExportSettings } from "@/components/dialogs/ExportModal";
import { ProgressBar } from "@/components/common/ProgressBar";
import {
  ExportClip,
  exportSequenceCancellable,
  ExportCancelledError,
} from "@app/export-renderer";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import type { AnyClipProps, ModelClipProps, GroupClipProps } from "@/lib/types";
import { prepareExportClipsForValue } from "@/lib/prepareExportClips";
import { sortClipsForStacking } from "@/lib/clipOrdering";
import { toast } from "sonner";
import { revealPathInFolder } from "@app/preload";
import { DEFAULT_FPS } from "@/lib/settings";
import { cn } from "@/lib/utils";
import { SettingsModal } from "../dialogs/SettingsModal";
import { ScrollArea } from "@/components/ui/scroll-area";

interface TopBarProps {

}

const Keycap: React.FC<{ label: string }> = ({ label }) => (
  <span className="inline-flex items-center justify-center px-1.5 h-5 min-w-[20px] shadow-sm rounded-[4px] bg-brand-light/[0.075] border border-brand-light/20 text-[10.5px]  text-brand-light/80">
    {label}
  </span>
);

const ShortcutRow: React.FC<{ action: string; keys: string[] }> = ({
  action,
  keys,
}) => (
  <div className="flex items-center justify-between">
    <span className="text-[11px] text-brand-light/90 font-medium">
      {action}
    </span>
    <div className="flex items-center gap-x-1.5">
      {keys.map((k, i) => (
        <React.Fragment key={`${action}-${k}-${i}`}>
          <Keycap label={k} />
          {i < keys.length - 1 && (
            <span className="text-brand-light/40 text-[10px]">+</span>
          )}
        </React.Fragment>
      ))}
    </div>
  </div>
);

const TopBar: React.FC<TopBarProps> = () => {
  const layout = useLayoutConfigStore((s) => s.layout);
  const setLayout = useLayoutConfigStore((s) => s.setLayout);
  const [layoutOpen, setLayoutOpen] = useState(false);
  const aspectRatio = useViewportStore((s) => s.aspectRatio);
  const setAspectRatio = useViewportStore((s) => s.setAspectRatio);
  const [sizeOpen, setSizeOpen] = useState(false);
  const setIsAspectEditing = useViewportStore((s) => s.setAspectEditing);
  const isAspectEditing = useViewportStore((s) => s.isAspectEditing);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const projects = useProjectsStore((s) => s.projects);
  const activeProjectId = useProjectsStore((s) => s.activeProjectId);
  const addProject = useProjectsStore((s) => s.addProject);
  const removeProject = useProjectsStore((s) => s.removeProject);
  const getAssetById = useClipStore((s) => s.getAssetById);
  const activeProject = useProjectsStore((s) => s.getActiveProject());
  const updateProject = useProjectsStore((s) => s.updateProject);
  const setActiveProjectId = useProjectsStore((s) => s.setActiveProjectId);

  const [projectNameDraft, setProjectNameDraft] = useState("");
  const [isEditingProjectName, setIsEditingProjectName] = useState(false);
  const [projectSearch, setProjectSearch] = useState("");
  const [deleteProjectOpen, setDeleteProjectOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<ProjectSettings | null>(
    null,
  );
  const [isDeletingProject, setIsDeletingProject] = useState(false);

  const [cancelExportFn, setCancelExportFn] = useState<(() => void) | null>(
    null,
  );
  const layoutLabel =
    layout === "default"
      ? "Default"
      : layout === "media"
        ? "Media"
        : "Properties";
  const clips = useClipStore((s) => s.clips);
  const timelines = useClipStore((s) => s.timelines);
  const getClipsForGroup = useClipStore((s) => s.getClipsForGroup);
  const getClipsByType = useClipStore((s) => s.getClipsByType);
  const getClipPositionScore = useClipStore((s) => s.getClipPositionScore);
  const fps = useControlsStore((s) => s.fps);
  const controlsFps = fps;

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

  const handleCreateProject = async () => {
    const name = getNextDefaultProjectName();
    const newFps = controlsFps || DEFAULT_FPS;
    try {
      const res = await createProject({
        name,
        fps: newFps,
      });

      if (!res?.success || !res?.data) {
        // eslint-disable-next-line no-console
        console.error("Failed to create project", res?.error);
        return;
      }

      const created = res.data as ProjectSettings;

      // Ensure the newly created project appears immediately in the list
      // (JSON persistence may refresh later, but we shouldn't wait for it).
      const alreadyInList = projects.some((p) => p.id === created.id);
      if (!alreadyInList) {
        addProject(created);
      }

      // Navigate to the newly created project.
      setActiveProjectId(created.id);
      setProjectSearch("");
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to create project", e);
    }
  };

  const handleCommitProjectName = async () => {
    if (!activeProject) return;
    const trimmed = projectNameDraft.trim();
    if (!trimmed || trimmed === activeProject.name) {
      return;
    }
    try {
      await updateProject(activeProject.id, { name: trimmed });
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to update project name", e);
    }
  };

  const handleConfirmDeleteProject = async () => {
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
      toast("Project deleted", {
        description: projectToDelete.name,
      });

      setDeleteProjectOpen(false);
      setProjectToDelete(null);
      setProjectSearch("");
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error("Failed to delete project", e);
      toast("Failed to delete project");
    } finally {
      setIsDeletingProject(false);
    }
  };
  const handleExport = async (settings: ExportSettings) => {
    const outpath = `${settings.path}/${settings.name}.${settings.format}`;

    // Derive pixel dimensions from the chosen resolution and the editor rect ratio.
    // In the editor we compute rectWidth/rectHeight as:
    //   rectWidth  = BASE_LONG_SIDE * (aspectRatio.width / aspectRatio.height)
    //   rectHeight = BASE_LONG_SIDE
    // so the effective ratio is rectWidth / rectHeight = aspectRatio.width / aspectRatio.height.
    const base = Math.max(1, settings.resolution || 0);
    const rawRatio = aspectRatio.width / aspectRatio.height;
    const ratio = Number.isFinite(rawRatio) && rawRatio > 0 ? rawRatio : 1;

    // Mirror the editor rect: fix the "height" (short side in editor units) to `base`
    // and scale the width by the same ratio derived from rectWidth:rectHeight.
    const heightPx = base;
    const widthPx = Math.round(base * ratio);

    const preserveAlpha = !!settings.preserveAlpha;

    setIsExporting(true);
    setExportProgress(0);
    

    try {
      // Prepare export-ready clips (attach filters, preprocessors, normalize paths, etc.).
      const preparedClips: ExportClip[] = [];
      let contentClips = sortClipsForStacking(
        clips as AnyClipProps[],
        timelines,
      );
      // filter clips with non visible timelines
      contentClips = contentClips.filter((clip) => {
        const timeline = timelines.find(
          (t) => t.timelineId === clip.timelineId,
        );
        return !timeline?.hidden;
      });

      for (const clip of contentClips) {
        if (clip.type === "filter") continue; // filters become applicators, not standalone content
        const prepared = prepareExportClipsForValue(
          clip as AnyClipProps,
          {
            aspectRatio,
            getAssetById,
            getClipsForGroup,
            getClipsByType,
            getClipPositionScore,
            timelines,
          },
          {
            clearMasks: false,
            applyCentering: false,
            dimensionsFrom: "aspect",
          },
        );
        preparedClips.push(...prepared.exportClips);
      }


      const { promise, cancel } = exportSequenceCancellable({
        mode: "video",
        filename: outpath,
        includeAudio: settings.includeAudio,
        clips: preparedClips,
        fps: fps,
        width: widthPx,
        height: heightPx,
        encoderOptions: {
          format: settings.format,
          codec: settings.codec,
          bitrate: settings.bitrate,
          alpha: preserveAlpha,
        },
        backgroundColor: preserveAlpha ? undefined : "#000000",
        audioOptions: {
          format: settings.audioFormat ?? "mp3",
        },
        onProgress: ({ ratio }) => {
          setIsExporting(true);
          setExportProgress(typeof ratio === "number" ? ratio : 0);
        },
        onDone: async () => {
          toast("Export completed", {
            description: outpath,
          });
          // Best-effort: open the exported file location for the user.
          try {
            await revealPathInFolder(outpath);
          } catch (e) {
            // eslint-disable-next-line no-console
            console.error("Failed to reveal export path", e);
          }
        },
      });

      setCancelExportFn(() => cancel);

      await promise;
    } catch (err) {
      if (
        err instanceof ExportCancelledError ||
        (err as any)?.name === "ExportCancelledError"
      ) {
        // Export was cancelled by the user; do not treat as an error or show a completion toast.
      } else {
        // Swallow other errors; the user can retry.
        // Optionally log for debugging.
        console.error(err);
      }
    } finally {
      setIsExporting(false);
      setExportProgress(null);
      setCancelExportFn(null);
    }
  };

  const filteredProjects = projects.filter((project) => {
    const query = projectSearch.trim().toLowerCase();
    if (!query) return true;
    return (project.name ?? "").toLowerCase().includes(query);
  });

  const [settingsOpen, setSettingsOpen] = useState(false);

  const canExport = useMemo(() => {
    if (!clips || clips.length === 0) return false;

    // Only consider clips that are on a visible (non-hidden) timeline
    const visibleTimelineIds = new Set(
      timelines.filter((t) => !t.hidden).map((t) => t.timelineId),
    );

    const hasExportableMedia = (clip: AnyClipProps): boolean => {
      // Group clips may contain exportable children; for groups we ignore the
      // group's own timeline/hidden status and instead inspect their children.
      if (clip.type === "group") {
        const group = clip as GroupClipProps;
        const children = getClipsForGroup(group.children) || [];
        for (const child of children) {
          if (hasExportableMedia(child as AnyClipProps)) {
            return true;
          }
        }
        return false;
      }

      if (clip.hidden && !clip.groupId) return false;
      if (!clip.timelineId || !visibleTimelineIds.has(clip.timelineId)) {
        return false;
      }

      // Basic media clips must have an assetId
      if (clip.type === "image" || clip.type === "audio" || clip.type === "video") {
        return Boolean((clip as AnyClipProps & { assetId?: string }).assetId);
      }

      // Model clips are exportable only once they have a concrete media result
      if (clip.type === "model") {
        const modelClip = clip as ModelClipProps;
        const hasAsset = !!modelClip.assetId;
        const hasSrcGeneration =
          Array.isArray(modelClip.generations) &&
          modelClip.generations.some((g: any) => !!g?.src);
        return hasAsset || hasSrcGeneration;
      }

      if (clip.type === "draw" || clip.type === "text" || clip.type === "shape") {
        return true;
      }

      return false;
    };

    return clips.some((clip) => hasExportableMedia(clip));
  }, [clips, timelines, getClipsForGroup]);

  return (
    <div className="w-full relative h-8 mt-2 px-6 pl-4 flex items-center justify-between space-x-2">
      {/* Project name editor */}
      <div className="flex flex-row items-center gap-1.5 min-w-0">
        {/* Projects dropdown (top-left) */}
        <SettingsModal open={settingsOpen} onOpenChange={setSettingsOpen} />
        <DropdownMenu>
          <DropdownMenuTrigger className="text-brand-light/90 dark max-w-[220px] h-[34px] flex items-center space-x-2 px-3 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
            <LuFolder className="w-4 h-4 shrink-0" />
            <span className="text-[11px] truncate">
              Projects
            </span>
            <div className="ml-1">
              <LuChevronDown className="w-3.5 h-3.5" />
            </div>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="start"
            className="dark w-60 font-poppins bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px] p-0 gap-0"
          >
            
            <div className=" relative">
              <LuSearch className="w-3.5 h-3.5 text-brand-light/60 absolute left-2 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={projectSearch}
                onChange={(e) => setProjectSearch(e.target.value)}
                placeholder="Search by name..."
                className="w-full h-8 rounded-[5px] pl-7  px-2 text-[11px] text-brand-light/90 placeholder:text-brand-light/40 focus:outline-none focus:border-brand-light/30 "
              />
            </div>
            <DropdownMenuSeparator className="my-0" />
            
            <ScrollArea className="max-h-64">
              <div className="px-1 py-1 flex flex-col">
                {filteredProjects.length === 0 && (
                  <DropdownMenuItem
                    disabled
                    className="dark text-[11px] text-brand-light/60 px-2"
                  >
                    {projects.length === 0
                      ? "No projects yet"
                      : "No matching projects"}
                  </DropdownMenuItem>
                )}
                {filteredProjects.map((project, index) => (
                  <React.Fragment key={project.id}>
                    <DropdownMenuItem
                      className={cn(
                        "dark text-[11px] font-medium flex items-center gap-x-2 cursor-pointer",
                        index === 0 && "mt-1",
                      )}
                      onClick={() => {
                        setActiveProjectId(project.id);
                      }}
                    >
                      <LuFolder className="w-3.5 h-3.5 text-brand-light/70" />
                      <span className="truncate">{project.name}</span>
                      <div className="ml-auto flex items-center gap-1">
                        {activeProjectId === project.id && (
                          <LuCheck className="w-4 h-4 text-brand-light" />
                        )}
                        {project.id !== activeProjectId && (
                          <button
                            type="button"
                            title={`Delete ${project.name}`}
                            className="inline-flex items-center justify-center w-6 h-6 rounded-[4px] hover:bg-red-500/15 text-brand-light/60 hover:text-red-400 transition-colors"
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setProjectToDelete(project);
                              setDeleteProjectOpen(true);
                            }}
                          >
                            <LuTrash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </DropdownMenuItem>
                    {index !== filteredProjects.length - 1 && (
                      <DropdownMenuSeparator />
                    )}
                  </React.Fragment>
                ))}
              </div>
            </ScrollArea>

            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="dark text-[11px] font-medium flex items-center gap-x-2 cursor-pointer mb-1"
              onClick={() => {
                void handleCreateProject();
              }}
            >
              <LuFolderPlus className="w-3.5 h-3.5 text-brand-light/80" />
              <span>New Project</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Project name input */}
        {activeProject && (
        <input
          type="text"
          value={
            activeProject
              ? isEditingProjectName
                ? projectNameDraft
                : activeProject.name
              : ""
          }
          onFocus={() => {
            if (!activeProject) return;
            setIsEditingProjectName(true);
            setProjectNameDraft(activeProject.name ?? "");
          }}
          onChange={(e) => {
            if (!activeProject) return;
            setProjectNameDraft(e.target.value);
          }}
          onBlur={async () => {
            if (!activeProject) return;
            await handleCommitProjectName();
            setIsEditingProjectName(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              (e.currentTarget as HTMLInputElement).blur();
            }
          }}
          disabled={!activeProject}
          placeholder="Project 1"
          className="max-w-xs w-[220px] h-[34px] focus:border transition-all border border-transparent font-medium duration-150 focus:bg-brand-light/10 hover:bg-brand-background/80  focus:border-brand-light/10 focus:outline-none rounded-[6px] px-3 flex items-center justify-between text-[12px] text-brand-light/90 placeholder:text-brand-light/40 shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
        />
        )}
      </div>
      <div className="flex items-center justify-end gap-2">
      {/* Exit Custom button when in aspect editing mode */}
        {isAspectEditing && (
        <button
          onClick={() => setIsAspectEditing(false)}
          className="text-brand-light/90 dark h-[34px] px-4 flex items-center gap-x-2 font-medium border border-red-500/30 hover:border-red-500/50 bg-red-500/10 hover:bg-red-500/20 rounded-[6px] transition-all duration-300 cursor-pointer"
        >
          <span className="text-[11px]">Exit Aspect Editing</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
        )}
        <SystemMemoryMenu />
        <JobsMenu />
      <DropdownMenu open={shortcutsOpen} onOpenChange={setShortcutsOpen}>
        <DropdownMenuTrigger className="text-brand-light/90 dark w-32 h-[34px] relative flex items-center space-x-2 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
          <span className="text-[11px] inline-flex items-center space-x-1 w-full">
            <span className="text-brand-light/50 font-light">⌘</span>{" "}
            <span>Shortcuts</span>
          </span>
          {shortcutsOpen ? (
            <LuChevronUp className="w-3.5 h-3.5" />
          ) : (
            <LuChevronDown className="w-3.5 h-3.5" />
          )}
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="dark w-[400px] font-poppins p-0 overflow-hidden bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px] z-[100]"
        >
          <div className="border-b border-brand-light/10 px-3 py-2">
            <span className="text-[11px] uppercase tracking-wide text-brand-light/70">
              Keyboard Shortcuts
            </span>
          </div>
          <div className="divide-y divide-brand-light/10 ">
            <div className="p-3">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                <ShortcutRow action="Zoom to Fit" keys={["⌘", "0"]} />
                <ShortcutRow action="Zoom to 50%" keys={["⌘", "5"]} />
                <ShortcutRow action="Zoom to 100%" keys={["⌘", "1"]} />
                <ShortcutRow action="Zoom to 200%" keys={["⌘", "2"]} />
                <ShortcutRow action="Group" keys={["⌘", "G"]} />
                <ShortcutRow action="Ungroup" keys={["⌘", "⇧", "G"]} />
              </div>
            </div>
            <div className="p-3">
              <DropdownMenuLabel className="dark text-[11px] font-semibold text-brand-light p-0">
                Tools
              </DropdownMenuLabel>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
                <ShortcutRow action="Pointer" keys={["V"]} />
                <ShortcutRow action="Hand" keys={["H"]} />
                <ShortcutRow action="Draw" keys={["D"]} />
                <ShortcutRow action="Text" keys={["T"]} />
                <ShortcutRow action="Mask" keys={["M"]} />
                <ShortcutRow action="Shape" keys={["S"]} />
              </div>
            </div>

            <div className="p-3">
              <DropdownMenuLabel className="dark text-[11px] font-semibold text-brand-light p-0">
                Shape Mode
              </DropdownMenuLabel>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">
                <ShortcutRow action="Rectangle" keys={["1"]} />
                <ShortcutRow action="Ellipse" keys={["2"]} />
                <ShortcutRow action="Polygon" keys={["3"]} />
                <ShortcutRow action="Line" keys={["4"]} />
                <ShortcutRow action="Star" keys={["5"]} />
              </div>
            </div>
            <div className="p-3">
              <DropdownMenuLabel className="dark text-[11px] font-semibold text-brand-light p-0">
                Mask Mode
              </DropdownMenuLabel>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">
                <ShortcutRow action="Lasso" keys={["1"]} />
                <ShortcutRow action="Shape" keys={["2"]} />
                <ShortcutRow action="Draw" keys={["3"]} />
                <ShortcutRow action="Touch" keys={["4"]} />
              </div>
            </div>
            <div className="p-3">
              <DropdownMenuLabel className="dark text-[11px] font-semibold text-brand-light p-0">
                Draw Mode
              </DropdownMenuLabel>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">
                <ShortcutRow action="Brush" keys={["1"]} />
                <ShortcutRow action="Highlighter" keys={["2"]} />
                <ShortcutRow action="Eraser" keys={["3"]} />
              </div>
            </div>
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu open={sizeOpen} onOpenChange={setSizeOpen}>
        <DropdownMenuTrigger className="text-brand-light/90 dark w-24 h-[34px] flex items-center space-x-1  px-2 relative font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
          <PiResize className="w-4 h-4" /> <span className="text-xs">Size</span>
          <div className="absolute right-2">
            {sizeOpen ? (
              <LuChevronUp className="w-3.5 h-3.5" />
            ) : (
              <LuChevronDown className="w-3.5 h-3.5" />
            )}
          </div>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="dark w-48 flex flex-col bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px] font-poppins"
        >
          {[
            { id: "16:9", name: "Wide", w: 16, h: 9 },
            { id: "9:16", name: "Vertical", w: 9, h: 16 },
            { id: "1:1", name: "Square", w: 1, h: 1 },
            { id: "4:3", name: "Classic", w: 4, h: 3 },
            { id: "4:5", name: "Social", w: 4, h: 5 },
            { id: "21:9", name: "Cinema", w: 21, h: 9 },
            { id: "2:3", name: "Portrait", w: 2, h: 3 },
          ].map((opt, index) => (
            <>
              <DropdownMenuItem
                key={opt.id}
                textValue={opt.name}
                className="dark text-[12px] font-medium flex flex-row items-center cursor-pointer gap-x-3 w-full bg-transparent"
                onClick={() => {
                  setAspectRatio({ width: opt.w, height: opt.h, id: opt.id });
                  setIsAspectEditing(false);
                  setSizeOpen(false);
                }}
              >
                <div
                  className="w-[24px] border-[1.5px] border-brand-light rounded-xs"
                  style={{ height: `${(24 * (opt.h / opt.w)).toFixed(1)}px` }}
                />
                <div className="flex flex-col items-start gap-y-0.5">
                  <span>{opt.name}</span>
                  <span className="text-brand-light/50 text-xs font-light">
                    {opt.id}
                  </span>
                </div>
                {aspectRatio.id === opt.id && (
                  <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
                )}
              </DropdownMenuItem>
              {index !== 6 && <DropdownMenuSeparator />}
            </>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            key={"custom"}
            textValue={"Custom"}
            className="dark text-[12px] font-medium flex flex-row items-center cursor-pointer gap-x-3 w-full bg-transparent"
            onClick={(e) => {
              // Only prevent default if clicking on the edit button
              if ((e.target as HTMLElement).closest("button")) {
                e.preventDefault();
                e.stopPropagation();
                return;
              }
              // If not custom already, set to custom mode
              if (aspectRatio.id !== "custom") {
                setAspectRatio({
                  width: aspectRatio.width,
                  height: aspectRatio.height,
                  id: "custom",
                });
                setIsAspectEditing(true);
              }
              setSizeOpen(false);
            }}
          >
            <div className="w-[24px] h-[24px] border-[1.5px] border-dashed border-brand-light rounded-xs" />
            <div className="flex flex-col items-start gap-y-0.5 flex-1">
              <span>Custom</span>
              <span className="text-brand-light/50 text-xs font-light">
                {aspectRatio.id === "custom"
                  ? `${aspectRatio.width}:${aspectRatio.height}`
                  : "Set W:H"}
              </span>
            </div>
            {aspectRatio.id === "custom" && (
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsAspectEditing(!isAspectEditing);
                  setSizeOpen(false);
                }}
                className="px-3 py-1 text-[10px] bg-brand-light/10 hover:bg-brand-light/20 rounded-[4px] transition-colors"
              >
                {isAspectEditing ? "Done" : "Edit"}
              </button>
            )}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu open={layoutOpen} onOpenChange={setLayoutOpen}>
        <DropdownMenuTrigger className="text-brand-light/90 dark w-32 h-[34px] relative flex items-center space-x-2 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand/90 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer">
          <LayoutIcon type={layout as any} />
          <span className="text-[11px]">{layoutLabel}</span>
          <div className="absolute right-2">
            {layoutOpen ? (
              <LuChevronUp className="w-4 h-4" />
            ) : (
              <LuChevronDown className="w-4 h-4" />
            )}
          </div>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="dark w-48 font-poppins bg-brand-background/90 backdrop-blur-md border border-brand-light/10 rounded-[8px]"
        >
          <DropdownMenuItem
            className="dark text-[11px] font-medium flex items-center gap-x-2"
            onClick={() => {
              setLayout("default");
              setLayoutOpen(false);
            }}
          >
            <LayoutIcon type="default" />
            <span>Default</span>
            {layout === "default" && (
              <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
            )}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="dark text-[11px] font-medium flex items-center gap-x-2"
            onClick={() => {
              setLayout("media");
              setLayoutOpen(false);
            }}
          >
            <LayoutIcon type="media" />
            <span>Media</span>
            {layout === "media" && (
              <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
            )}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="dark text-[11px] font-medium flex items-center gap-x-2"
            onClick={() => {
              setLayout("properties");
              setLayoutOpen(false);
            }}
          >
            <LayoutIcon type="properties" />
            <span>Properties</span>
            {layout === "properties" && (
              <LuCheck className="w-4 h-4 ml-auto text-brand-light" />
            )}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <div className="flex items-center gap-2">
        {!isExporting ? (
          <button
            disabled={!canExport} 
            type="button"
            className="text-brand-light space-x-1.5 flex items-center justify-center px-5 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-brand disabled:border-brand font-medium h-[34px] hover:text-brand-light bg-brand-accent border border-brand-accent-two-shade hover:bg-brand-accent-two-shade rounded-[6px] py-1.5 transition-all duration-300 cursor-pointer"
            onClick={() => setExportOpen(true)}
          >
            <TbPackageExport size={16} />
            <span className="text-[11px]">Export</span>
          </button>
        ) : (
          <button
            type="button"
            className="hidden md:flex items-center gap-2 min-w-[180px] h-[34px] px-3 pr-5 rounded-[6px] border border-brand-light/10 bg-brand-background/80 hover:bg-brand-background cursor-pointer transition-colors"
            onClick={() => setExportOpen(true)}
          >
            <TbPackageExport size={14} className="text-brand-light/80" />
            <span className="text-[10px] text-brand-light/70 whitespace-nowrap">
              Exporting
            </span>
            <div className="flex-1 w-24">
              <ProgressBar
                percent={Math.round(
                  Math.max(0, Math.min(1, exportProgress ?? 0)) * 100,
                )}
                className="h-1.5 border-brand-light/20 bg-brand-background-dark/80"
                barClassName="bg-brand-accent"
              />
            </div>
            <span className="text-[10px] text-brand-light/60 w-4 text-right">
              {Math.round(Math.max(0, Math.min(1, exportProgress ?? 0)) * 100)}%
            </span>
          </button>
        )}
      </div>
      </div>

      <ExportModal
        open={exportOpen}
        onOpenChange={setExportOpen}
        onExport={handleExport}
        isExporting={isExporting}
        exportProgress={exportProgress}
        onCancelExport={() => {
          if (cancelExportFn) {
            cancelExportFn();
          }
        }}
      />

      <AlertDialog
        open={deleteProjectOpen}
        onOpenChange={(open) => {
          setDeleteProjectOpen(open);
          if (!open) {
            setProjectToDelete(null);
            setIsDeletingProject(false);
          }
        }}
        
      >
        <AlertDialogContent className="dark bg-brand-background/95 font-poppins backdrop-blur-md border border-brand-light/10">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-brand-light">
              Delete project?
            </AlertDialogTitle>
            <AlertDialogDescription className="text-brand-light/70">
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
              className="dark border-brand-light/15 text-brand-light hover:bg-brand-light/10 text-[12px] font-medium px-5"
            >
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={!projectToDelete || isDeletingProject}
              onClick={(e) => {
                // Prevent radix from closing early while we await deletion.
                e.preventDefault();
                void handleConfirmDeleteProject();
              }}
              className="bg-red-500/80 hover:bg-red-500/70 text-brand-light text-[12px] font-medium px-5"
            >
              {isDeletingProject ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default TopBar;

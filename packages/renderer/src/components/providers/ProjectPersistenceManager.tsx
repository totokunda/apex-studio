import React, { useEffect, useMemo, useRef } from "react";
import { useClipStore } from "@/lib/clip";
import { useControlsStore } from "@/lib/control";
import { useViewportStore } from "@/lib/viewport";
import { DEFAULT_FPS, TIMELINE_DURATION_SECONDS } from "@/lib/settings";
import { useProjectStore } from "@/lib/projectStore";
import { TICKS_PER_SECOND } from "@/lib/persistence/constants";
import {
  deserializeClipState,
  deserializeControlState,
  deserializeViewportState,
  serializeClipState,
  serializeControlState,
  serializeViewportState,
} from "@/lib/persistence/converters";
import {
  createProject,
  fetchProjects,
  loadProjectSnapshot,
  renameProject,
  saveProjectSnapshot,
} from "@/lib/persistence/client";
import type {
  PersistedProjectSnapshot,
  ProjectSummary,
} from "@/lib/persistence/types";

const LOCAL_STORAGE_KEY = "apex-studio:last-project-id";

const resetStoresToDefaults = () => {
  const defaultTotalFrames = TIMELINE_DURATION_SECONDS * DEFAULT_FPS;
  useControlsStore.setState({
    maxZoomLevel: 10,
    minZoomLevel: 1,
    zoomLevel: 1,
    totalTimelineFrames: defaultTotalFrames,
    timelineDuration: [0, defaultTotalFrames],
    fps: DEFAULT_FPS,
    focusFrame: 0,
    focusAnchorRatio: 0.5,
    selectedClipIds: [],
    isPlaying: false,
    isFullscreen: false,
    selectedMaskId: null,
  });

  useClipStore.setState({
    clips: [],
    timelines: [],
    clipDuration: 0,
    activeMediaItem: null,
    ghostStartEndFrame: [0, 0],
    ghostX: 0,
    ghostGuideLines: null,
    hoveredTimelineId: null,
    ghostTimelineId: null,
    ghostInStage: false,
    draggingClipId: null,
    isDragging: false,
    selectedPreprocessorId: null,
    snapGuideX: null,
    clipboard: [],
  });

  useViewportStore.setState({
    scale: 0.75,
    minScale: 0.1,
    maxScale: 4,
    position: { x: 0, y: 0 },
    tool: "pointer",
    shape: "rectangle",
    clipPositions: {},
    viewportSize: { width: 0, height: 0 },
    contentBounds: null,
    aspectRatio: { width: 16, height: 9, id: "16:9" },
    isAspectEditing: false,
  });
};

const applySnapshotToStores = (snapshot: PersistedProjectSnapshot) => {
  const tickRate = snapshot.tickRate || TICKS_PER_SECOND;
  const controls = deserializeControlState(snapshot.controls, tickRate);
  useControlsStore.setState({
    ...controls,
    isPlaying: false,
  });

  const clipData = deserializeClipState(
    snapshot.clips,
    controls.fps ?? DEFAULT_FPS,
    tickRate,
  );
  // Timelines first to preserve ordering when clips resolve
  useClipStore.getState().setTimelines(clipData.timelines || []);
  useClipStore.getState().setClips(clipData.clips || []);

  const viewport = deserializeViewportState(snapshot.viewport);
  useViewportStore.setState(viewport);
};

const hydrateProject = async (
  project: ProjectSummary,
  hydratingRef: React.MutableRefObject<boolean>,
) => {
  hydratingRef.current = true;
  useProjectStore.getState().setCurrentProject(project.id, project.name);
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, project.id);
  } catch {
    // ignore storage errors
  }

  resetStoresToDefaults();
  const snapshotRes = await loadProjectSnapshot(project.id);
  if (snapshotRes.success && snapshotRes.data) {
    const snapshot = snapshotRes.data as PersistedProjectSnapshot;
    applySnapshotToStores({
      tickRate: snapshot.tickRate || TICKS_PER_SECOND,
      clips: snapshot.clips,
      controls: snapshot.controls,
      viewport: snapshot.viewport,
    });
  }
  hydratingRef.current = false;
};

export const ProjectPersistenceManager: React.FC = () => {
  const hydratingRef = useRef(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentProjectId = useProjectStore((s) => s.currentProjectId);

  const bootstrap = useMemo(
    () => async () => {
      useProjectStore.getState().setLoading(true);
      const res = await fetchProjects();
      let projects: ProjectSummary[] = [];
      if (res.success && res.data && Array.isArray(res.data)) {
        projects = res.data;
      }

      if (projects.length === 0) {
        const created = await createProject("Project 1");
        if (created.success && created.data) {
          projects = [created.data as ProjectSummary];
        }
      }

      useProjectStore.getState().setProjects(projects);
      useProjectStore.getState().setLoading(false);

      const lastId = (() => {
        try {
          return localStorage.getItem(LOCAL_STORAGE_KEY);
        } catch {
          return null;
        }
      })();
      const active =
        projects.find((p) => p.id === lastId) ?? projects[0] ?? null;

      if (active) {
        await hydrateProject(active, hydratingRef);
      }

      // Wire project actions once bootstrap completes
      useProjectStore.setState({
        createProject: async (name?: string) => {
          const created = await createProject(name);
          if (created.success && created.data) {
            const project = created.data as ProjectSummary;
            useProjectStore.getState().upsertProject(project);
            await hydrateProject(project, hydratingRef);
            return project;
          }
          return null;
        },
        renameProject: async (id: string, name: string) => {
          const renamed = await renameProject(id, name);
          if (renamed.success && renamed.data) {
            const project = renamed.data as ProjectSummary;
            useProjectStore.getState().upsertProject(project);
            return project;
          }
          return null;
        },
        switchProject: async (id: string) => {
          const target = useProjectStore
            .getState()
            .projects.find((p) => p.id === id);
          if (!target) return;
          await hydrateProject(target, hydratingRef);
        },
      });
    },
    [],
  );

  useEffect(() => {
    bootstrap();
  }, [bootstrap]);

  useEffect(() => {
    if (!currentProjectId) return () => {};
    const scheduleSave = () => {
      if (hydratingRef.current) return;
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
      saveTimerRef.current = setTimeout(async () => {
        try {
          const projectId = useProjectStore.getState().currentProjectId;
          if (!projectId) return;
          const controls = useControlsStore.getState();
          const clips = useClipStore.getState();
          const viewport = useViewportStore.getState();
          const fps = Math.max(1, controls.fps || DEFAULT_FPS);
          useProjectStore.getState().markSaving(true);
          const snapshot: PersistedProjectSnapshot = {
            tickRate: TICKS_PER_SECOND,
            clips: serializeClipState(
              { clips: clips.clips, timelines: clips.timelines },
              fps,
              TICKS_PER_SECOND,
            ),
            controls: serializeControlState(controls, TICKS_PER_SECOND),
            viewport: serializeViewportState(viewport),
          };
          const res = await saveProjectSnapshot(projectId, snapshot);
          if (res.success) {
            useProjectStore.getState().markSavedAt(Date.now());
          } else {
            useProjectStore.getState().markSaving(false);
          }
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error("Project autosave failed", error);
          useProjectStore.getState().markSaving(false);
        }
      }, 600);
    };

    const unsubscribes = [
      useClipStore.subscribe(
        (state) => ({ clips: state.clips, timelines: state.timelines }),
        scheduleSave,
      ),
      useControlsStore.subscribe(
        (state) => ({
          fps: state.fps,
          totalTimelineFrames: state.totalTimelineFrames,
          timelineDuration: state.timelineDuration,
          focusFrame: state.focusFrame,
          zoomLevel: state.zoomLevel,
          maxZoomLevel: state.maxZoomLevel,
          minZoomLevel: state.minZoomLevel,
          focusAnchorRatio: state.focusAnchorRatio,
          selectedClipIds: state.selectedClipIds,
          isFullscreen: state.isFullscreen,
        }),
        scheduleSave,
      ),
      useViewportStore.subscribe(
        (state) => ({
          scale: state.scale,
          position: state.position,
          tool: state.tool,
          shape: state.shape,
          clipPositions: state.clipPositions,
          viewportSize: state.viewportSize,
          contentBounds: state.contentBounds,
          aspectRatio: state.aspectRatio,
          isAspectEditing: state.isAspectEditing,
        }),
        scheduleSave,
      ),
    ];

    return () => {
      unsubscribes.forEach((fn) => {
        try {
          fn();
        } catch {}
      });
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
    };
  }, [currentProjectId]);

  return null;
};

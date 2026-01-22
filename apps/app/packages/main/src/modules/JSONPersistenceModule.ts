import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { ipcMain } from "electron";
import path from "node:path";
import fs from "node:fs/promises";
import { randomUUID } from "node:crypto";

type JsonResponse<T> =
  | { success: true; data: T }
  | { success: false; error: string };

interface ProjectJsonV1 {
  version: string;
  meta: {
    id: string;
    name: string;
    createdAt: number;
    lastModified: number;
    appVersion?: string;
    projectId?: number;
  };
  settings: {
    fps: number;
    [key: string]: unknown;
  };
  editorState: Record<string, unknown>;
  assets: Record<string, unknown>;
  timeline: {
    tracks: any[];
    clips: any[];
  };
  manifests?: Record<string, unknown>;
  preprocessors?: Record<string, unknown>;
  [key: string]: unknown;
}

function ensureProjectJson(
  raw: any,
  projectId: number,
  appVersion: string | undefined,
): ProjectJsonV1 {
  const now = Date.now();
  const version = typeof raw?.version === "string" ? raw.version : "1.0.0";

  const metaIn = raw?.meta ?? {};
  const id =
    typeof metaIn.id === "string" && metaIn.id.length > 0
      ? metaIn.id
      : randomUUID();
  const name =
    typeof metaIn.name === "string" && metaIn.name.length > 0
      ? metaIn.name
      : "Untitled Project";
  const createdAt =
    typeof metaIn.createdAt === "number" && Number.isFinite(metaIn.createdAt)
      ? metaIn.createdAt
      : now;
  const lastModified =
    typeof metaIn.lastModified === "number" && Number.isFinite(metaIn.lastModified)
      ? metaIn.lastModified
      : now;

  const settings = {
    fps:
      typeof raw?.settings?.fps === "number" &&
      Number.isFinite(raw.settings.fps)
        ? raw.settings.fps
        : 24,
    ...(raw?.settings ?? {}),
  };

  const editorState = {
    ...(raw?.editorState ?? {}),
  };

  const assets = {
    ...(raw?.assets ?? {}),
  };

  const timeline = {
    tracks: Array.isArray(raw?.timeline?.tracks)
      ? raw.timeline.tracks
      : ([] as any[]),
    clips: Array.isArray(raw?.timeline?.clips)
      ? raw.timeline.clips
      : ([] as any[]),
  };

  const inputControls = {
    selectedRangeByInputId: raw?.inputControls?.selectedRangeByInputId ?? {},
    selectedInputClipIdByInputId: raw?.inputControls?.selectedInputClipIdByInputId ?? {},
    totalTimelineFramesByInputId: raw?.inputControls?.totalTimelineFramesByInputId ?? {},
    timelineDurationByInputId: raw?.inputControls?.timelineDurationByInputId ?? {},
    fpsByInputId: raw?.inputControls?.fpsByInputId ?? {},
    focusFrameByInputId: raw?.inputControls?.focusFrameByInputId ?? {},
    focusAnchorRatioByInputId: raw?.inputControls?.focusAnchorRatioByInputId ?? {},
    zoomLevelByInputId: raw?.inputControls?.zoomLevelByInputId ?? {},
    isPlayingByInputId: raw?.inputControls?.isPlayingByInputId ?? {},
  };


  return {
    version,
    meta: {
      id,
      name,
      createdAt,
      lastModified,
      appVersion,
      projectId,
    },
    settings,
    editorState,
    assets,
    timeline,
    inputControls,
  };
}

export class JSONPersistenceModule implements AppModule {
  #projectsDir: string | undefined;
  #appVersion: string | undefined;

  async enable({ app }: ModuleContext): Promise<void> {
    await app.whenReady();
    const userData = app.getPath("userData");
    const rootDir = path.join(userData, "projects-json");

    try {
      await fs.mkdir(rootDir, { recursive: true });
      await fs.mkdir(path.join(rootDir, "covers"), { recursive: true });
    } catch {
      // ignore mkdir errors; write will surface any real issues
    }

    this.#projectsDir = rootDir;
    this.#appVersion = app.getVersion?.() ?? undefined;
    this.registerHandlers();
  }

  private get projectsDir(): string {
    if (!this.#projectsDir) {
      throw new Error("JSONPersistenceModule: projects directory not initialized");
    }
    return this.#projectsDir;
  }

  private getProjectPath(projectId: number): string {
    return path.join(this.projectsDir, `project-${projectId}.json`);
  }

  private getProjectCoverPath(projectId: number): string {
    return path.join(this.projectsDir, "covers", `project-${projectId}.jpg`);
  }

  private get localRootDir(): string {
    return path.join(this.projectsDir, "local");
  }

  private get localManifestDir(): string {
    return path.join(this.localRootDir, "manifest");
  }

  private get localPreprocessorDir(): string {
    return path.join(this.localRootDir, "preprocessor");
  }

  private get localMaskDir(): string {
    return path.join(this.localRootDir, "mask");
  }

  private getMaskProjectDir(projectId: number): string {
    return path.join(this.localMaskDir, `project-${projectId}`);
  }

  private async ensureLocalDirs(): Promise<void> {
    try {
      await fs.mkdir(this.localManifestDir, { recursive: true });
    } catch {
      // ignore
    }
    try {
      await fs.mkdir(this.localPreprocessorDir, { recursive: true });
    } catch {
      // ignore
    }
    try {
      await fs.mkdir(this.localMaskDir, { recursive: true });
    } catch {
      // ignore
    }
  }

  private getManifestLocalPath(id: string): string {
    const safe = (value: string): string =>
      String(value ?? "")
        .trim()
        .replace(/[^a-zA-Z0-9._@-]/g, "_") || "unknown";
    return path.join(this.localManifestDir, `${safe(id)}.json`);
  }

  private getPreprocessorLocalPath(id: string): string {
    return path.join(this.localPreprocessorDir, `${id}.json`);
  }

  private getMaskBinaryRelativePath(
    projectId: number,
    clipId: string | number | undefined,
    maskId: string | number | undefined,
    kind: "lasso" | "contours",
  ): string {
    const safe = (value: string | number | undefined): string =>
      String(value ?? "")
        .trim()
        .replace(/[^a-zA-Z0-9_-]/g, "_") || "unknown";

    const clipPart = safe(clipId);
    const maskPart = safe(maskId);
    const fileName = `mask-${clipPart}-${maskPart}-${kind}.f32.bin`;
    return path.join("local", "mask", `project-${projectId}`, fileName);
  }

  private getMaskBinaryAbsolutePath(relativePath: string): string {
    return path.join(this.projectsDir, relativePath);
  }

  private async saveMaskDataBinariesFromPayload(
    projectId: number,
    payload: any,
  ): Promise<void> {
    if (!payload || typeof payload !== "object") return;

    const timeline = payload.timeline;
    const clips: any[] = Array.isArray(timeline?.clips) ? timeline.clips : [];
    if (clips.length === 0) return;

    const projectMaskDir = this.getMaskProjectDir(projectId);
    try {
      await fs.mkdir(projectMaskDir, { recursive: true });
    } catch {
      // ignore; writeFile will surface real errors
    }

    const writeOps: Array<Promise<void>> = [];

    for (const clip of clips) {
      if (!clip || typeof clip !== "object") continue;
      const anyClip: any = clip;
      const clipId = anyClip.clipId ?? anyClip.id ?? "";
      const masks: any[] = Array.isArray(anyClip.masks) ? anyClip.masks : [];
      if (masks.length === 0) continue;

      for (const mask of masks) {
        if (!mask || typeof mask !== "object") continue;
        const anyMask: any = mask;
        const maskId = anyMask.id ?? "";
        const keyframes = anyMask.keyframes;
        if (!keyframes || typeof keyframes !== "object") continue;
        const entries = Object.entries(keyframes as any);

        // Collect all keyframes with lassoPoints for this mask
        const lassoFrames: Array<{ frameKey: string; points: number[] }> = [];
        for (const [frameKey, rawData] of entries) {
          if (!rawData || typeof rawData !== "object") continue;
          const data: any = rawData;
          if (
            Array.isArray(data.lassoPoints) &&
            data.lassoPoints.length > 0 &&
            Number.isFinite(Number(frameKey))
          ) {
            lassoFrames.push({
              frameKey,
              points: data.lassoPoints as number[],
            });
          }
        }

        if (lassoFrames.length > 0) {
          const relPath = this.getMaskBinaryRelativePath(
            projectId,
            clipId,
            maskId,
            "lasso",
          );
          const absPath = this.getMaskBinaryAbsolutePath(relPath);

          const k = lassoFrames.length;
          const headerSize = 1 + k * 2; // [nKeyframes, frame, len, frame, len, ...]
          let valuesCount = 0;
          for (const lf of lassoFrames) {
            valuesCount += lf.points.length;
          }

          const arr = new Float32Array(headerSize + valuesCount);
          arr[0] = k;
          for (let i = 0; i < k; i++) {
            const lf = lassoFrames[i];
            const frameNum = Number(lf.frameKey);
            arr[1 + i * 2] = Number.isFinite(frameNum) ? frameNum : 0;
            arr[1 + i * 2 + 1] = lf.points.length;
          }

          let offset = headerSize;
          for (const lf of lassoFrames) {
            for (let i = 0; i < lf.points.length; i++) {
              const v = lf.points[i];
              arr[offset++] = Number.isFinite(v) ? v : 0;
            }
          }

          const buf = Buffer.from(
            arr.buffer,
            arr.byteOffset,
            arr.byteLength,
          );

          writeOps.push(
            fs
              .writeFile(absPath, buf)
              .catch((error) => {
                console.error(
                  "JSONPersistenceModule: failed to save lassoPoints binary",
                  absPath,
                  error,
                );
              }),
          );

          // Store only the binary reference in JSON for all keyframes that had lassoPoints
          for (const lf of lassoFrames) {
            const data: any = (keyframes as any)[lf.frameKey];
            if (!data || typeof data !== "object") continue;
            data.lassoPointsBinPath = relPath;
            delete data.lassoPoints;
          }
        }

        // Collect all keyframes with contours for this mask
        const contourFrames: Array<{ frameKey: string; contours: number[][] }> =
          [];
        for (const [frameKey, rawData] of entries) {
          if (!rawData || typeof rawData !== "object") continue;
          const data: any = rawData;
          if (
            Array.isArray(data.contours) &&
            data.contours.length > 0 &&
            Number.isFinite(Number(frameKey))
          ) {
            const normalized: number[][] = [];
            for (const c of data.contours as any[]) {
              if (Array.isArray(c) && c.length > 0) {
                normalized.push(
                  (c as any[]).map((v) =>
                    Number.isFinite(Number(v)) ? Number(v) : 0,
                  ),
                );
              }
            }
            if (normalized.length > 0) {
              contourFrames.push({
                frameKey,
                contours: normalized,
              });
            }
          }
        }

        if (contourFrames.length > 0) {
          const relPath = this.getMaskBinaryRelativePath(
            projectId,
            clipId,
            maskId,
            "contours",
          );
          const absPath = this.getMaskBinaryAbsolutePath(relPath);

          const k = contourFrames.length;
          // Layout:
          // [nKeyframes,
          //   frame, nContours, len0, len1, ..., values...,
          //   frame, nContours, len0, len1, ..., values..., ...]
          // We'll build this sequentially.

          // First compute total length
          let totalLength = 1; // nKeyframes
          for (const cf of contourFrames) {
            const nContours = cf.contours.length;
            const lengths = cf.contours.map((c) => c.length);
            const valuesCount = lengths.reduce(
              (sum, len) => sum + len,
              0,
            );
            totalLength += 2; // frame, nContours
            totalLength += nContours; // lengths
            totalLength += valuesCount; // values
          }

          const arr = new Float32Array(totalLength);
          arr[0] = k;
          let offset = 1;

          for (const cf of contourFrames) {
            const frameNum = Number(cf.frameKey);
            arr[offset++] = Number.isFinite(frameNum) ? frameNum : 0;

            const nContours = cf.contours.length;
            arr[offset++] = nContours;

            const lengths = cf.contours.map((c) => c.length);
            for (let i = 0; i < lengths.length; i++) {
              arr[offset++] = lengths[i];
            }

            for (const c of cf.contours) {
              for (let i = 0; i < c.length; i++) {
                const v = c[i];
                arr[offset++] = Number.isFinite(v) ? v : 0;
              }
            }
          }

          const buf = Buffer.from(
            arr.buffer,
            arr.byteOffset,
            arr.byteLength,
          );

          writeOps.push(
            fs
              .writeFile(absPath, buf)
              .catch((error) => {
                console.error(
                  "JSONPersistenceModule: failed to save contours binary",
                  absPath,
                  error,
                );
              }),
          );

          // Store only the binary reference in JSON for all keyframes that had contours
          for (const cf of contourFrames) {
            const data: any = (keyframes as any)[cf.frameKey];
            if (!data || typeof data !== "object") continue;
            data.contoursBinPath = relPath;
            delete data.contours;
          }
        }
      }
    }

    if (writeOps.length > 0) {
      await Promise.all(writeOps);
    }
  }

  private async saveLocalManifestsAndPreprocessorsFromPayload(
    payload: any,
  ): Promise<void> {
    if (!payload || typeof payload !== "object") return;

    const manifests =
      payload && typeof payload.manifests === "object"
        ? (payload.manifests as Record<string, unknown>)
        : undefined;
    const preprocessors =
      payload && typeof payload.preprocessors === "object"
        ? (payload.preprocessors as Record<string, unknown>)
        : undefined;

    if (!manifests && !preprocessors) return;

    await this.ensureLocalDirs();

    const ops: Array<Promise<void>> = [];

    if (manifests) {
      for (const [keyRaw, manifest] of Object.entries(manifests)) {
        const key = String(keyRaw || "").trim();
        if (!key) continue;
        const filePath = this.getManifestLocalPath(key);
        ops.push(
          fs
            .writeFile(filePath, JSON.stringify(manifest, null, 2), "utf8")
            .catch((error) => {
              console.error(
                "JSONPersistenceModule: failed to save local manifest",
                key,
                error,
              );
            }),
        );
      }
    }

    if (preprocessors) {
      for (const [idRaw, preprocessor] of Object.entries(preprocessors)) {
        const id = String(idRaw || "").trim();
        if (!id) continue;
        const filePath = this.getPreprocessorLocalPath(id);
        ops.push(
          fs
            .writeFile(filePath, JSON.stringify(preprocessor, null, 2), "utf8")
            .catch((error) => {
              console.error(
                "JSONPersistenceModule: failed to save local preprocessor",
                id,
                error,
              );
            }),
        );
      }
    }

    if (ops.length > 0) {
      await Promise.all(ops);
    }
  }

  private async hydrateDocWithLocalManifestsAndPreprocessors(
    doc: ProjectJsonV1,
  ): Promise<void> {
    try {
      const clips = Array.isArray(doc.timeline?.clips)
        ? doc.timeline.clips
        : [];

      const manifestKeys = new Set<string>();
      const preprocessorIds = new Set<string>();

      for (const c of clips) {
        if (!c || typeof c !== "object") continue;
        const anyClip: any = c;
        const manifestRef = anyClip.manifestRef;
        if (typeof manifestRef === "string" && manifestRef.length > 0) {
          const vRaw = anyClip.manifestVersion;
          const v =
            typeof vRaw === "string" && vRaw.trim().length > 0
              ? vRaw.trim()
              : undefined;
          if (v) {
            // Prefer a versioned key when possible to keep per-clip manifests
            // in sync with the version they were saved with.
            manifestKeys.add(`${manifestRef}@@${v}`);
          }
          // Backwards compatibility: also try the unversioned key.
          manifestKeys.add(manifestRef);
        }

        if (Array.isArray(anyClip.preprocessors)) {
          for (const p of anyClip.preprocessors as any[]) {
            if (!p || typeof p !== "object") continue;
            const anyPre: any = p;
            const preIdRaw =
              anyPre.preprocessorId ?? anyPre.id ?? anyPre.name ?? "";
            const preId =
              typeof preIdRaw === "string"
                ? preIdRaw
                : preIdRaw != null
                  ? String(preIdRaw)
                  : "";
            if (preId) {
              preprocessorIds.add(preId);
            }
          }
        }
      }

      if (manifestKeys.size === 0 && preprocessorIds.size === 0) {
        return;
      }

      await this.ensureLocalDirs();

      const manifests: Record<string, unknown> = {};
      const preprocessors: Record<string, unknown> = {};

      for (const key of manifestKeys) {
        const filePath = this.getManifestLocalPath(key);
        try {
          const raw = await fs.readFile(filePath, "utf8");
          manifests[key] = JSON.parse(raw);
        } catch {
          // Missing or invalid manifest file – skip
        }
      }

      for (const id of preprocessorIds) {
        const filePath = this.getPreprocessorLocalPath(id);
        try {
          const raw = await fs.readFile(filePath, "utf8");
          preprocessors[id] = JSON.parse(raw);
        } catch {
          // Missing or invalid preprocessor file – skip
        }
      }

      if (Object.keys(manifests).length > 0) {
        doc.manifests = manifests;
      }
      if (Object.keys(preprocessors).length > 0) {
        doc.preprocessors = preprocessors;
      }
    } catch (error) {
      console.error(
        "JSONPersistenceModule: failed to hydrate local manifests/preprocessors",
        error,
      );
    }
  }

  private async hydrateDocWithMaskBinaries(
    projectId: number,
    doc: ProjectJsonV1,
  ): Promise<void> {
    try {
      const clips = Array.isArray(doc.timeline?.clips)
        ? doc.timeline.clips
        : [];
      if (clips.length === 0) return;

      for (const clip of clips) {
        if (!clip || typeof clip !== "object") continue;
        const anyClip: any = clip;
        const masks: any[] = Array.isArray(anyClip.masks)
          ? anyClip.masks
          : [];
        if (masks.length === 0) continue;

        for (const mask of masks) {
          if (!mask || typeof mask !== "object") continue;
          const anyMask: any = mask;
          const keyframes = anyMask.keyframes;
          if (!keyframes || typeof keyframes !== "object") continue;

          const entries = Object.entries(keyframes as any);

          // Determine, per mask, if we have aggregated lasso/contours binaries.
          let lassoRelPath: string | undefined;
          let contoursRelPath: string | undefined;

          for (const [, rawData] of entries) {
            if (!rawData || typeof rawData !== "object") continue;
            const data: any = rawData;
            if (
              !lassoRelPath &&
              typeof data.lassoPointsBinPath === "string" &&
              data.lassoPointsBinPath.length > 0
            ) {
              lassoRelPath = data.lassoPointsBinPath;
            }
            if (
              !contoursRelPath &&
              typeof data.contoursBinPath === "string" &&
              data.contoursBinPath.length > 0
            ) {
              contoursRelPath = data.contoursBinPath;
            }
          }

          // Hydrate lassoPoints for all keyframes in this mask from a single binary, if present.
          if (lassoRelPath) {
            const absPath = this.getMaskBinaryAbsolutePath(lassoRelPath);
            const fileName = path.basename(lassoRelPath);
            const nameCore = fileName
              .replace(/^mask-/, "")
              .replace(/\.f32\.bin$/, "");
            const parts = nameCore.split("-");
            const kindSegment = parts[parts.length - 1] ?? "";
            const maybeFrameSegment = parts[parts.length - 2] ?? "";
            // Old per-keyframe files had an explicit numeric frame segment
            // right before the kind (lasso/contours). New aggregated files
            // omit that frame segment. Clip/mask ids may contain hyphens, so
            // we only inspect the second-to-last segment for a pure integer.
            const isOldPerFrameFormat = /^[0-9]+$/.test(maybeFrameSegment);

            try {
              const buf = await fs.readFile(absPath);
              if (buf.byteLength >= 4) {
                const arr = new Float32Array(
                  buf.buffer,
                  buf.byteOffset,
                  Math.floor(buf.byteLength / 4),
                );
                if (!isOldPerFrameFormat) {
                  // New aggregated per-mask layout for lasso:
                  // [nKeyframes, frame, len, frame, len, ..., values...]
                  if (arr.length >= 1) {
                    const k = Math.max(0, Math.floor(arr[0]));
                    const headerSize = 1 + k * 2;
                    if (k > 0 && arr.length >= headerSize) {
                      const frames: number[] = new Array(k);
                      const lengths: number[] = new Array(k);
                      let totalLen = 0;
                      for (let i = 0; i < k; i++) {
                        const frameNum = Math.floor(arr[1 + i * 2]);
                        const len = Math.max(
                          0,
                          Math.floor(arr[1 + i * 2 + 1]),
                        );
                        frames[i] = frameNum;
                        lengths[i] = len;
                        totalLen += len;
                      }
                      if (headerSize + totalLen <= arr.length) {
                        const frameToPoints = new Map<string, number[]>();
                        let valuesOffset = headerSize;
                        for (let i = 0; i < k; i++) {
                          const len = lengths[i];
                          if (len <= 0) continue;
                          const end = valuesOffset + len;
                          if (end > arr.length) break;
                          const slice = arr.subarray(valuesOffset, end);
                          valuesOffset = end;
                          const frameKey = String(frames[i]);
                          frameToPoints.set(frameKey, Array.from(slice));
                        }

                        for (const [frameKey, rawData] of entries) {
                          if (!rawData || typeof rawData !== "object") continue;
                          const data: any = rawData;
                          const frameNum = Number(frameKey);
                          const key =
                            Number.isFinite(frameNum) ?
                              String(frameNum)
                            : frameKey;
                          const pts = frameToPoints.get(key);
                          if (pts && pts.length > 0) {
                            data.lassoPoints = pts;
                          }
                        }
                      }
                    }
                  }
                } else {
                  // Old per-keyframe format: the entire array is just the lasso points.
                  const pts = Array.from(arr);
                  for (const [, rawData] of entries) {
                    if (!rawData || typeof rawData !== "object") continue;
                    const data: any = rawData;
                    if (data.lassoPointsBinPath === lassoRelPath) {
                      data.lassoPoints = pts;
                    }
                  }
                }
              }
            } catch {
              // Missing or invalid binary – leave lassoPoints undefined.
            }
          }

          // Hydrate contours for all keyframes in this mask from a single binary, if present.
          if (contoursRelPath) {
            const absPath = this.getMaskBinaryAbsolutePath(contoursRelPath);
            const fileName = path.basename(contoursRelPath);
            const nameCore = fileName
              .replace(/^mask-/, "")
              .replace(/\.f32\.bin$/, "");
            const parts = nameCore.split("-");
            const kindSegment = parts[parts.length - 1] ?? "";
            const maybeFrameSegment = parts[parts.length - 2] ?? "";
            const isOldPerFrameFormat = /^[0-9]+$/.test(maybeFrameSegment);

            try {
              const buf = await fs.readFile(absPath);
              if (buf.byteLength >= 4) {
                const arr = new Float32Array(
                  buf.buffer,
                  buf.byteOffset,
                  Math.floor(buf.byteLength / 4),
                );
                if (!isOldPerFrameFormat) {
                  // New aggregated per-mask layout for contours:
                  // [nKeyframes,
                  //   frame, nContours, len0, len1, ..., values...,
                  //   frame, nContours, len0, len1, ..., values..., ...]
                  if (arr.length >= 1) {
                    const k = Math.max(0, Math.floor(arr[0]));
                    let offset = 1;
                    const frameToContours = new Map<string, number[][]>();

                    for (let i = 0; i < k; i++) {
                      if (offset + 2 > arr.length) break;
                      const frameNum = Math.floor(arr[offset++]);
                      const nContours = Math.max(
                        0,
                        Math.floor(arr[offset++]),
                      );
                      if (offset + nContours > arr.length) break;

                      const lengths: number[] = [];
                      for (let j = 0; j < nContours; j++) {
                        lengths.push(Math.max(0, Math.floor(arr[offset++])));
                      }

                      const valuesCount = lengths.reduce(
                        (sum, len) => sum + len,
                        0,
                      );
                      if (offset + valuesCount > arr.length) break;

                      const contours: number[][] = [];
                      for (let j = 0; j < nContours; j++) {
                        const len = lengths[j];
                        if (len <= 0) {
                          contours.push([]);
                          continue;
                        }
                        const slice = arr.subarray(offset, offset + len);
                        contours.push(Array.from(slice));
                        offset += len;
                      }

                      const frameKey = String(frameNum);
                      frameToContours.set(frameKey, contours);
                    }

                    for (const [frameKey, rawData] of entries) {
                      if (!rawData || typeof rawData !== "object") continue;
                      const data: any = rawData;
                      const frameNum = Number(frameKey);
                      const key =
                        Number.isFinite(frameNum) ?
                          String(frameNum)
                        : frameKey;
                      const contours = frameToContours.get(key);
                      if (contours && contours.length > 0) {
                        data.contours = contours;
                      }
                    }
                  }
                } else {
                  // Old per-keyframe format for contours:
                  // [nContours, len0, len1, ..., values...]
                  if (arr.length >= 1) {
                    const nContours = Math.max(0, Math.floor(arr[0]));
                    if (arr.length >= 1 + nContours) {
                      const lengths: number[] = [];
                      for (let i = 0; i < nContours; i++) {
                        lengths.push(Math.max(0, Math.floor(arr[1 + i])));
                      }
                      let offset = 1 + nContours;
                      const contours: number[][] = [];
                      for (let i = 0; i < nContours; i++) {
                        const len = lengths[i];
                        if (len <= 0) {
                          contours.push([]);
                          continue;
                        }
                        if (offset + len > arr.length) break;
                        const slice = arr.subarray(offset, offset + len);
                        contours.push(Array.from(slice));
                        offset += len;
                      }
                      for (const [, rawData] of entries) {
                        if (!rawData || typeof rawData !== "object") continue;
                        const data: any = rawData;
                        if (data.contoursBinPath === contoursRelPath) {
                          data.contours = contours;
                        }
                      }
                    }
                  }
                }
              }
            } catch {
              // Missing or invalid binary – leave contours undefined.
            }
          }
        }
      }
    } catch (error) {
      console.error(
        "JSONPersistenceModule: failed to hydrate mask binary data",
        error,
      );
    }
  }

  private registerHandlers() {
    // Save full JSON snapshot for a specific project
    ipcMain.handle(
      "projects:save-json",
      async (
        _event,
        projectId: number,
        payload: any,
      ): Promise<JsonResponse<{ path: string }>> => {
        try {
          const id = Number(projectId);
          if (!Number.isInteger(id) || id <= 0) {
            return { success: false, error: "Invalid project id" };
          }
          if (!payload || typeof payload !== "object") {
            return {
              success: false,
              error: "Invalid project JSON payload",
            };
          }

          await this.saveLocalManifestsAndPreprocessorsFromPayload(payload);

          // Clone the payload so we can safely mutate heavy mask keyframe data
          // when serializing into external Float32Array binaries.
          const payloadClone: any = { ...payload };

          // Persist mask keyframes that contain large lassoPoints/contours
          // arrays into .bin files and replace the JSON arrays with path
          // references to keep the on-disk JSON compact.
          await this.saveMaskDataBinariesFromPayload(id, payloadClone);

          const baseDoc = ensureProjectJson(
            payloadClone,
            id,
            this.#appVersion ?? undefined,
          );
          baseDoc.meta.lastModified = Date.now();
          const filePath = this.getProjectPath(id);

          await fs.writeFile(
            filePath,
            JSON.stringify(baseDoc, null, 2),
            "utf8",
          );

          return { success: true, data: { path: filePath } };
        } catch (error) {
          console.error("JSONPersistenceModule: failed to save project JSON", error);
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to save project JSON",
          };
        }
      },
    );

    // Load full JSON snapshot for a specific project
    ipcMain.handle(
      "projects:load-json",
      async (
        _event,
        projectId: number,
      ): Promise<JsonResponse<ProjectJsonV1 | null>> => {
        try {
          const id = Number(projectId);
          if (!Number.isInteger(id) || id <= 0) {
            return { success: false, error: "Invalid project id" };
          }

          const filePath = this.getProjectPath(id);
          let raw: string;
          try {
            raw = await fs.readFile(filePath, "utf8");
          } catch (err: any) {
            if (err && (err as any).code === "ENOENT") {
              // No JSON file yet for this project – not an error.
              return { success: true, data: null };
            }
            throw err;
          }

          let parsed: any;
          try {
            parsed = JSON.parse(raw);
          } catch {
            return {
              success: false,
              error: "Failed to parse project JSON file",
            };
          }

          const doc = ensureProjectJson(
            parsed,
            id,
            this.#appVersion ?? undefined,
          );
          await this.hydrateDocWithLocalManifestsAndPreprocessors(doc);
          await this.hydrateDocWithMaskBinaries(id, doc);
          return { success: true, data: doc };
        } catch (error) {
          console.error("JSONPersistenceModule: failed to load project JSON", error);
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to load project JSON",
          };
        }
      },
    );

    ipcMain.handle(
      "projects:save-cover",
      async (
        _event,
        projectId: number,
        buffer: Uint8Array,
      ): Promise<JsonResponse<{ path: string }>> => {
        try {
          const id = Number(projectId);
          if (!Number.isInteger(id) || id <= 0) {
            return { success: false, error: "Invalid project id" };
          }
          const coverPath = this.getProjectCoverPath(id);
          await fs.writeFile(coverPath, buffer);
          return { success: true, data: { path: coverPath } };
        } catch (err) {
          console.error("Failed to save project cover", err);
          return { success: false, error: String(err) };
        }
      },
    );

    ipcMain.handle(
      "projects:clear-cover",
      async (
        _event,
        projectId: number,
      ): Promise<JsonResponse<{ ok: true }>> => {
        try {
          const id = Number(projectId);
          if (!Number.isInteger(id) || id <= 0) {
            return { success: false, error: "Invalid project id" };
          }
          const coverPath = this.getProjectCoverPath(id);
          try {
            await fs.unlink(coverPath);
          } catch (err: any) {
            if (err && err.code === "ENOENT") {
              // Already cleared
            } else {
              throw err;
            }
          }
          return { success: true, data: { ok: true } };
        } catch (err) {
          console.error("Failed to clear project cover", err);
          return { success: false, error: String(err) };
        }
      },
    );

    // List projects using only lightweight metadata from JSON files.
    ipcMain.handle(
      "projects:list-json",
      async (): Promise<
        JsonResponse<
          Array<{
            id: number;
            name: string;
            fps: number;
            folderUuid: string;
            aspectRatio?: { width: number; height: number; id: string };
            createdAt?: number;
            lastModified?: number;
            coverPath?: string;
          }>
        >
      > => {
        try {
          const entries = await fs.readdir(this.projectsDir);
          
          const coversDir = path.join(this.projectsDir, "covers");
          const existingCovers = new Set<string>();
          try {
            const coverFiles = await fs.readdir(coversDir);
            for (const f of coverFiles) existingCovers.add(f);
          } catch {
            // ignore
          }

          const projects: Array<{
            id: number;
            name: string;
            fps: number;
            folderUuid: string;
            aspectRatio?: { width: number; height: number; id: string };
            createdAt?: number;
            lastModified?: number;
            coverPath?: string;
          }> = [];

          for (const entry of entries) {
            const match = /^project-(\d+)\.json$/i.exec(entry);
            if (!match) continue;
            const id = Number(match[1]);
            if (!Number.isInteger(id) || id <= 0) continue;

            const filePath = this.getProjectPath(id);
            let raw: string;
            try {
              raw = await fs.readFile(filePath, "utf8");
            } catch {
              // Skip unreadable files
              continue;
            }

            let parsed: any;
            try {
              parsed = JSON.parse(raw);
            } catch {
              // Skip invalid JSON
              continue;
            }

            const doc = ensureProjectJson(
              parsed,
              id,
              this.#appVersion ?? undefined,
            );

            const name =
              typeof doc.meta?.name === "string" && doc.meta.name.length > 0
                ? doc.meta.name
                : "Untitled Project";
            const fps =
              typeof doc.settings?.fps === "number" &&
              Number.isFinite(doc.settings.fps)
                ? doc.settings.fps
                : 24;

            const aspectRatio =
              (doc.settings as any)?.aspectRatio ??
              undefined;

            const folderUuid =
              typeof doc.meta?.id === "string" && doc.meta.id.length > 0
                ? doc.meta.id
                : `project-${id}`;
            
            const createdAt = doc.meta.createdAt;
            const lastModified = doc.meta.lastModified;

            const coverFilename = `project-${id}.jpg`;
            const coverPath = existingCovers.has(coverFilename)
              ? path.join(coversDir, coverFilename)
              : undefined;

            projects.push({
              id,
              name,
              fps,
              folderUuid,
              aspectRatio,
              createdAt,
              lastModified,
              coverPath,
            });
          }

          // Sort by last modified descending (newest first)
          projects.sort((a, b) => (b.lastModified ?? 0) - (a.lastModified ?? 0));

          return { success: true, data: projects };
        } catch (error) {
          console.error(
            "JSONPersistenceModule: failed to list JSON projects",
            error,
          );
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to list JSON projects",
          };
        }
      },
    );

    // Create a new empty project backed by a JSON file.
    ipcMain.handle(
      "projects:create-json",
      async (
        _event,
        payload: { name: string; fps: number },
      ): Promise<
        JsonResponse<{
          id: number;
          name: string;
          fps: number;
          folderUuid: string;
          aspectRatio?: { width: number; height: number; id: string };
        }>
      > => {
        try {
          const name = String(payload?.name ?? "").trim() || "Project 1";
          const fps = Number(payload?.fps ?? 24);

          if (!Number.isFinite(fps) || fps <= 0) {
            return { success: false, error: "FPS must be a positive number" };
          }

          // Determine next available numeric project id from existing files.
          const entries = await fs.readdir(this.projectsDir);
          let maxId = 0;
          for (const entry of entries) {
            const match = /^project-(\d+)\.json$/i.exec(entry);
            if (!match) continue;
            const id = Number(match[1]);
            if (Number.isInteger(id) && id > maxId) {
              maxId = id;
            }
          }
          const id = maxId > 0 ? maxId + 1 : 1;

          const now = Date.now();
          const raw = {
            version: "1.0.0",
            meta: {
              id: String(randomUUID()),
              name,
              createdAt: now,
              lastModified: now,
            },
            settings: {
              fps,
              aspectRatio: {
                width: 16,
                height: 9,
                id: "16:9",
              },
              defaultClipLength: 5,
            },
            editorState: {},
            assets: {},
            timeline: {
              tracks: [] as any[],
              clips: [] as any[],
            },
          };

          const baseDoc = ensureProjectJson(
            raw,
            id,
            this.#appVersion ?? undefined,
          );
          const filePath = this.getProjectPath(id);

          await fs.writeFile(
            filePath,
            JSON.stringify(baseDoc, null, 2),
            "utf8",
          );

          const folderUuid =
            typeof baseDoc.meta?.id === "string" &&
            baseDoc.meta.id.length > 0
              ? baseDoc.meta.id
              : `project-${id}`;

          const aspectRatio =
            (baseDoc.settings as any)?.aspectRatio ?? undefined;

          return {
            success: true,
            data: {
              id,
                name: baseDoc.meta.name,
                fps: baseDoc.settings.fps,
              folderUuid,
              aspectRatio,
            },
          };
        } catch (error) {
          console.error(
            "JSONPersistenceModule: failed to create JSON project",
            error,
          );
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to create JSON project",
          };
        }
      },
    );

    // Delete a project's JSON file.
    ipcMain.handle(
      "projects:delete-json",
      async (
        _event,
        projectId: number,
      ): Promise<JsonResponse<{ id: number }>> => {
        try {
          const id = Number(projectId);
          if (!Number.isInteger(id) || id <= 0) {
            return { success: false, error: "Invalid project id" };
          }

          const filePath = this.getProjectPath(id);
          // Best-effort: read the project JSON first so we can remove its media folder
          // (keyed by meta.id / folderUuid) alongside the project file.
          let folderUuid: string | null = null;
          try {
            const raw = await fs.readFile(filePath, "utf8");
            const parsed = JSON.parse(raw);
            const metaId = parsed?.meta?.id;
            if (typeof metaId === "string" && metaId.trim().length > 0) {
              folderUuid = metaId.trim();
            }
          } catch {
            // ignore (missing/invalid JSON)
          }

          try {
            await fs.unlink(filePath);
          } catch (err: any) {
            if (err && (err as any).code === "ENOENT") {
              // File already removed – treat as success.
              // Still attempt to clean up the media folder if we managed to read it earlier.
              if (folderUuid) {
                try {
                  const safeFolderUuid = folderUuid.replace(/[^a-zA-Z0-9_-]/g, "_");
                  if (safeFolderUuid.length > 0) {
                    const userDataDir = path.dirname(this.projectsDir);
                    const mediaDir = path.join(userDataDir, "media", safeFolderUuid);
                    await fs.rm(mediaDir, { recursive: true, force: true });
                  }
                } catch {
                  // ignore
                }
              }
              return { success: true, data: { id } };
            }
            throw err;
          }

          // Best-effort: delete the project's media folder (<userData>/media/<folderUuid>).
          if (folderUuid) {
            try {
              const safeFolderUuid = folderUuid.replace(/[^a-zA-Z0-9_-]/g, "_");
              if (safeFolderUuid.length > 0) {
                const userDataDir = path.dirname(this.projectsDir);
                const mediaDir = path.join(userDataDir, "media", safeFolderUuid);
                await fs.rm(mediaDir, { recursive: true, force: true });
              }
            } catch {
              // ignore
            }
          }

          return { success: true, data: { id } };
        } catch (error) {
          console.error(
            "JSONPersistenceModule: failed to delete JSON project",
            error,
          );
          return {
            success: false,
            error:
              error instanceof Error
                ? error.message
                : "Failed to delete JSON project",
          };
        }
      },
    );
  }
}

export function jsonPersistenceModule(
  ...args: ConstructorParameters<typeof JSONPersistenceModule>
) {
  return new JSONPersistenceModule(...args);
}



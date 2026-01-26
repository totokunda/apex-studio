import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ensureFfmpegInstalled,
  extractServerBundle,
  listServerReleaseBundles,
  launchMainWindow,
  onInstallerProgress,
  onInstallerSetupProgress,
  pickMediaPaths,
  resolvePath,
  runSetupScript,
  setInstallerActive,
  setApiPathSetting,
  setMaskModelSetting,
  setRenderImageStepsSetting,
  setRenderVideoStepsSetting,
  getPythonStatus,
  startPythonApi,
  stopPythonApi,
} from "@app/preload";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ProgressBar } from "@/components/common/ProgressBar";
import { LuCheck, LuLoader, LuX } from "react-icons/lu";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { ScrollBar } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type InstallerStepId = "version" | "storage" | "render" | "mask" | "install";

type InstallerUIMode = "setup" | "installing" | "done";

type InstallPhaseId =
  | "download_bundle"
  | "extract_bundle"
  | "download_models"
  | "verify_attention"
  | "update_configs";

type InstallPhaseState = {
  status: "pending" | "active" | "completed" | "skipped" | "error";
  percent: number | null; // 0..100
  message: string | null;
};

const MASK_MODEL_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "sam2_tiny", label: "Sam2 Tiny" },
  { value: "sam2_small", label: "Sam2 Small" },
  { value: "sam2_base_plus", label: "Sam2 Base Plus" },
  { value: "sam2_large", label: "Sam2 Large" },
];

const Installer: React.FC<{ hasBackend: boolean; setShowInstaller: (show: boolean) => void }> = ({ hasBackend, setShowInstaller }) => {
  const steps = useMemo(
    () =>
      [
        { id: "version" as const, label: "Version" },
        { id: "storage" as const, label: "Storage" },
        { id: "render" as const, label: "Render" },
        { id: "mask" as const, label: "Models" },
        { id: "install" as const, label: "Install" },
      ] satisfies Array<{ id: InstallerStepId; label: string }>,
    [],
  );

  const [activeStep, setActiveStep] = useState<InstallerStepId>("version");
  const appendSubdir = (basePath: string, subdir: string) => {
    const base = String(basePath || "").trim();
    if (!base) return subdir;
    const sep = base.includes("\\") ? "\\" : "/";
    const trimmed = base.replace(/[\\/]+$/, "");
    return `${trimmed}${sep}${subdir}`;
  };

  const [saveLocation, setSaveLocation] = useState<string>(() =>
    resolvePath("~"),
  );
  const [codeLocation, setCodeLocation] = useState<string>(() =>
    appendSubdir(resolvePath("~"), "apex-server"),
  );


  useEffect(() => {
    // If an existing installation is running (python API running), stop it.
    (async () => {
      try {
        if (!hasBackend) return;
        const st = await getPythonStatus();
        if (!st?.success || !st.data) return;
        if (st.data.status !== "running" && st.data.status !== "starting") return;

        const res = await stopPythonApi();
        if (res?.success) {
          console.log("Backend stopped");
        } else {
          console.error("Failed to stop backend:", res?.error);
        }
      } catch (e) {
        console.error("Failed to stop backend:", e);
      }
    })();
  }, [hasBackend]);

  const [availableServerBundles, setAvailableServerBundles] = useState<
    Array<{
      tag: string;
      tagVersion: string;
      assetVersion: string;
      assetName: string;
      downloadUrl: string;
      platform: string;
      arch: string;
      device: string;
      pythonTag: string;
      publishedAt?: string;
      prerelease?: boolean;
    }>
  >([]);
  const [serverBundlesHost, setServerBundlesHost] = useState<{
    platform: string;
    arch: string;
  } | null>(null);
  const [serverBundlesLoading, setServerBundlesLoading] =
    useState<boolean>(true);
  const [serverBundlesError, setServerBundlesError] = useState<string | null>(
    null,
  );
  const [selectedServerBundleKey, setSelectedServerBundleKey] = useState<
    string | null
  >(null);

  const [localBundlePath, setLocalBundlePath] = useState<string>("");
  const [localBundleError, setLocalBundleError] = useState<string | null>(null);
  const [showSelectedBundleDetails, setShowSelectedBundleDetails] =
    useState<boolean>(false);
  const [renderImage, setRenderImage] = useState<boolean>(false);
  const [renderVideo, setRenderVideo] = useState<boolean>(false);
  const [maskModel, setMaskModel] = useState<string>("sam2_base_plus");
  const [installRifeFrameInterpolation, setInstallRifeFrameInterpolation] =
    useState<boolean>(true);
  const [installing, setInstalling] = useState<boolean>(false);
  const [installStatus, setInstallStatus] = useState<string | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);
  const [installJobId, setInstallJobId] = useState<string | null>(null);

  const [uiMode, setUiMode] = useState<InstallerUIMode>("setup");
  // When the installer is rendered automatically (e.g. because !hasBackend), the parent may
  // switch away as soon as the backend becomes available. Once an install starts we force
  // the installer to stay mounted until the user explicitly presses "Launch".
  const requestCloseInstaller = () => {
    if (installing || uiMode === "installing") return;
    setShowInstaller(false);
  };
  const [backendStatus, setBackendStatus] = useState<
    "idle" | "starting" | "started" | "error"
  >("idle");
  const [backendError, setBackendError] = useState<string | null>(null);
  const phaseList = useMemo(
    () =>
      [
        { id: "download_bundle" as const, label: "Download Bundle" },
        { id: "extract_bundle" as const, label: "Extract Bundle" },
        { id: "download_models" as const, label: "Download Models" },
        { id: "verify_attention" as const, label: "Verify Attention Backends" },
        { id: "update_configs" as const, label: "Update Configs" },
      ] satisfies Array<{ id: InstallPhaseId; label: string }>,
    [],
  );
  const [activePhase, setActivePhase] =
    useState<InstallPhaseId>("download_bundle");
  const activePhaseRef = useRef<InstallPhaseId>(activePhase);
  useEffect(() => {
    activePhaseRef.current = activePhase;
  }, [activePhase]);
  const phaseStateRef = useRef<Record<InstallPhaseId, InstallPhaseState> | null>(null);
  const [phaseState, setPhaseState] = useState<Record<InstallPhaseId, InstallPhaseState>>(() => ({
    download_bundle: { status: "pending", percent: null, message: null },
    extract_bundle: { status: "pending", percent: null, message: null },
    download_models: { status: "pending", percent: null, message: null },
    verify_attention: { status: "pending", percent: null, message: null },
    update_configs: { status: "pending", percent: null, message: null },
  }));
  useEffect(() => {
    phaseStateRef.current = phaseState;
  }, [phaseState]);

  // Attention backend verification emits one event per backend; keep a small rolling log.
  const attentionEventsRef = useRef<string[]>([]);
  const [attentionListVersion, setAttentionListVersion] = useState<number>(0);
  const appendAttentionEvent = (line: string) => {
    const s = String(line || "").trim();
    if (!s) return;
    const prev = attentionEventsRef.current;
    // Avoid spamming duplicates when backend retries.
    if (prev.length > 0 && prev[prev.length - 1] === s) return;
    attentionEventsRef.current = prev.concat([s]).slice(-200);
    setAttentionListVersion((v) => v + 1);
  };

  const lastDownloadUiUpdateAtRef = useRef<number>(0);
  const lastDownloadPctRef = useRef<number | null>(null);
  const lastExtractUiUpdateAtRef = useRef<number>(0);
  const lastExtractPctRef = useRef<number | null>(null);

  const stopPythonApiIfRunning = async (opts?: { reason?: string }) => {
    const reason = String(opts?.reason || "").trim();
    if (!hasBackend) return;
    const statusRes = await getPythonStatus().catch(() => null);
    const status = statusRes?.success ? statusRes.data?.status : null;

    if (status !== "running" && status !== "starting") return;

    // If an install already exists and the python API is running, we must stop it
    // before extracting/overwriting the runtime/code on disk.
    setInstallStatus(reason ? `Stopping backend (${reason})…` : "Stopping backend…");
    const stopRes = await stopPythonApi().catch(() => null);
    if (!stopRes?.success) return;

    // Best-effort: wait briefly for the process to fully stop.
    const deadline = Date.now() + 12_000;
    while (Date.now() < deadline) {
      const st = await getPythonStatus().catch(() => null);
      if (st?.success && st.data?.status === "stopped") return;
      await new Promise<void>((resolve) => window.setTimeout(resolve, 250));
    }
  };

  const resetToReadyToInstall = (opts?: { keepError?: boolean }) => {
    setUiMode("setup");
    setActiveStep("install");
    setActivePhase("download_bundle");
    setInstallJobId(null);
    if (!opts?.keepError) {
      setInstallError(null);
      setInstallStatus(null);
    }
    setBackendStatus("idle");
    setBackendError(null);
    setPhaseState({
      download_bundle: { status: "pending", percent: null, message: null },
      extract_bundle: { status: "pending", percent: null, message: null },
      download_models: { status: "pending", percent: null, message: null },
      verify_attention: { status: "pending", percent: null, message: null },
      update_configs: { status: "pending", percent: null, message: null },
    });
  };

  const patchPhase = (
    id: InstallPhaseId,
    patch: Partial<InstallPhaseState>,
  ) => {
    setPhaseState((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  };

  const phaseOrder: InstallPhaseId[] = useMemo(
    () => [
      "download_bundle",
      "extract_bundle",
      "download_models",
      "verify_attention",
      "update_configs",
    ],
    [],
  );
  const phaseIndex = (id: InstallPhaseId) => phaseOrder.indexOf(id);

  const activatePhaseExclusive = (
    id: InstallPhaseId,
    patch?: Partial<InstallPhaseState>,
  ) => {
    const idx = phaseIndex(id);
    setPhaseState((prev) => {
      const next = { ...prev };
      for (const pid of phaseOrder) {
        const st = next[pid];
        if (pid === id) {
          // Do not override terminal states.
          if (st.status !== "completed" && st.status !== "skipped" && st.status !== "error") {
            next[pid] = { ...st, status: "active", ...(patch || {}) };
          } else if (patch) {
            next[pid] = { ...st, ...patch };
          }
          continue;
        }
        if (st.status !== "active") continue;
        // Ensure only one active phase. When advancing forward, mark prior phase completed.
        if (phaseIndex(pid) < idx) {
          next[pid] = {
            ...st,
            status: "completed",
            percent: st.percent ?? 100,
          };
        } else {
          // Defensive: future phase should not be active yet.
          next[pid] = { ...st, status: "pending" };
        }
      }
      return next;
    });
    if (activePhaseRef.current !== id) setActivePhase(id);
  };

  const completePhase = (id: InstallPhaseId, patch?: Partial<InstallPhaseState>) => {
    patchPhase(id, {
      status: "completed",
      percent: 100,
      ...(patch || {}),
    });
  };

  const activeIndex = useMemo(
    () => Math.max(0, steps.findIndex((s) => s.id === activeStep)),
    [activeStep, steps],
  );

  const canGoBack = activeIndex > 0;
  const canGoNext = activeIndex >= 0 && activeIndex < steps.length - 1;

  const goBack = () => {
    if (!canGoBack) return;
    setActiveStep(steps[activeIndex - 1]!.id);
  };

  const goNext = () => {
    if (!canGoNext) return;
    setActiveStep(steps[activeIndex + 1]!.id);
  };

  const fieldLabelClass =
    "flex flex-col gap-0.5 text-[12px] font-medium text-brand-light/90";

  const pickSaveLocation = async () => {
    const picked = await pickMediaPaths({
      directory: true,
      title: "Choose a save location",
      defaultPath: saveLocation || undefined,
    });
    const first = Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
    if (first) setSaveLocation(first);
  };

  const pickCodeLocation = async () => {
    const picked = await pickMediaPaths({
      directory: true,
      title: "Choose a code location",
      defaultPath: codeLocation || undefined,
    });
    const first = Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
    if (first) setCodeLocation(appendSubdir(first, "apex-studio"));
  };

  const parsePythonApiAssetName = (
    filename: string,
  ):
    | {
        assetVersion: string;
        platform: string;
        arch: string;
        device: string;
        pythonTag: string;
      }
    | null => {
    const re =
      /^python-api-(?<version>[^-]+)-(?<platform>[^-]+)-(?<arch>[^-]+)-(?<device>[^-]+)-(?<python>[^.]+)\.tar\.zst$/i;
    const m = re.exec(filename);
    if (!m || !m.groups) return null;
    return {
      assetVersion: m.groups.version,
      platform: m.groups.platform,
      arch: m.groups.arch,
      device: m.groups.device,
      pythonTag: m.groups.python,
    };
  };

  const normalizePlatform = (p: string) => {
    const s = String(p || "").trim().toLowerCase();
    if (!s) return s;
    if (s === "windows") return "win32";
    if (s === "macos") return "darwin";
    return s;
  };

  const normalizeArch = (a: string) => {
    const raw = String(a || "").trim().toLowerCase();
    if (!raw) return raw;
    // Normalize separators/aliases commonly seen across Node and release artifacts.
    const s = raw.replace(/[^a-z0-9]+/g, "_");
    if (s === "x64" || s === "amd64" || s === "x86_64") return "x86_64";
    if (s === "arm64" || s === "aarch64") return "arm64";
    if (s === "ia32" || s === "x86" || s === "i386") return "x86";
    return s;
  };

  const validateLocalBundlePath = (p: string) => {
    const trimmed = String(p || "").trim();
    if (!trimmed) {
      setLocalBundleError(null);
      return;
    }
    if (!trimmed.toLowerCase().endsWith(".tar.zst")) {
      setLocalBundleError("File must end with .tar.zst");
      return;
    }
    const base = trimmed.split(/[\\/]/).pop() || trimmed;
    const parsed = parsePythonApiAssetName(base);
    if (!parsed) {
      setLocalBundleError(
        "Filename must match python-api-<version>-<platform>-<arch>-<device>-<python>.tar.zst",
      );
      return;
    }
    if (serverBundlesHost) {
      if (
        normalizePlatform(parsed.platform) !== normalizePlatform(serverBundlesHost.platform)
      ) {
        setLocalBundleError(
          `Platform mismatch: file is "${parsed.platform}" but this machine is "${serverBundlesHost.platform}"`,
        );
        return;
      }
      if (normalizeArch(parsed.arch) !== normalizeArch(serverBundlesHost.arch)) {
        setLocalBundleError(
          `Architecture mismatch: file is "${parsed.arch}" but this machine is "${serverBundlesHost.arch}"`,
        );
        return;
      }
    }
    setLocalBundleError(null);
  };

  const pickLocalBundle = async () => {
    const picked = await pickMediaPaths({
      directory: false,
      title: "Choose a .tar.zst bundle",
      defaultPath: localBundlePath || undefined,
      filters: [{ name: "Apex server bundle", extensions: ["zst"] }],
    });
    const first = Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
    if (first) {
      setLocalBundlePath(first);
      validateLocalBundlePath(first);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setServerBundlesLoading(true);
      setServerBundlesError(null);
      try {
        const res = await listServerReleaseBundles();
        if (cancelled) return;
        if (!res?.success || !res.data) {
          setAvailableServerBundles([]);
          setServerBundlesHost(null);
          setServerBundlesError(res?.error || "Failed to load versions");
          setServerBundlesLoading(false);
          return;
        }

        const host = res.data.host;
        setServerBundlesHost(host);
        const items = Array.isArray(res.data.items) ? res.data.items : [];
        // Installer expects a "python-api-*.tar.zst" server bundle (see local bundle validator below).
        // The remote feed may contain other bundle types (e.g. python-code), so filter them out here.
        const filteredItems = items.filter(
          (it) =>
            typeof it?.assetName === "string" &&
            it.assetName.toLowerCase().startsWith("python-api-"),
        );
        const normalized = filteredItems.map((it) => ({
          tag: it.tag,
          tagVersion: it.tagVersion,
          assetVersion: it.assetVersion,
          assetName: it.assetName,
          downloadUrl: it.downloadUrl,
          platform: it.platform,
          arch: it.arch,
          device: it.device,
          pythonTag: it.pythonTag,
          publishedAt: it.publishedAt,
          prerelease: it.prerelease,
        }));

        // Default bundle ordering: prefer CUDA first (then ROCm/MPS), then CPU.
        // This makes "recommended" defaults match GPU-first expectations on capable machines.
        const devicePriority = (d: string) => {
          const s = String(d || "").trim().toLowerCase();
          if (s === "cuda" || s.startsWith("cuda")) return 0;
          if (s === "rocm" || s.startsWith("rocm")) return 1;
          if (s === "mps" || s === "metal") return 2;
          if (s === "cpu") return 3;
          return 9;
        };
        const parseSemver = (v: string) => {
          // Accept "v0.1.0" or "0.1.0" (ignore any suffix)
          const s = String(v || "").trim().replace(/^v/i, "");
          const m = /^(\d+)\.(\d+)\.(\d+)/.exec(s);
          if (!m) return null;
          return [Number(m[1]), Number(m[2]), Number(m[3])] as const;
        };
        const cmpSemverDesc = (a: string, b: string) => {
          const av = parseSemver(a);
          const bv = parseSemver(b);
          if (!av && !bv) return 0;
          if (!av) return 1;
          if (!bv) return -1;
          if (av[0] !== bv[0]) return bv[0] - av[0];
          if (av[1] !== bv[1]) return bv[1] - av[1];
          return bv[2] - av[2];
        };

        const sorted = normalized
          .map((b, idx) => ({ ...b, __idx: idx }))
          .sort((a, b) => {
            const dp = devicePriority(a.device) - devicePriority(b.device);
            if (dp !== 0) return dp;
            // Prefer non-prerelease when otherwise equal.
            const preA = a.prerelease ? 1 : 0;
            const preB = b.prerelease ? 1 : 0;
            if (preA !== preB) return preA - preB;
            const sv = cmpSemverDesc(a.tagVersion, b.tagVersion);
            if (sv !== 0) return sv;
            // Stable fallback: preserve source order
            return a.__idx - b.__idx;
          })
          .map(({ __idx, ...b }) => b);

        setAvailableServerBundles(sorted);
        setSelectedServerBundleKey(
          sorted.length > 0 ? `${sorted[0]!.tag}::${sorted[0]!.assetName}` : null,
        );
        setShowSelectedBundleDetails(false);
        setServerBundlesLoading(false);

        if (localBundlePath) validateLocalBundlePath(localBundlePath);
      } catch (e) {
        if (cancelled) return;
        setAvailableServerBundles([]);
        setServerBundlesHost(null);
        setServerBundlesError(
          e instanceof Error ? e.message : "Failed to load versions",
        );
        setServerBundlesLoading(false);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedServerBundle = useMemo(() => {
    if (!selectedServerBundleKey) return null;
    const [tag, assetName] = selectedServerBundleKey.split("::");
    return (
      availableServerBundles.find(
        (b) => b.tag === tag && b.assetName === assetName,
      ) || null
    );
  }, [availableServerBundles, selectedServerBundleKey]);

  const runInstall = async () => {
    if (installing) return;
    setInstallError(null);
    setInstallStatus(null);
    setBackendStatus("idle");
    setBackendError(null);
    attentionEventsRef.current = [];
    setAttentionListVersion((v) => v + 1);

    const dest = String(codeLocation || "").trim();
    if (!dest) {
      setInstallError("Code Location is required.");
      return;
    }

    const hasLocal = Boolean(String(localBundlePath || "").trim());
    if (!hasLocal && !selectedServerBundle) {
      setInstallError("Please select a server bundle version or choose a local .tar.zst file.");
      return;
    }

    // Tell main process to suppress any backend auto-starts while the installer is running.
    try {
      await setInstallerActive(true, "install");
    } catch {}

    // Make the installer "sticky" in the parent so it cannot be unmounted mid-install when
    // launcher status flips from !hasBackend -> hasBackend.
    setShowInstaller(true);

    // Disable setup navigation immediately (there's a small window before uiMode flips).
    setInstalling(true);

    // Lock the installer UI into phase mode; configuration can no longer be changed.
    setUiMode("installing");

    // Initialize phases
    setPhaseState({
      download_bundle: {
        status: hasLocal ? "skipped" : "active",
        percent: hasLocal ? 100 : null,
        message: hasLocal ? "Using local bundle (download skipped)" : "Starting download…",
      },
      extract_bundle: { status: hasLocal ? "active" : "pending", percent: null, message: null },
      download_models: { status: "pending", percent: null, message: null },
      verify_attention: { status: "pending", percent: null, message: null },
      update_configs: { status: "pending", percent: null, message: null },
    });
    setActivePhase(hasLocal ? "extract_bundle" : "download_bundle");

    const jobId =
      (globalThis.crypto && "randomUUID" in globalThis.crypto
        ? (globalThis.crypto as Crypto).randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`) || null;
    setInstallJobId(jobId);

    let unsubscribeProgress: null | (() => void) = null;
    if (jobId) {
      unsubscribeProgress = onInstallerProgress(jobId, (ev: any) => {
        if (!ev) return;
        if (ev.phase === "download") {
          // Enforce sequential UI: ignore late download events once we've moved past download.
          const dlSt = phaseStateRef.current?.download_bundle?.status;
          if (dlSt === "completed" || dlSt === "skipped") return;
          const now = Date.now();
          const pctIntRaw =
            typeof ev.percent === "number"
              ? Math.round(Math.max(0, Math.min(1, ev.percent)) * 100)
              : null;
          const shouldUpdate =
            (pctIntRaw !== null && pctIntRaw !== lastDownloadPctRef.current) ||
            now - lastDownloadUiUpdateAtRef.current > 900;
          if (!shouldUpdate && !ev.message) return;

          lastDownloadUiUpdateAtRef.current = now;
          if (pctIntRaw !== null) lastDownloadPctRef.current = pctIntRaw;

          activatePhaseExclusive("download_bundle", {
            // Do not clear percent when the event doesn't carry it.
            ...(pctIntRaw !== null ? { percent: pctIntRaw } : {}),
            message: ev.message || "Downloading…",
          });
          if (typeof ev.percent === "number" && ev.percent >= 1) {
            completePhase("download_bundle");
            activatePhaseExclusive("extract_bundle", { message: "Starting extraction…" });
          }
        } else if (ev.phase === "extract") {
          // Enforce sequential UI: ignore late extract events once extraction is done.
          const exSt = phaseStateRef.current?.extract_bundle?.status;
          if (exSt === "completed" || exSt === "skipped") return;
          const now = Date.now();
          const pctIntRaw =
            typeof ev.percent === "number"
              ? Math.round(Math.max(0, Math.min(1, ev.percent)) * 100)
              : null;

          activatePhaseExclusive("extract_bundle");
          const shouldUpdate =
            (pctIntRaw !== null && pctIntRaw !== lastExtractPctRef.current) ||
            now - lastExtractUiUpdateAtRef.current > 1200;
          if (shouldUpdate) {
            lastExtractUiUpdateAtRef.current = now;
            if (pctIntRaw !== null) lastExtractPctRef.current = pctIntRaw;
            patchPhase("extract_bundle", {
              // Do not clear percent when the event doesn't carry it.
              ...(pctIntRaw !== null ? { percent: pctIntRaw } : {}),
              message:
                typeof ev.percent === "number" && ev.percent >= 1
                  ? "Extraction complete"
                  : "Extracting…",
            });
          }
          if (typeof ev.percent === "number" && ev.percent >= 1) {
            completePhase("extract_bundle", { message: "Extraction complete" });
          }
        } else if (ev.phase === "status") {
          const msg = String(ev.message || "");
          // Streamed logs from setup.py (stdout/stderr). We log them to DevTools console
          // but avoid spamming the phase UI message.
          if (msg.startsWith("[setup:")) {
            // eslint-disable-next-line no-console
            console.log(msg);
            return;
          }
          // Detect local bundle download skip emitted by main.
          if (msg.toLowerCase().includes("local bundle")) {
            patchPhase("download_bundle", {
              status: "skipped",
              percent: 100,
              message: msg,
            });
            activatePhaseExclusive("extract_bundle", { message: "Starting extraction…" });
            return;
          }
          // Otherwise attach to the currently active phase.
          patchPhase(activePhaseRef.current, { message: msg || null });
        }
      });
    }

    try {
      // IMPORTANT: stop any running Python backend before modifying/installing the runtime/code.
      // Otherwise extraction/setup can fail or the app may keep using an old runtime.
      try {
        await stopPythonApiIfRunning({ reason: "pre-install" });
      } catch {
        // best-effort; continue install anyway
      }

      setInstallStatus("Running installer…");
      const extractRes = await extractServerBundle({
        source: hasLocal
          ? { kind: "local", path: localBundlePath }
          : {
              kind: "remote",
              url: selectedServerBundle!.downloadUrl,
              assetName: selectedServerBundle!.assetName,
            },
        destinationDir: dest,
        jobId: jobId || undefined,
      });
      if (!extractRes.success) {
        throw new Error(extractRes.error || "Failed to extract server bundle");
      }
      // Extraction is complete at this point; ensure phases are consistent even if
      // progress events were missed or throttled.
      if (!hasLocal) {
        completePhase("download_bundle");
      } else {
        patchPhase("download_bundle", { status: "skipped", percent: 100 });
      }
      completePhase("extract_bundle");

      // Persist basic settings used by the launcher / app defaults.
      // - apiPath is used by Launcher as a "we have something installed on disk" fallback.
      // - mask model + render step toggles are used by the backend once it's running.
      try {
        await setApiPathSetting(dest);
      } catch {}
      try {
        await setMaskModelSetting(maskModel);
      } catch {}
      try {
        await setRenderImageStepsSetting(renderImage);
      } catch {}
      try {
        await setRenderVideoStepsSetting(renderVideo);
      } catch {}

      // Run setup.py for BOTH:
      // - Download Models (mask + optional RIFE)
      // - Update Configs (render step toggles)
      activatePhaseExclusive("download_models", {
        percent: 0,
        message: "Starting setup…",
      });
      patchPhase("verify_attention", {
        status: "pending",
        percent: null,
        message: null,
      });
      patchPhase("update_configs", {
        status: "pending",
        percent: null,
        message: null,
      });

      // Subscribe to setup progress BEFORE starting setup.py.
      // This fixes a reinstall edge-case where setup.py can exit quickly (no-op) and emit
      // the final progress event before the renderer begins listening.
      const awaitSetupCompletion = async (initialJobId: string) => {
        let done = false;
        let unsubscribeSetupProgress: (() => void) | null = null;

        let resolveCompletion!: () => void;
        let rejectCompletion!: (e: Error) => void;
        const completion = new Promise<void>((resolve, reject) => {
          resolveCompletion = resolve;
          rejectCompletion = reject;
        });

        const cleanupListener = () => {
          try {
            if (typeof unsubscribeSetupProgress === "function") unsubscribeSetupProgress();
          } catch {}
          unsubscribeSetupProgress = null;
        };

        const finishOk = () => {
          if (done) return;
          done = true;
          cleanupListener();
          resolveCompletion();
        };

        const finishErr = (err: unknown) => {
          if (done) return;
          done = true;
          cleanupListener();
          rejectCompletion(err instanceof Error ? err : new Error(String(err || "Setup failed")));
        };

        const onPayload = (payload: any) => {
          try {
            const status = String(payload?.status || "processing");
            const message = String(payload?.message || "");
            const progressRaw = payload?.progress;
            const pctOverall =
              typeof progressRaw === "number" && Number.isFinite(progressRaw)
                ? Math.round(Math.max(0, Math.min(1, progressRaw)) * 100)
                : null;
            const task = String(payload?.metadata?.task || "");
            const taskProgressRaw = payload?.metadata?.task_progress;
            const pctTask =
              typeof taskProgressRaw === "number" && Number.isFinite(taskProgressRaw)
                ? Math.round(Math.max(0, Math.min(1, taskProgressRaw)) * 100)
                : null;
            const pct = pctTask ?? pctOverall;

            if (task === "mask" || task === "rife") {
              activatePhaseExclusive("download_models", {
                status: status === "error" ? "error" : "active",
                ...(pct !== null ? { percent: pct } : {}),
                message: message || "Downloading models…",
              });
            } else if (task === "attention") {
              // Attention runs after model downloads; advance phases sequentially.
              if (phaseStateRef.current?.download_models?.status === "active") {
                completePhase("download_models", { percent: 100 });
              }
              activatePhaseExclusive("verify_attention", {
                status: status === "error" ? "error" : "active",
                ...(pct !== null ? { percent: pct } : {}),
                message: message || "Verifying attention backends…",
              });
              appendAttentionEvent(message || "Verifying attention backends…");
            } else if (task === "config") {
              // Config runs after attention verification; keep things sequential.
              if (phaseStateRef.current?.verify_attention?.status === "active") {
                completePhase("verify_attention", { percent: 100 });
              }
              activatePhaseExclusive("update_configs", {
                // Keep update_configs active; we'll only mark it completed after ffmpeg install.
                status: status === "error" ? "error" : "active",
                ...(pct !== null ? { percent: pct } : {}),
                message: message || "Updating config…",
              });
            } else if (task === "setup") {
              if (status === "complete") {
                finishOk();
                return;
              }
            }

            if (status === "error") {
              finishErr(new Error(message || "Setup failed"));
              return;
            }

            if (status === "complete" && message.toLowerCase().includes("setup complete")) {
              finishOk();
            }
          } catch (e) {
            finishErr(e);
          }
        };

        const listenOnJobId = (listenJobId: string) => {
          cleanupListener();
          unsubscribeSetupProgress = onInstallerSetupProgress(listenJobId, onPayload);
        };

        // Start listening immediately (before setup starts).
        listenOnJobId(initialJobId);

        patchPhase("download_models", {
          status: "active",
          message: "Running setup…",
        });

        const setupRes = await runSetupScript({
          apexHomeDir: saveLocation,
          apiInstallDir: dest,
          maskModelType: maskModel,
          installRife: installRifeFrameInterpolation,
          enableImageRenderSteps: renderImage,
          enableVideoRenderSteps: renderVideo,
          jobId: initialJobId,
        });
        if (!setupRes?.success) {
          cleanupListener();
          throw new Error(setupRes?.error || "Failed to start setup");
        }
        const setupJobId = setupRes.data?.jobId;
        if (!setupJobId) {
          cleanupListener();
          throw new Error("Failed to start setup (missing jobId)");
        }
        if (setupJobId !== initialJobId) {
          // Extremely defensive: if main chose a different jobId, switch listener.
          listenOnJobId(setupJobId);
        }

        await completion;
      };

      await awaitSetupCompletion(jobId!);

      // Ensure phases are sequentially finalized even if some tasks were skipped.
      if (phaseStateRef.current?.download_models?.status === "active") {
        completePhase("download_models", { message: "Models installed" });
      } else {
        patchPhase("download_models", {
          ...(phaseStateRef.current?.download_models?.status === "pending"
            ? { status: "skipped", percent: 100 }
            : {}),
          message: phaseStateRef.current?.download_models?.message || "Models installed",
        });
      }
      if (phaseStateRef.current?.verify_attention?.status === "active") {
        completePhase("verify_attention", { message: "Attention verified" });
      } else if (phaseStateRef.current?.verify_attention?.status === "pending") {
        patchPhase("verify_attention", {
          status: "skipped",
          percent: 100,
          message: "Attention verification skipped",
        });
      }

      // Update Configs (ffmpeg install + any config flags already applied by setup.py)
      activatePhaseExclusive("update_configs", {
        message: "Installing ffmpeg…",
      });

      const ffRes = await ensureFfmpegInstalled();
      if (!ffRes.success) {
        throw new Error(ffRes.error || "Failed to install ffmpeg");
      }

      completePhase("update_configs", {
        message: `Completed (ffmpeg: ${ffRes.data?.method})`,
      });
      setUiMode("done");

      // After installation completes, start the Python API from the newly installed location.
      setInstallStatus("Installed. Starting backend…");
      setBackendStatus("starting");
      try {
        await setInstallerActive(false, "install complete");
      } catch {}
      const pyRes = await startPythonApi();
      if (!pyRes?.success || !pyRes.data) {
        const msg = pyRes?.error || "Failed to start backend";
        setBackendStatus("error");
        setBackendError(msg);
        setInstallError(msg);
        return;
      }
      if (pyRes.data.status !== "running") {
        const msg = pyRes.data.error || `Backend did not start (status: ${pyRes.data.status})`;
        setBackendStatus("error");
        setBackendError(msg);
        setInstallError(msg);
        return;
      }
      setBackendStatus("started");
      setInstallStatus(
        `Installed. Backend running at http://${pyRes.data.host}:${pyRes.data.port}`,
      );
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : "Install failed");
      setInstallStatus(null);
      // Mark current phase as errored.
      patchPhase(activePhase, {
        status: "error",
        message: e instanceof Error ? e.message : "Install failed",
      });
      // Return to the setup tabs on any installation failure.
      resetToReadyToInstall({ keepError: true });
    } finally {
      setInstalling(false);
      try {
        await setInstallerActive(false, "installer finished");
      } catch {}
      if (unsubscribeProgress) unsubscribeProgress();
    }
  };

  const renderPhaseIcon = (st: InstallPhaseState) => {
    if (st.status === "active") return <LuLoader className="h-4 w-4 animate-spin" />;
    if (st.status === "completed" || st.status === "skipped")
      return <LuCheck className="h-4 w-4" />;
    if (st.status === "error") return <LuX className="h-4 w-4" />;
    return <span className="h-4 w-4 inline-block rounded-full border border-brand-light/25" />;
  };

  return (
    <main className="w-full h-screen flex flex-col bg-black font-poppins">
      <div className="relative flex-1 overflow-hidden">
        <div className="absolute inset-0 bg-linear-to-br from-slate-950 via-black to-slate-900" />
        <div className="absolute inset-0 backdrop-blur-md bg-black" />

        <div className="relative z-10 h-full w-full flex flex-col items-center justify-center px-6 py-8">
          <div className="w-full max-w-3xl">
            <h3 className="text-2xl font-semibold tracking-tight text-brand-light text-center mb-2">
              Apex Studio
            </h3>
            <div className="text-center mb-6">
              <div className="text-[13px] uppercase tracking-[0.35em] text-brand-light/60">
                Installer
              </div>
            </div>

            <div className="rounded-xl border border-brand-light/10 bg-brand-background-dark backdrop-blur-md shadow-lg">
              {uiMode === "setup" ? (
                <Tabs
                  value={activeStep}
                  onValueChange={(v) => setActiveStep(v as InstallerStepId)}
                  className="w-full"
                >
                  <div className="p-4 border-b border-white/10">
                    <TabsList className="bg-transparent p-0 gap-2 flex flex-wrap justify-center">
                      {steps.map((s, idx) => (
                        <TabsTrigger
                          key={s.id}
                          value={s.id}
                          className="px-3 py-2 text-[11px] text-brand-light/80 data-[state=active]:text-brand-light data-[state=active]:bg-brand/70 rounded-[7px]"
                        >
                          <span className="mr-2 text-[10px] text-brand-light/50">
                            {idx + 1}
                          </span>
                          {s.label}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                  </div>

                  <div className="p-5 min-h-[320px]">
                  <TabsContent value="version" className="m-0">
                    <div className="flex flex-col gap-4 text-brand-light">
                      <div className="flex flex-col gap-1 border-b border-brand-light/8 pb-4">
                        <div className="text-[13px] font-semibold text-brand-light">
                          Available Versions
                        </div>
                        <div className="flex flex-col gap-2 text-[10.5px] text-brand-light/60 font-normal">
                          Select a server bundle version to download for your machine.
                          {serverBundlesHost ? (
                            <span>
                              Detected:{" "}
                              <span className="text-brand-light/75">
                                {serverBundlesHost.platform} / {serverBundlesHost.arch}
                              </span>
                            </span>
                          ) : null}
                        </div>
                      </div>

                      <div className="flex flex-col gap-2.5">
                        <label className={fieldLabelClass}>
                          <span>Downloadable Versions</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            Fetched from Hugging Face and filtered by platform + architecture.
                          </span>
                        </label>

                        {serverBundlesLoading ? (
                          <div className="text-[11px] text-brand-light/60">
                            Loading versions…
                          </div>
                        ) : availableServerBundles.length === 0 ? (
                          <div className="text-[11px] text-brand-light/60">
                            No versions available.
                          </div>
                        ) : (
                          <div className="flex flex-col gap-2">
                            <Select
                              value={selectedServerBundleKey ?? undefined}
                                onValueChange={(v) => {
                                  setSelectedServerBundleKey(v);
                                  setShowSelectedBundleDetails(false);
                                }}
                            >
                              <SelectTrigger
                                size="sm"
                                className="w-full h-8.5! text-[11.5px] bg-brand-background/70 font-medium rounded-[6px] border-white/10 text-brand-light"
                              >
                                <SelectValue placeholder="Select a version" />
                              </SelectTrigger>
                              <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                                {availableServerBundles.map((b) => {
                                  const key = `${b.tag}::${b.assetName}`;
                                  return (
                                    <SelectItem
                                      key={key}
                                      value={key}
                                      className="text-[11px] font-medium"
                                    >
                                      Version {b.tagVersion} · Python API {b.assetVersion} ·{" "}
                                      {b.device.toUpperCase()} · {b.pythonTag}
                                      {b.prerelease ? " (pre-release)" : ""}
                                      
                                    </SelectItem>
                                  );
                                })}
                              </SelectContent>
                            </Select>

                            {selectedServerBundle ? (
                              <div className="w-full">
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="h-8 text-[11px] dark border-white/10 bg-brand-background text-brand-light/90 rounded-[6px] w-fit px-3"
                                  onClick={() =>
                                    setShowSelectedBundleDetails((s) => !s)
                                  }
                                >
                                  {showSelectedBundleDetails
                                    ? "Hide details"
                                    : "Show details"}
                                </Button>

                                {showSelectedBundleDetails ? (
                                  <div className="text-[10.5px] text-brand-light/60 leading-relaxed text-start w-full mt-2">
                                    <div className="flex gap-1 flex-col text-start justify-start items-start mb-2">
                                      <span className="text-brand-light font-medium">
                                        Asset Name
                                      </span>
                                      <span className=" text-brand-light/80">
                                        {selectedServerBundle.assetName}
                                      </span>
                                    </div>
                                    <div className="flex gap-1 flex-col text-start justify-start items-start">
                                      <span className="text-brand-light font-medium text-start">
                                        Download URL
                                      </span>
                                      <span
                                        className="text-brand-light/80 block max-w-full truncate"
                                        title={selectedServerBundle.downloadUrl}
                                      >
                                        {selectedServerBundle.downloadUrl}
                                      </span>
                                    </div>
                                  </div>
                                ) : null}
                              </div>
                            ) : null}
                          </div>
                        )}

                        {serverBundlesError ? (
                          <div className="text-[11px] text-red-300/90">
                            {serverBundlesError}
                          </div>
                        ) : null}
                      </div>

                      <div className="flex flex-col gap-2.5">
                        <label className={fieldLabelClass}>
                          <span>Use a local bundle (.tar.zst)</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            If you already have a server bundle file, choose it from disk.
                          </span>
                        </label>
                        <div className="flex items-center gap-2">
                          <Input
                            value={localBundlePath}
                            onChange={(e) => {
                              const v = e.target.value;
                              setLocalBundlePath(v);
                              validateLocalBundlePath(v);
                            }}
                            placeholder="/path/to/python-api-...tar.zst"
                            className="h-9 text-[12px]!  bg-brand-background/80 border-white/10 text-brand-light rounded-[6px]"
                          />
                          <Button
                            type="button"
                            variant="outline"
                            className="h-9  text-[11.5px] dark border-white/10 bg-brand-background text-brand-light/90 rounded-[6px]"
                            onClick={() => void pickLocalBundle()}
                          >
                            Browse
                          </Button>
                        </div>
                        {localBundleError ? (
                          <div className="text-[11px] text-red-300/90">
                            {localBundleError}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="storage" className="m-0">
                    <div className="flex flex-col gap-4 text-brand-light">
                    <div className="flex flex-col gap-1 border-b border-brand-light/8 pb-4">
                        <div className="text-[13px] font-semibold  text-brand-light">
                          Storage Configuration
                        </div>
                        <div className="flex flex-col gap-2 text-[10.5px] text-brand-light/60 font-normal">
                          Configure where Apex Studio stores its data/resources and where downloaded server bundles are saved.
                        </div>
                      </div>
                      <div className="flex flex-col gap-2.5">
                        <label className={fieldLabelClass}>
                          <span>Save Location</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            This folder will be used as the base location for all API related resources. The path <code className="text-brand-light">apex-diffusion</code> will be appended to the save location.
                          </span>
                        </label>
                        <div className="flex items-center gap-2">
                          <Input
                            value={saveLocation}
                            onChange={(e) => setSaveLocation(e.target.value)}
                            placeholder="/path/to/save-location"
                            className="h-9 text-[12px]!  bg-brand-background/80 border-white/10 text-brand-light rounded-[6px]"
                          />
                          <Button
                            type="button"
                            variant="outline"
                            className="h-9  text-[11.5px] dark border-white/10 bg-brand-background text-brand-light/90 rounded-[6px]"
                            onClick={() => void pickSaveLocation()}
                          >
                            Browse
                          </Button>
                        </div>
                      </div>

                      <div className="flex flex-col gap-2.5">
                        <label className={fieldLabelClass}>
                          <span>Code Location</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            This folder is where downloaded server bundle archives (.tar.zst)
                            will be saved. This is separate from your Save Location.
                          </span>
                        </label>
                        <div className="flex items-center gap-2">
                          <Input
                            value={codeLocation}
                            onChange={(e) => setCodeLocation(e.target.value)}
                            placeholder="/path/to/code-location"
                            className="h-9 text-[12px]!  bg-brand-background/80 border-white/10 text-brand-light rounded-[6px]"
                          />
                          <Button
                            type="button"
                            variant="outline"
                            className="h-9  text-[11.5px] dark border-white/10 bg-brand-background text-brand-light/90 rounded-[6px]"
                            onClick={() => void pickCodeLocation()}
                          >
                            Browse
                          </Button>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="render" className="m-0 ">
                    <div className="flex flex-col gap-4 text-brand-light">
                      <div className="flex flex-col gap-1 border-b border-brand-light/8 pb-4">
                        <div className="text-[13px] font-semibold text-brand-light">
                          Render Generation Steps
                        </div>
                        <div className="flex flex-col gap-2 text-[10.5px] text-brand-light/60 font-normal">
                          Note that when we render generation steps, the generation speed will be much slower, and this will consume more system resources.
                        </div>
                      </div>
                      

                      <div className="flex items-start justify-between gap-3">
                        <label
                          htmlFor="installer-render-image"
                          className={fieldLabelClass + " flex-1 cursor-pointer"}
                        >
                          <span>Render for Image</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            Enable image rendering outputs/steps where applicable.
                          </span>
                        </label>
                        <Checkbox
                          id="installer-render-image"
                          checked={renderImage}
                          onCheckedChange={(checked) =>
                            setRenderImage(Boolean(checked))
                          }
                          className="mt-1"
                        />
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <label
                          htmlFor="installer-render-video"
                          className={fieldLabelClass + " flex-1 cursor-pointer"}
                        >
                          <span>Render for Video</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            Enable video rendering outputs/steps where applicable.
                          </span>
                        </label>
                        <Checkbox
                          id="installer-render-video"
                          checked={renderVideo}
                          onCheckedChange={(checked) =>
                            setRenderVideo(Boolean(checked))
                          }
                          className="mt-1"
                        />
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="mask" className="m-0">
                    <div className="flex flex-col gap-4 text-brand-light">
                    <div className="flex flex-col gap-1 border-b border-brand-light/8 pb-4">
                        <div className="text-[13px] font-semibold text-brand-light">
                          Models
                        </div>
                        <div className="flex flex-col gap-2 text-[10.5px] text-brand-light/60 font-normal">
                          Models that will be downloaded to power advanced image and video generation features.
                        </div>
                      </div>
                      <div className="flex flex-col gap-2.5">
                        <label className={fieldLabelClass}>
                          <span>Mask Model</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            The backend model to download and use for mask generation.
                          </span>
                        </label>
                        <Select value={maskModel} onValueChange={setMaskModel}>
                          <SelectTrigger
                            size="sm"
                            className="w-full h-8.5! text-[11.5px] bg-brand-background/70 font-medium rounded-[6px] border-white/10 text-brand-light"
                          >
                            <SelectValue placeholder="Select mask model" />
                          </SelectTrigger>
                          <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                            {MASK_MODEL_OPTIONS.map((opt) => (
                              <SelectItem
                                key={opt.value}
                                value={opt.value}
                                className="text-[11px] font-medium"
                              >
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <label
                          htmlFor="installer-install-rife"
                          className={fieldLabelClass + " flex-1 cursor-pointer"}
                        >
                          <span>Install Rife Frame Interpolation</span>
                          <span className="text-[10px] text-brand-light/60 font-normal">
                            By selecting, we will install the rife model onto your machine as
                            well.
                          </span>
                        </label>
                        <Checkbox
                          id="installer-install-rife"
                          checked={installRifeFrameInterpolation}
                          onCheckedChange={(checked) =>
                            setInstallRifeFrameInterpolation(Boolean(checked))
                          }
                          className="mt-1"
                        />
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="install" className="m-0">
                    <div className="flex flex-col gap-4 text-brand-light">
                      <div>
                        <div className="text-[13px] font-semibold text-brand-light">
                          Ready to Install
                        </div>
                        <div className="text-[11px] text-brand-light/60 mt-1">
                          This will install the selected models and configurations to your machine.
                        </div>
                      </div>

                      <div className="rounded-md border border-brand-light/5 bg-brand-background/70 backdrop-blur-md p-4 text-[11.5px]">
                        <div className="flex items-center justify-between gap-4 mb-2">
                          <span className="text-brand-light font-medium ">
                            Server Bundle
                          </span>
                          <span className="text-brand-light/90 truncate max-w-[70%]">
                            {localBundlePath
                              ? localBundlePath
                              : selectedServerBundle
                                ? selectedServerBundle.tag
                                : "—"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-4">
                          <span className="text-brand-light font-medium ">Save Location</span>
                          <span className="text-brand-light/90 truncate max-w-[70%]">
                            {saveLocation || "—"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-4 mt-2">
                          <span className="text-brand-light font-medium ">Code Location</span>
                          <span className="text-brand-light/90 truncate max-w-[70%]">
                            {codeLocation || "—"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-4 mt-2">
                          <span className="text-brand-light font-medium ">Render</span>
                          <span className="text-brand-light/90">
                            {renderImage ? "Image" : ""}
                            {renderImage && renderVideo ? " + " : ""}
                            {renderVideo ? "Video" : ""}
                            {!renderImage && !renderVideo ? "—" : ""}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-4 mt-2">
                           <span className="text-brand-light font-medium ">Mask Model</span>
                          <span className="text-brand-light/90">{maskModel}</span>
                        </div>
                        <div className="flex items-center justify-between gap-4 mt-2">
                          <span className="text-brand-light font-medium">Rife</span>
                          <span className="text-brand-light/90">
                            {installRifeFrameInterpolation ? "Install" : "Skip"}
                          </span>
                        </div>
                      </div>

                      {installStatus ? (
                        <div className="text-[11px] text-brand-light/70">
                          {installStatus}
                        </div>
                      ) : null}
                      {installError ? (
                        <div className="text-[11px] text-red-300/90">
                          {installError}
                        </div>
                      ) : null}

                    </div>
                  </TabsContent>
                </div>

                <div className="px-5 pb-5 pt-0 flex items-center justify-between gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="rounded-[6px]  bg-brand-background-light border border-brand-light/5 hover:bg-brand/70 hover:text-brand-light/90 px-6 py-2 dark text-brand-light/90 text-[11.5px]"
                    disabled={installing}
                    onClick={canGoBack ? goBack : requestCloseInstaller}
                  >
                    {canGoBack ? "Back" : hasBackend ? "Cancel" : "Back"}
                  </Button>

                  <div className="text-[11px] text-brand-light/50">
                    Step {activeIndex + 1} of {steps.length}
                  </div>

                  <Button
                    type="button"
                    className={`rounded-[6px] px-6 py-2 text-[11.5px] border border-brand-light/5 ${activeStep === "install" ? "bg-brand-accent-shade hover:bg-brand-accent-shade/80 border-brand-light/10 text-brand-light" : "bg-brand-background-light hover:bg-brand/70 hover:text-brand-light/90 text-brand-light"}`}
                    disabled={
                      installing || (!canGoNext && activeStep !== "install")
                    }
                    onClick={
                      activeStep === "install"
                        ? () => void runInstall()
                        : goNext
                    }
                  >
                    {canGoNext ? "Next" : installing ? "Installing…" : "Install"}
                  </Button>
                </div>
                </Tabs>
              ) : (
                <div className="w-full flex h-[70vh] min-h-[420px] max-h-[620px] overflow-hidden min-w-0">
                  {/* Left: vertical phase tabs */}
                  <div className="w-[240px] border-r border-white/10 p-4 shrink-0">
                    <div className="text-[11px] uppercase tracking-wide text-brand-light/60 mb-3">
                      Installation Phases
                    </div>
                    <div className="flex flex-col gap-2">
                      {phaseList.map((p) => {
                        const st = phaseState[p.id];
                        const isActive = activePhase === p.id;
                        const isDisabled = st.status === "pending";
                        return (
                          <button
                            key={p.id}
                            type="button"
                            disabled={isDisabled}
                            onClick={() => setActivePhase(p.id)}
                            className={[
                              "w-full text-left rounded-[8px] border px-3 py-2 flex items-center gap-2 transition-colors",
                              isActive
                                ? "bg-brand border-brand-light/15 hover:bg-brand!"
                                : "bg-brand-background border-white/10",
                              isDisabled
                                ? "opacity-50 cursor-not-allowed"
                                : "hover:bg-brand/70",
                            ].join(" ")}
                          >
                            <span className={isActive ? "text-brand-light" : "text-brand-light/70"}>
                              {renderPhaseIcon(st)}
                            </span>
                            <div className="flex flex-col min-w-0">
                              <span className="text-[11px] font-medium text-brand-light/90">
                                {p.label}
                              </span>
                              <span className="text-[10px] text-brand-light/50 truncate">
                                {st.status === "completed"
                                  ? "Completed"
                                  : st.status === "skipped"
                                    ? "Skipped"
                                    : st.status === "error"
                                      ? "Error"
                                      : st.status === "active"
                                        ? "In progress"
                                        : "Pending"}
                              </span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Right: phase details */}
                  <div className="flex-1 p-5 min-w-0 flex flex-col min-h-0">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex flex-col">
                        <div className="text-[13px] font-semibold text-brand-light">
                          {phaseList.find((p) => p.id === activePhase)?.label}
                        </div>
                        <div className="text-[11px] text-brand-light/60 mt-1">
                          {phaseState[activePhase]?.message ||
                            (phaseState[activePhase]?.status === "pending"
                              ? "Waiting…"
                              : "Working…")}
                        </div>
                      </div>
                      {installJobId ? (
                        <div className="text-[10px] text-brand-light/40 font-mono truncate max-w-[220px]">
                          Job: {installJobId}
                        </div>
                      ) : null}
                    </div>

                    {phaseState[activePhase]?.percent !== null ? (
                      <div className="mt-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[10.5px] text-brand-light/60">
                            Progress
                          </span>
                          <span className="text-[10.5px] text-brand-light/60">
                            {phaseState[activePhase]!.percent}%
                          </span>
                        </div>
                        <ProgressBar
                          percent={phaseState[activePhase]!.percent!}
                          className="h-2 border-brand-light/15 bg-brand-background-dark/70"
                          barClassName="bg-brand-accent"
                        />
                      </div>
                    ) : null}

                    {/* Phase-specific content */}
                    <div className="mt-5 flex-1 min-h-0">
                      {activePhase === "extract_bundle" ? (
                        <div className="rounded-md border border-brand-light/5 bg-brand-background/60 backdrop-blur-md p-4 text-[11px] text-brand-light/70">
                          Extracting bundle…
                        </div>
                      ) : activePhase === "download_models" ? (
                        <div className="rounded-md border border-brand-light/5 bg-brand-background/60 backdrop-blur-md p-4 text-[11px] text-brand-light/70">
                          {phaseState.download_models.message || "Running setup…"}
                        </div>
                      ) : activePhase === "verify_attention" ? (
                        <div className="rounded-md border border-brand-light/5 bg-brand-background/60 backdrop-blur-md p-3 h-full flex flex-col min-h-0">
                          <div className="text-[11px] text-brand-light/70 mb-2">
                            Attention backend checks
                          </div>
                          <div
                            key={attentionListVersion}
                            className="flex-1 min-h-0 rounded border border-brand-light/10 bg-black/30 overflow-auto p-2"
                          >
                            {attentionEventsRef.current.length === 0 ? (
                              <div className="text-[10.5px] text-brand-light/50">
                                Waiting for verification to start…
                              </div>
                            ) : (
                              <ScrollAreaPrimitive.Root className="flex flex-col gap-1">
                                {attentionEventsRef.current.map((ln, idx) => (
                                  <div key={`${idx}-${ln}`} className="text-[10.5px] text-brand-light/60">
                                    {ln}
                                  </div>
                                ))}
                                 <ScrollBar orientation="vertical" />
                              </ScrollAreaPrimitive.Root>
                            )}
                          </div>
                        </div>
                      ) : activePhase === "update_configs" ? (
                        <div className="rounded-md border border-brand-light/5 bg-brand-background/60 backdrop-blur-md p-4 text-[11px] text-brand-light/70">
                          Config setup is complete.
                        </div>
                      ) : (
                        <div className="rounded-md border border-brand-light/5 bg-brand-background/60 backdrop-blur-md p-4 text-[11px] text-brand-light/70">
                          Bundle download in progress.
                        </div>
                      )}
                    </div>

                    {installError ? (
                      <div className="mt-4 text-[11px] text-red-300/90">
                        {installError}
                      </div>
                    ) : null}
                    {uiMode === "done" && installStatus ? (
                      <div className="mt-4 text-[11px] font-medium text-brand-light/80">
                        {installStatus}
                      </div>
                    ) : null}

                    {uiMode === "done" ? (
                      <div className="mt-5 flex items-center justify-between gap-3">
                        <div className="text-[11px] text-brand-light font-semibold">
                          {backendStatus === "starting"
                            ? "Starting Backend…"
                            : backendStatus === "started"
                              ? "Backend Started"
                              : backendStatus === "error"
                                ? `Backend failed to start${backendError ? `: ${backendError}` : ""}`
                                : null}
                        </div>
                        <div className="flex items-center gap-2">
                          {backendStatus === "started" ? (
                            <Button
                              type="button"
                              className="rounded-[6px] px-6 py-2 text-[11.5px] bg-brand-accent-shade hover:bg-brand-accent-shade/80 border border-brand-light/10 text-brand-light"
                              onClick={async () => {
                                const res = await launchMainWindow();
                                setShowInstaller(false);
                                if (!res.ok) {
                                  setInstallError(res.error || "Failed to launch main window");
                                } else {
                                }
                              }}
                            >
                              Launch
                            </Button>
                          ) : (
                            null
                          )}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

export default Installer;



import React, { useEffect, useState } from "react";
import { LuFolder, LuSettings } from "react-icons/lu";
import { LuEye, LuEyeOff } from "react-icons/lu";
import { LuCheck, LuLoader } from "react-icons/lu";
import { LuInfo } from "react-icons/lu";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { pickMediaPaths } from "@app/preload";
import {
  getBackendIsRemote,
  getBackendPathSizes,
  getMemorySettings,
  previewBackendUrl,
  verifyBackendUrlAndFetchSettings,
  setMemorySettings,
} from "@app/preload";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useControlsStore } from "@/lib/control";
import { useSettingsStore } from "@/lib/settings-store";
import { getFolderSize } from "@app/preload";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import NumberInputSlider from "@/components/properties/model/inputs/NumberInputSlider";

import {
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
  } from "@/components/ui/tabs";

interface SettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const fieldLabelClass =
  "text-brand-light text-[10.5px] font-medium mb-1 flex flex-col gap-1";

const normalizeUrl = (value: string | null | undefined): string => {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) return "";
  try {
    const u = new URL(trimmed);
    return u.toString().replace(/\/$/, "");
  } catch {
    return trimmed.replace(/\/$/, "");
  }
};

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  onOpenChange,
}) => {
  const {
    initialized,
    initializing,
    hydrate,
    cachePath: cachePathGlobal,
    componentsPath: componentsPathGlobal,
    configPath: configPathGlobal,
    loraPath: loraPathGlobal,
    preprocessorPath: preprocessorPathGlobal,
    postprocessorPath: postprocessorPathGlobal,
    hfToken: hfTokenGlobal,
    civitaiApiKey: civitaiApiKeyGlobal,
    backendUrl: backendUrlGlobal,
    renderImageSteps: renderImageStepsGlobal,
    renderVideoSteps: renderVideoStepsGlobal,
    useFastDownload: useFastDownloadGlobal,
    autoUpdateEnabled: autoUpdateEnabledGlobal,
    maskModel: maskModelGlobal,
    setMaskModel: setMaskModelGlobal,
    setCachePath: setCachePathGlobal,
    setComponentsPath: setComponentsPathGlobal,
    setConfigPath: setConfigPathGlobal,
    setLoraPath: setLoraPathGlobal,
    setPreprocessorPath: setPreprocessorPathGlobal,
    setPostprocessorPath: setPostprocessorPathGlobal,
    setHfToken: setHfTokenGlobal,
    setCivitaiApiKey: setCivitaiApiKeyGlobal,
    setBackendUrl: setBackendUrlGlobal,
    setRenderImageSteps: setRenderImageStepsGlobal,
    setRenderVideoSteps: setRenderVideoStepsGlobal,
    setUseFastDownload: setUseFastDownloadGlobal,
    setAutoUpdateEnabled: setAutoUpdateEnabledGlobal,
  } = useSettingsStore();

  const controlsFps = useControlsStore((s) => s.fps);
  const setGlobalFpsWithRescale = useControlsStore((s) => s.setFpsWithRescale);
  const setGlobalDefaultClipLength = useControlsStore(
    (s) => s.setDefaultClipLength,
  );

  const [cachePath, setCachePath] = useState<string>("");
  const [componentsPath, setComponentsPath] = useState<string>("");
  const [configPath, setConfigPath] = useState<string>("");
  const [loraPath, setLoraPath] = useState<string>("");
  const [preprocessorPath, setPreprocessorPath] = useState<string>("");
  const [postprocessorPath, setPostprocessorPath] = useState<string>("");
  const [hfToken, setHfToken] = useState<string>("");
  const [hfTokenVisible, setHfTokenVisible] = useState<boolean>(false);
  const [civitaiApiKey, setCivitaiApiKey] = useState<string>("");
  const [civitaiApiKeyVisible, setCivitaiApiKeyVisible] =
    useState<boolean>(false);
  const [projectFps, setProjectFps] = useState<string>(
    String(controlsFps || 24),
  );
  const [defaultClipSeconds, setDefaultClipSeconds] = useState<string>("5");
  const [backendUrl, setBackendUrlLocal] = useState<string>("");
  const [backendUrlVerifiedFor, setBackendUrlVerifiedFor] = useState<
    string | null
  >(null);
  const [backendUrlVerifyError, setBackendUrlVerifyError] = useState<
    string | null
  >(null);
  const [isVerifyingBackendUrl, setIsVerifyingBackendUrl] =
    useState<boolean>(false);
  const [cacheSizeLabel, setCacheSizeLabel] = useState<string | null>(null);
  const [componentsSizeLabel, setComponentsSizeLabel] = useState<string | null>(
    null,
  );
  const [configSizeLabel, setConfigSizeLabel] = useState<string | null>(null);
  const [loraSizeLabel, setLoraSizeLabel] = useState<string | null>(null);
  const [preSizeLabel, setPreSizeLabel] = useState<string | null>(null);
  const [postSizeLabel, setPostSizeLabel] = useState<string | null>(null);
  const [maskModel, setMaskModel] = useState<string>("sam2_base_plus");
  const [renderImageSteps, setRenderImageSteps] = useState<boolean>(false);
  const [renderVideoSteps, setRenderVideoSteps] = useState<boolean>(false);
  const [useFastDownload, setUseFastDownload] = useState<boolean>(true);
  const [autoUpdateEnabled, setAutoUpdateEnabled] = useState<boolean>(true);
  const [isBackendRemote, setIsBackendRemote] = useState<boolean>(false);

  const [memoryKnobs, setMemoryKnobs] = useState<Record<string, any> | null>(null);
  const [isLoadingMemoryKnobs, setIsLoadingMemoryKnobs] = useState<boolean>(false);
  const [isSavingMemoryKnobs, setIsSavingMemoryKnobs] = useState<boolean>(false);
  const memorySaveDebounceRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const pendingMemoryUpdatesRef = React.useRef<Record<string, any>>({});

  const normalizedBackendUrl = normalizeUrl(backendUrl);
  const normalizedBackendUrlGlobal = normalizeUrl(backendUrlGlobal);
  const backendUrlChanged = normalizedBackendUrl !== normalizedBackendUrlGlobal;
  const backendUrlIsVerified = backendUrlVerifiedFor === normalizedBackendUrl;
  const backendUrlRequiresVerify = backendUrlChanged && !!normalizedBackendUrl;
  const saveDisabled =
    isVerifyingBackendUrl || (backendUrlRequiresVerify && !backendUrlIsVerified);

  const inferRemoteFromUrl = (url: string): boolean => {
    try {
      const u = new URL(url);
      const host = (u.hostname || "").toLowerCase();
      return !(host === "localhost" || host === "127.0.0.1" || host === "::1");
    } catch {
      return false;
    }
  };

  const effectiveIsRemote =
    normalizedBackendUrl && backendUrlChanged
      ? inferRemoteFromUrl(normalizedBackendUrl)
      : isBackendRemote;

  const refreshBackendPathSizes = async (url?: string | null) => {
    try {
      const res = await getBackendPathSizes(url ?? null);
      if (!res?.success || !res.data) return;
      setCacheSizeLabel(formatBytes(res.data.cachePathBytes));
      setComponentsSizeLabel(formatBytes(res.data.componentsPathBytes));
      setConfigSizeLabel(formatBytes(res.data.configPathBytes));
      setLoraSizeLabel(formatBytes(res.data.loraPathBytes));
      setPreSizeLabel(formatBytes(res.data.preprocessorPathBytes));
      setPostSizeLabel(formatBytes(res.data.postprocessorPathBytes));
    } catch {
      // ignore
    }
  };

  const resetToGlobalSettings = () => {
    setCachePath(cachePathGlobal ?? "");
    setComponentsPath(componentsPathGlobal ?? "");
    setConfigPath(configPathGlobal ?? "");
    setLoraPath(loraPathGlobal ?? "");
    setPreprocessorPath(preprocessorPathGlobal ?? "");
    setPostprocessorPath(postprocessorPathGlobal ?? "");
    setHfToken(hfTokenGlobal ?? "");
    setCivitaiApiKey(civitaiApiKeyGlobal ?? "");
    setBackendUrlLocal(backendUrlGlobal ?? "");
    setBackendUrlVerifiedFor(normalizeUrl(backendUrlGlobal));
    setBackendUrlVerifyError(null);
    setMaskModel(maskModelGlobal ?? "sam2_base_plus");
    setRenderImageSteps(Boolean(renderImageStepsGlobal));
    setRenderVideoSteps(Boolean(renderVideoStepsGlobal));
    setUseFastDownload(Boolean(useFastDownloadGlobal));
    setAutoUpdateEnabled(Boolean(autoUpdateEnabledGlobal));

    // Refresh size labels for remote backends; local sizes are handled via getFolderSize effect.
    if (isBackendRemote) {
      void refreshBackendPathSizes(null);
    }

    // Also revert the main-process "active" backend URL (preview) so ApexApi/AppDirProtocol return
    // to the same remote/local mode as the saved settings.
    const saved = normalizeUrl(backendUrlGlobal);
    if (saved) {
      void previewBackendUrl(saved);
    }
    // Refresh remote/local flag so the UI (folder buttons + size strategy) matches the active backend.
    void (async () => {
      try {
        const res = await getBackendIsRemote();
        const isRemote = !!(
          res &&
          res.success &&
          res.data &&
          typeof res.data.isRemote === "boolean" &&
          res.data.isRemote
        );
        setIsBackendRemote(isRemote);
      } catch {
        setIsBackendRemote(false);
      }
    })();
  };

  const handleVerifyBackendUrl = async () => {
    const candidate = normalizeUrl(backendUrl);
    if (!candidate) {
      toast.error("Backend URL is required. Reverting to current settings.");
      resetToGlobalSettings();
      return;
    }

    setBackendUrlVerifyError(null);
    setIsVerifyingBackendUrl(true);
    try {
      const res = await verifyBackendUrlAndFetchSettings(candidate);
      if (!res?.success || !res.data) {
        toast.error(
          res?.error
            ? `${res.error} Reverting to current settings.`
            : "Failed to verify backend URL. Reverting to current settings.",
        );
        resetToGlobalSettings();
        return;
      }

      // Normalize the input and mark it verified
      setBackendUrlLocal(candidate);
      setBackendUrlVerifiedFor(candidate);

      // Sync local form fields from backend in realtime (API Config + Save Paths)
      setCachePath(res.data.cachePath ?? "");
      setComponentsPath(res.data.componentsPath ?? "");
      setConfigPath(res.data.configPath ?? "");
      setLoraPath(res.data.loraPath ?? "");
      setPreprocessorPath(res.data.preprocessorPath ?? "");
      setPostprocessorPath(res.data.postprocessorPath ?? "");
      setMaskModel(res.data.maskModel ?? "sam2_base_plus");
      setRenderImageSteps(Boolean(res.data.renderImageSteps));
      setRenderVideoSteps(Boolean(res.data.renderVideoSteps));
      setUseFastDownload(Boolean(res.data.useFastDownload));
      setAutoUpdateEnabled(Boolean(res.data.autoUpdateEnabled));

      // Switch app runtime (ApexApi/AppDirProtocol) to this backend immediately.
      // This updates remote/local mode in main based on /config/hostname probing.
      const previewRes = await previewBackendUrl(candidate);
      if (!previewRes?.success) {
        toast.error(
          previewRes?.error
            ? `${previewRes.error} Reverting to current settings.`
            : "Failed to switch backend URL. Reverting to current settings.",
        );
        resetToGlobalSettings();
        return;
      }

      // Keep size labels fresh after syncing paths (best-effort; uses current backend URL)
      void refreshBackendPathSizes(null);
    } catch (e) {
      toast.error(
        e instanceof Error
          ? `${e.message} Reverting to current settings.`
          : "Failed to verify backend URL. Reverting to current settings.",
      );
      resetToGlobalSettings();
    } finally {
      setIsVerifyingBackendUrl(false);
    }
  };


  useEffect(() => {
    if (!initialized && !initializing) {
      void hydrate();
    }
  }, [initialized, initializing, hydrate]);

  useEffect(() => {
    if (!open) return;
    // Sync local form state from global settings when dialog opens
    setCachePath(cachePathGlobal ?? "");
    setComponentsPath(componentsPathGlobal ?? "");
    setConfigPath(configPathGlobal ?? "");
    setLoraPath(loraPathGlobal ?? "");
    setPreprocessorPath(preprocessorPathGlobal ?? "");
    setPostprocessorPath(postprocessorPathGlobal ?? "");
    setHfToken(hfTokenGlobal ?? "");
    setCivitaiApiKey(civitaiApiKeyGlobal ?? "");
    setBackendUrlLocal(backendUrlGlobal ?? "");
    setBackendUrlVerifiedFor(normalizeUrl(backendUrlGlobal));
    setBackendUrlVerifyError(null);
    setMaskModel(maskModelGlobal ?? "sam2_base_plus");
    setRenderImageSteps(Boolean(renderImageStepsGlobal));
    setRenderVideoSteps(Boolean(renderVideoStepsGlobal));
    setUseFastDownload(Boolean(useFastDownloadGlobal));
    setAutoUpdateEnabled(Boolean(autoUpdateEnabledGlobal));
  }, [
    open,
    cachePathGlobal,
    componentsPathGlobal,
    configPathGlobal,
    loraPathGlobal,
    preprocessorPathGlobal,
    postprocessorPathGlobal,
    hfTokenGlobal,
    civitaiApiKeyGlobal,
    backendUrlGlobal,
    maskModelGlobal,
    renderImageStepsGlobal,
    renderVideoStepsGlobal,
    useFastDownloadGlobal,
    autoUpdateEnabledGlobal,
  ]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await getBackendIsRemote();
        const isRemote = !!(
          res &&
          res.success &&
          res.data &&
          typeof res.data.isRemote === "boolean" &&
          res.data.isRemote
        );
        if (!cancelled) setIsBackendRemote(isRemote);
        if (!cancelled && isRemote) {
          void refreshBackendPathSizes(null);
        }
      } catch {
        if (!cancelled) setIsBackendRemote(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, backendUrlGlobal]);

  useEffect(() => {
    // If the user edits the URL after verifying, require re-verify.
    const candidate = normalizeUrl(backendUrl);
    if (!candidate) {
      setBackendUrlVerifiedFor(null);
      return;
    }
    if (backendUrlVerifiedFor && backendUrlVerifiedFor !== candidate) {
      setBackendUrlVerifiedFor(null);
    }
  }, [backendUrl, backendUrlVerifiedFor]);

  const formatBytes = (bytes: number | null | undefined): string | null => {
    if (bytes == null || !Number.isFinite(bytes)) return null;
    const value = Math.max(0, bytes);
    if (value < 1024) return `${value} B`;
    const kb = value / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    const gb = mb / 1024;
    return `${gb.toFixed(2)} GB`;
  };

  const boolFromEnv = (v: any): boolean => {
    const s = String(v ?? "").trim().toLowerCase();
    if (!s) return false;
    return !(s === "0" || s === "false" || s === "no" || s === "off");
  };

  const getFloat = (key: string, fallback: number): number => {
    const v = memoryKnobs?.[key];
    const n = typeof v === "number" ? v : Number(String(v ?? ""));
    return Number.isFinite(n) ? n : fallback;
  };

  const getInt = (key: string, fallback: number): number => {
    const v = memoryKnobs?.[key];
    const n = typeof v === "number" ? v : Number(String(v ?? ""));
    if (!Number.isFinite(n)) return fallback;
    return Math.max(0, Math.trunc(n));
  };

  const refreshMemoryKnobs = async () => {
    setIsLoadingMemoryKnobs(true);
    try {
      const res = await getMemorySettings();
      if (!res?.success || !res.data?.settings) {
        toast.error(res?.error || "Failed to load memory settings.");
        return;
      }
      setMemoryKnobs(res.data.settings ?? {});
    } catch {
      toast.error("Failed to load memory settings.");
    } finally {
      setIsLoadingMemoryKnobs(false);
    }
  };

  const flushMemoryUpdates = async () => {
    if (memorySaveDebounceRef.current) {
      clearTimeout(memorySaveDebounceRef.current);
      memorySaveDebounceRef.current = null;
    }
    const payload = pendingMemoryUpdatesRef.current;
    pendingMemoryUpdatesRef.current = {};
    if (!Object.keys(payload).length) return;
    setIsSavingMemoryKnobs(true);
    try {
      const res = await setMemorySettings(payload);
      if (!res?.success || !res.data?.settings) {
        toast.error(res?.error || "Failed to update memory settings.");
        return;
      }
      setMemoryKnobs(res.data.settings ?? {});
    } catch {
      toast.error("Failed to update memory settings.");
    } finally {
      setIsSavingMemoryKnobs(false);
    }
  };

  const queueMemoryUpdate = (partial: Record<string, any>) => {
    setMemoryKnobs((prev) => ({ ...(prev ?? {}), ...partial }));
    pendingMemoryUpdatesRef.current = {
      ...pendingMemoryUpdatesRef.current,
      ...partial,
    };
    if (memorySaveDebounceRef.current) {
      clearTimeout(memorySaveDebounceRef.current);
    }
    memorySaveDebounceRef.current = setTimeout(() => {
      void flushMemoryUpdates();
    }, 350);
  };

  useEffect(() => {
    if (!open) return;
    void refreshMemoryKnobs();
    return () => {
      // Ensure the most recent knob changes are sent before the modal closes.
      void flushMemoryUpdates();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, backendUrlGlobal]);

  const InfoTip: React.FC<{ text: string }> = ({ text }) => {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="text-brand-light/70 hover:text-brand-light focus:outline-none"
          >
            <LuInfo className="w-3 h-3" />
          </button>
        </TooltipTrigger>
        <TooltipContent
          sideOffset={6}
          className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
        >
          {text}
        </TooltipContent>
      </Tooltip>
    );
  };

  const useFolderSizeEffect = (
    pathValue: string,
    setter: (label: string | null) => void,
    enabled: boolean,
  ) => {
    useEffect(() => {
      if (!enabled) return;
      const trimmed = (pathValue || "").trim();
      if (!trimmed) {
        setter(null);
        return;
      }
      let cancelled = false;
      (async () => {
        try {
          const res = await getFolderSize(trimmed);
          if (!res?.success || !res.data) {
            if (!cancelled) setter(null);
            return;
          }
          const label = formatBytes(res.data.size_bytes);
          if (!cancelled) setter(label);
        } catch {
          if (!cancelled) setter(null);
        }
      })();
      return () => {
        cancelled = true;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pathValue, enabled]);
  };

  // Local-only size computation. For remote backends, we fetch sizes from the server.
  useFolderSizeEffect(cachePath, setCacheSizeLabel, !effectiveIsRemote);
  useFolderSizeEffect(componentsPath, setComponentsSizeLabel, !effectiveIsRemote);
  useFolderSizeEffect(configPath, setConfigSizeLabel, !effectiveIsRemote);
  useFolderSizeEffect(loraPath, setLoraSizeLabel, !effectiveIsRemote);
  useFolderSizeEffect(preprocessorPath, setPreSizeLabel, !effectiveIsRemote);
  useFolderSizeEffect(postprocessorPath, setPostSizeLabel, !effectiveIsRemote);

  useEffect(() => {
    if (!open) return;
    if (!effectiveIsRemote) return;
    // Poll backend path sizes while modal is open (remote backends only).
    const id = setInterval(() => {
      void refreshBackendPathSizes(null);
    }, 10_000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, effectiveIsRemote]);

  const handlePickDirectory = async (
    title: string,
    setter: (value: string) => void,
    currentPath?: string,
  ) => {
    try {
      const picked = await pickMediaPaths({
        directory: true,
        title,
        defaultPath:
          currentPath && currentPath.trim().length > 0
            ? currentPath.trim()
            : undefined,
      });
      const dir =
        Array.isArray(picked) && picked.length > 0 ? picked[0] : null;
      if (dir && typeof dir === "string") {
        setter(dir);
      }
    } catch {
      // Swallow errors; keep existing path untouched
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>

      <DialogContent className="w-full max-w-xl dark p-0 h-[500px] gap-0 space-y-0 items-start flex flex-col">
        <DialogTitle className="text-brand-light font-poppins text-xs font-medium px-5 pt-4 h-fit">
            Settings
        </DialogTitle>
        <Tabs defaultValue="general" className="p-5 pt-0 mt-5 w-full flex-1 flex flex-col overflow-hidden">
          <TabsList className="font-poppins text-brand-light/90 dark gap-1 mb-1.5 w-full ">
          <TabsTrigger
              className="data-[state=active]:bg-brand-light/10 data-[state=active]:text-brand-light text-[11px] rounded-[6px] px-2 py-1 w-full!"
              value="project"
            >
             Project
            </TabsTrigger>
            <TabsTrigger
              className="data-[state=active]:bg-brand-light/10 data-[state=active]:text-brand-light text-[11px] rounded-[6px] px-2 py-1 w-full!"
              value="paths"
            >
             Save Paths
            </TabsTrigger>
            <TabsTrigger
              className="data-[state=active]:bg-brand-light/10 data-[state=active]:text-brand-light text-[11px] rounded-[6px] px-2 py-1 w-full!"
              value="config"
            >
            API Config
            </TabsTrigger>
            <TabsTrigger
              className="data-[state=active]:bg-brand-light/10 data-[state=active]:text-brand-light text-[11px] rounded-[6px] px-2 py-1 w-full!"
              value="memory"
            >
              Memory
            </TabsTrigger>
            <TabsTrigger
              className="data-[state=active]:bg-brand-light/10 data-[state=active]:text-brand-light text-[11px] rounded-[6px] px-2 py-1 w-full!"
              value="tokens"
            >
             API Tokens
            </TabsTrigger>
           
          </TabsList>
          <div className="flex-1 overflow-y-auto pb-8 mb-12 pr-2.5 custom-scrollbar">
          <TabsContent value="project">
            <div className="flex flex-col gap-4 font-poppins text-brand-light ">
              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Project FPS</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    The current FPS of the project.
                  </span>
                </label>
                <Select
                  value={projectFps}
                  onValueChange={(value) => {
                    setProjectFps(value);
                  }}
                >
                  <SelectTrigger
                    size="sm"
                    className="w-full h-7.5! text-[11px] bg-brand-background/70 rounded-[6px]"
                  >
                    <SelectValue placeholder="Select FPS" />
                  </SelectTrigger>
                  <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                    <SelectItem value="24" className="text-[11px] font-medium">
                      24 FPS
                    </SelectItem>
                    <SelectItem value="25" className="text-[11px] font-medium">
                      25 FPS
                    </SelectItem>
                    <SelectItem value="30" className="text-[11px] font-medium">
                      30 FPS
                    </SelectItem>
                    <SelectItem value="48" className="text-[11px] font-medium">
                      48 FPS
                    </SelectItem>
                    <SelectItem value="60" className="text-[11px] font-medium">
                      60 FPS
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass} >
                  <span>Default Clip Length</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    For image, filter, and preprocessor clips.
                  </span>
                </label>
                <Select
                  value={defaultClipSeconds}
                  onValueChange={(value) => {
                    setDefaultClipSeconds(value);
                  }}
                >
                  <SelectTrigger
                    size="sm"
                    className="w-full h-7.5! text-[11px] bg-brand-background/70 rounded-[6px]"
                  >
                    <SelectValue placeholder="Select duration" />
                  </SelectTrigger>
                  <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                    <SelectItem value="3" className="text-[11px] font-medium">
                      3 seconds
                    </SelectItem>
                    <SelectItem value="5" className="text-[11px] font-medium">
                      5 seconds
                    </SelectItem>
                    <SelectItem value="10" className="text-[11px] font-medium">
                      10 seconds
                    </SelectItem>
                    <SelectItem value="15" className="text-[11px] font-medium">
                      15 seconds
                    </SelectItem>
                    <SelectItem value="30" className="text-[11px] font-medium">
                      30 seconds
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="tokens">
            <div className="flex flex-col gap-4 font-poppins text-brand-light">
              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Hugging Face Hub Token</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    This is used for any models that may require authentication.
                  </span>
                </label>
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={hfTokenVisible ? "text" : "password"}
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      placeholder="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                      className="h-7.5 text-[11.5px]! rounded-[6px] pr-8"
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-1 flex items-center px-1 text-brand-light/70 hover:text-brand-light"
                      onClick={() => setHfTokenVisible((v) => !v)}
                    >
                      {hfTokenVisible ? (
                        <LuEyeOff className="w-3.5 h-3.5" />
                      ) : (
                        <LuEye className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>CivitAI API Key</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Used for authenticated access to CivitAI when downloading
                    models.
                  </span>
                </label>
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <Input
                      type={civitaiApiKeyVisible ? "text" : "password"}
                      value={civitaiApiKey}
                      onChange={(e) => setCivitaiApiKey(e.target.value)}
                      placeholder="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                      className="h-7.5 text-[11.5px]! rounded-[6px] pr-8"
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-1 flex items-center px-1 text-brand-light/70 hover:text-brand-light"
                      onClick={() =>
                        setCivitaiApiKeyVisible((v) => !v)
                      }
                    >
                      {civitaiApiKeyVisible ? (
                        <LuEyeOff className="w-3.5 h-3.5" />
                      ) : (
                        <LuEye className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="config">
            <div className="flex flex-col gap-4 font-poppins text-brand-light h-full">
              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Backend URL</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Where the Apex Engine HTTP API is running.
                  </span>
                </label>
                <div className="flex items-center gap-2">
                  <Input
                    value={backendUrl}
                    onChange={(e) => {
                      setBackendUrlLocal(e.target.value);
                      setBackendUrlVerifyError(null);
                    }}
                    placeholder="http://127.0.0.1:8765"
                    className="h-7.5 text-[11.5px]! rounded-[6px] flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    disabled={isVerifyingBackendUrl || !normalizedBackendUrl}
                    className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px] flex items-center gap-1.5 disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={() => void handleVerifyBackendUrl()}
                    title={
                      backendUrlIsVerified
                        ? "Verified"
                        : "Verify the backend URL"
                    }
                  >
                    {isVerifyingBackendUrl ? (
                      <LuLoader className="w-3.5 h-3.5 text-brand-light/70 animate-spin" />
                    ) : backendUrlIsVerified ? (
                      <LuCheck className="w-3.5 h-3.5 text-brand-light/70" />
                    ) : null}
                    <span>
                      {isVerifyingBackendUrl
                        ? "Verifying..."
                        : backendUrlIsVerified
                          ? "Verified"
                          : "Verify"}
                    </span>
                  </Button>
                </div>
                {backendUrlRequiresVerify && !backendUrlIsVerified && (
                  <div className="text-[10px] text-brand-light/60 font-normal">
                    Verify this URL to enable saving.
                  </div>
                )}
                {backendUrlVerifyError && (
                  <div className="text-[10px] text-red-400 font-normal">
                    {backendUrlVerifyError}
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-1">
                <label className={fieldLabelClass}>
                  <span>Mask Model</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    The model to use for mask generation.
                  </span>
                </label>
                <Select
                  value={maskModel}
                  onValueChange={(value) => {
                    setMaskModel(value);
                  }}
                >
                  <SelectTrigger
                    size="sm"
                    className="w-full h-7.5! text-[11px] bg-brand-background/70 rounded-[6px]"
                  >
                    <SelectValue placeholder="Select mask model" />
                  </SelectTrigger>
                  <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                    <SelectItem value="sam2_tiny" className="text-[11px] font-medium">
                      Sam2 Tiny  
                    </SelectItem>
                    <SelectItem value="sam2_small" className="text-[11px] font-medium">
                      Sam2 Small
                    </SelectItem>
                    <SelectItem value="sam2_base_plus" className="text-[11px] font-medium">
                      Sam2 Base Plus
                    </SelectItem>
                    <SelectItem value="sam2_large" className="text-[11px] font-medium">
                      Sam2 Large
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-start justify-between gap-3">
                <label
                  htmlFor="render-image-steps"
                  className={fieldLabelClass + " flex-1 cursor-pointer"}
                >
                  <span>Render Image Steps</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Render intermediary steps when denoising an image model.
                  </span>
                </label>
                <Checkbox
                  id="render-image-steps"
                  checked={renderImageSteps}
                  onCheckedChange={(checked) =>
                    setRenderImageSteps(Boolean(checked))
                  }
                  className="mt-1"
                />
              </div>

              <div className="flex items-start justify-between gap-3">
                <label
                  htmlFor="render-video-steps"
                  className={fieldLabelClass + " flex-1 cursor-pointer"}
                >
                  <span>Render Video Steps</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Render intermediary steps when denoising a video model.
                  </span>
                </label>
                <Checkbox
                  id="render-video-steps"
                  checked={renderVideoSteps}
                  onCheckedChange={(checked) =>
                    setRenderVideoSteps(Boolean(checked))
                  }
                  className="mt-1"
                />
              </div>

              <div className="flex items-start justify-between gap-3">
                <label
                  htmlFor="use-fast-download"
                  className={fieldLabelClass + " flex-1 cursor-pointer"}
                >
                  <span>Fast Download</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Enable faster model downloads (when supported by the
                    backend).
                  </span>
                </label>
                <Checkbox
                  id="use-fast-download"
                  checked={useFastDownload}
                  onCheckedChange={(checked) =>
                    setUseFastDownload(Boolean(checked))
                  }
                  className="mt-1"
                />
              </div>

              <div className="flex items-start justify-between gap-3">
                <label
                  htmlFor="api-auto-update"
                  className={fieldLabelClass + " flex-1 cursor-pointer"}
                >
                  <span>Automatic API Updates</span>
                  <span className="text-[10px] text-brand-light/60 font-normal">
                    Check for and apply backend updates automatically (default:
                    every 4 hours).
                  </span>
                </label>
                <Checkbox
                  id="api-auto-update"
                  checked={autoUpdateEnabled}
                  onCheckedChange={(checked) =>
                    setAutoUpdateEnabled(Boolean(checked))
                  }
                  className="mt-1"
                />
              </div>
            </div>
          </TabsContent>
          <TabsContent value="memory">
            <div className="flex flex-col gap-4 font-poppins text-brand-light h-full">
              <div className="flex items-center justify-between">
                <div className="text-[10px] text-brand-light/70">
                  These knobs apply immediately on the backend process (no restart).
                </div>
                <div className="flex items-center gap-2">
                  {(isLoadingMemoryKnobs || isSavingMemoryKnobs) && (
                    <LuLoader className="w-3.5 h-3.5 text-brand-light/70 animate-spin" />
                  )}
                  <Button
                    type="button"
                    variant="outline"
                    className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                    onClick={() => void refreshMemoryKnobs()}
                    disabled={isLoadingMemoryKnobs}
                  >
                    Refresh
                  </Button>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <div className="text-[11px] text-brand-light/80 font-medium">
                  Offload pressure
                </div>
                <NumberInputSlider
                  label="Min free VRAM fraction"
                  description="APEX_OFFLOAD_MIN_FREE_VRAM_FRACTION — General guardrail. If free VRAM drops below this, offloading becomes more aggressive."
                  value={getFloat("APEX_OFFLOAD_MIN_FREE_VRAM_FRACTION", 0.1)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_OFFLOAD_MIN_FREE_VRAM_FRACTION: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="Min free RAM fraction"
                  description="APEX_OFFLOAD_MIN_FREE_RAM_FRACTION — Avoid overcommitting system RAM while offloading."
                  value={getFloat("APEX_OFFLOAD_MIN_FREE_RAM_FRACTION", 0.08)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_OFFLOAD_MIN_FREE_RAM_FRACTION: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="VRAM pressure min free VRAM fraction"
                  description="APEX_VRAM_PRESSURE_MIN_FREE_VRAM_FRACTION — Pressure detector threshold. When below this, we may offload to keep GPU stable."
                  value={getFloat(
                    "APEX_VRAM_PRESSURE_MIN_FREE_VRAM_FRACTION",
                    0.06,
                  )}
                  onChange={(v) =>
                    queueMemoryUpdate({
                      APEX_VRAM_PRESSURE_MIN_FREE_VRAM_FRACTION: v,
                    })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="CPU safety buffer"
                  description="APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES — Minimum CPU RAM headroom to keep free (GiB shown; stored as bytes)."
                  value={
                    getInt("APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES", 2 * 1024 ** 3) /
                    1024 ** 3
                  }
                  onChange={(gib) =>
                    queueMemoryUpdate({
                      APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES: Math.round(
                        Math.max(0, gib) * 1024 ** 3,
                      ),
                    })
                  }
                  min={0}
                  max={64}
                  step={0.25}
                  toFixed={2}
                  suffix="GiB"
                />
                <div className="text-[10px] text-brand-light/60">
                  Current:{" "}
                  {formatBytes(getInt("APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES", 0)) ||
                    "—"}
                </div>
                <NumberInputSlider
                  label="CPU multiplier"
                  description="APEX_VRAM_PRESSURE_CPU_MULTIPLIER — Extra CPU RAM headroom multiplier when deciding whether to offload."
                  value={getFloat("APEX_VRAM_PRESSURE_CPU_MULTIPLIER", 1.25)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_VRAM_PRESSURE_CPU_MULTIPLIER: v })
                  }
                  min={0.5}
                  max={4}
                  step={0.05}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="Max CPU offload bytes"
                  description="APEX_VRAM_PRESSURE_MAX_CPU_OFFLOAD_BYTES — Cap how much we can spill to CPU (GiB shown; stored as bytes)."
                  value={
                    getInt(
                      "APEX_VRAM_PRESSURE_MAX_CPU_OFFLOAD_BYTES",
                      32 * 1024 ** 3,
                    ) /
                    1024 ** 3
                  }
                  onChange={(gib) =>
                    queueMemoryUpdate({
                      APEX_VRAM_PRESSURE_MAX_CPU_OFFLOAD_BYTES: Math.round(
                        Math.max(0, gib) * 1024 ** 3,
                      ),
                    })
                  }
                  min={0}
                  max={256}
                  step={1}
                  toFixed={0}
                  suffix="GiB"
                />
                <div className="text-[10px] text-brand-light/60">
                  Current:{" "}
                  {formatBytes(
                    getInt("APEX_VRAM_PRESSURE_MAX_CPU_OFFLOAD_BYTES", 0),
                  ) || "—"}
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <div className="text-[11px] text-brand-light/80 font-medium">
                  Preforward (just-in-time offload)
                </div>
                <div className="flex items-center justify-between bg-brand-light/5 border border-brand-light/10 rounded-[8px] p-3">
                  <div className="flex items-center gap-2">
                    <div className="text-[11px] font-medium">Enable preforward</div>
                    <InfoTip text="APEX_PREFWD_ENABLE — When enabled, we can offload modules just before their forward pass if VRAM is tight." />
                  </div>
                  <Checkbox
                    checked={boolFromEnv(memoryKnobs?.APEX_PREFWD_ENABLE)}
                    onCheckedChange={(checked) =>
                      queueMemoryUpdate({
                        APEX_PREFWD_ENABLE: checked ? "1" : "",
                      })
                    }
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <label className="text-brand-light text-[10.5px] font-medium">
                      Preforward flush target
                    </label>
                    <InfoTip text='APEX_PREFWD_FLUSH_TARGET — Where to offload to when preforward decides to flush. "cpu" is faster; "disk" is safest.' />
                  </div>
                  <Select
                    value={String(memoryKnobs?.APEX_PREFWD_FLUSH_TARGET ?? "cpu")}
                    onValueChange={(value) =>
                      queueMemoryUpdate({ APEX_PREFWD_FLUSH_TARGET: value })
                    }
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full h-7.5! text-[11px] bg-brand-background/70 rounded-[6px]"
                    >
                      <SelectValue placeholder="Select target" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                      <SelectItem value="cpu" className="text-[11px] font-medium">
                        CPU
                      </SelectItem>
                      <SelectItem value="disk" className="text-[11px] font-medium">
                        Disk
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <NumberInputSlider
                  label="Preforward min free bytes"
                  description="APEX_PREFWD_MIN_FREE_BYTES — If free VRAM is below this, preforward may offload (GiB shown; stored as bytes)."
                  value={getInt("APEX_PREFWD_MIN_FREE_BYTES", 0) / 1024 ** 3}
                  onChange={(gib) =>
                    queueMemoryUpdate({
                      APEX_PREFWD_MIN_FREE_BYTES: Math.round(
                        Math.max(0, gib) * 1024 ** 3,
                      ),
                    })
                  }
                  min={0}
                  max={32}
                  step={0.25}
                  toFixed={2}
                  suffix="GiB"
                />
                <div className="text-[10px] text-brand-light/60">
                  Current:{" "}
                  {formatBytes(getInt("APEX_PREFWD_MIN_FREE_BYTES", 0)) || "—"}
                </div>
                <NumberInputSlider
                  label="Preforward min free fraction"
                  description="APEX_PREFWD_MIN_FREE_FRAC — If free VRAM ≥ this (and ≥ min bytes), skip offloading."
                  value={getFloat("APEX_PREFWD_MIN_FREE_FRAC", 0.5)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_PREFWD_MIN_FREE_FRAC: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
              </div>

              <div className="flex flex-col gap-3">
                <div className="text-[11px] text-brand-light/80 font-medium">
                  Decode guards (VAE/video/audio)
                </div>
                <div className="flex items-center justify-between bg-brand-light/5 border border-brand-light/10 rounded-[8px] p-3">
                  <div className="flex items-center gap-2">
                    <div className="text-[11px] font-medium">
                      Force flush during decode
                    </div>
                    <InfoTip text="APEX_VAE_DECODE_FORCE_FLUSH — When enabled, we flush/offload to avoid OOM during decode-heavy stages." />
                  </div>
                  <Checkbox
                    checked={boolFromEnv(memoryKnobs?.APEX_VAE_DECODE_FORCE_FLUSH)}
                    onCheckedChange={(checked) =>
                      queueMemoryUpdate({
                        APEX_VAE_DECODE_FORCE_FLUSH: checked ? "1" : "",
                      })
                    }
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <label className="text-brand-light text-[10.5px] font-medium">
                      Decode flush target
                    </label>
                    <InfoTip text='APEX_VAE_DECODE_FLUSH_TARGET — Where to offload during decode flush. "cpu" is faster; "disk" is safest.' />
                  </div>
                  <Select
                    value={String(memoryKnobs?.APEX_VAE_DECODE_FLUSH_TARGET ?? "cpu")}
                    onValueChange={(value) =>
                      queueMemoryUpdate({ APEX_VAE_DECODE_FLUSH_TARGET: value })
                    }
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-full h-7.5! text-[11px] bg-brand-background/70 rounded-[6px]"
                    >
                      <SelectValue placeholder="Select target" />
                    </SelectTrigger>
                    <SelectContent className="bg-brand-background text-brand-light font-poppins z-101 dark">
                      <SelectItem value="cpu" className="text-[11px] font-medium">
                        CPU
                      </SelectItem>
                      <SelectItem value="disk" className="text-[11px] font-medium">
                        Disk
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <NumberInputSlider
                  label="Decode min free bytes"
                  description="APEX_VAE_DECODE_MIN_FREE_BYTES — Absolute VRAM floor before we start flushing (GiB shown; stored as bytes)."
                  value={getInt("APEX_VAE_DECODE_MIN_FREE_BYTES", 0) / 1024 ** 3}
                  onChange={(gib) =>
                    queueMemoryUpdate({
                      APEX_VAE_DECODE_MIN_FREE_BYTES: Math.round(
                        Math.max(0, gib) * 1024 ** 3,
                      ),
                    })
                  }
                  min={0}
                  max={32}
                  step={0.25}
                  toFixed={2}
                  suffix="GiB"
                />
                <div className="text-[10px] text-brand-light/60">
                  Current:{" "}
                  {formatBytes(getInt("APEX_VAE_DECODE_MIN_FREE_BYTES", 0)) || "—"}
                </div>
                <NumberInputSlider
                  label="Decode min free fraction"
                  description="APEX_VAE_DECODE_MIN_FREE_FRAC — Fractional VRAM floor to trigger decode flushing."
                  value={getFloat("APEX_VAE_DECODE_MIN_FREE_FRAC", 0.12)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_VAE_DECODE_MIN_FREE_FRAC: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="Decode safety multiplier"
                  description="APEX_VAE_DECODE_SAFETY_MULT — Extra safety factor when estimating decode memory usage."
                  value={getFloat("APEX_VAE_DECODE_SAFETY_MULT", 8)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_VAE_DECODE_SAFETY_MULT: v })
                  }
                  min={1}
                  max={16}
                  step={0.25}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="Decode target free VRAM fraction"
                  description="APEX_VAE_DECODE_TARGET_FREE_FRACTION — Target headroom to maintain during decode (higher = more conservative)."
                  value={getFloat("APEX_VAE_DECODE_TARGET_FREE_FRACTION", 0.3)}
                  onChange={(v) =>
                    queueMemoryUpdate({
                      APEX_VAE_DECODE_TARGET_FREE_FRACTION: v,
                    })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
              </div>

              <div className="flex flex-col gap-3">
                <div className="text-[11px] text-brand-light/80 font-medium">
                  Weight manager
                </div>
                <div className="flex items-center justify-between bg-brand-light/5 border border-brand-light/10 rounded-[8px] p-3">
                  <div className="flex items-center gap-2">
                    <div className="text-[11px] font-medium">
                      Disable warm weights
                    </div>
                    <InfoTip text="APEX_DISABLE_WARM_WEIGHTS — When enabled, avoid keeping weights warm/resident between runs." />
                  </div>
                  <Checkbox
                    checked={boolFromEnv(memoryKnobs?.APEX_DISABLE_WARM_WEIGHTS)}
                    onCheckedChange={(checked) =>
                      queueMemoryUpdate({
                        APEX_DISABLE_WARM_WEIGHTS: checked ? "1" : "",
                      })
                    }
                  />
                </div>
                <div className="flex items-center justify-between bg-brand-light/5 border border-brand-light/10 rounded-[8px] p-3">
                  <div className="flex items-center gap-2">
                    <div className="text-[11px] font-medium">Force disk only</div>
                    <InfoTip text="APEX_FORCE_DISK_ONLY — Force everything to disk offload (safest on low VRAM/RAM; slowest)." />
                  </div>
                  <Checkbox
                    checked={boolFromEnv(memoryKnobs?.APEX_FORCE_DISK_ONLY)}
                    onCheckedChange={(checked) =>
                      queueMemoryUpdate({
                        APEX_FORCE_DISK_ONLY: checked ? "1" : "",
                      })
                    }
                  />
                </div>
                <NumberInputSlider
                  label="Target free VRAM fraction"
                  description="APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION — Desired VRAM headroom for weight residency decisions."
                  value={getFloat("APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION", 0.12)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
                <NumberInputSlider
                  label="Target free RAM fraction"
                  description="APEX_WEIGHT_TARGET_FREE_RAM_FRACTION — Desired CPU RAM headroom for weight residency decisions."
                  value={getFloat("APEX_WEIGHT_TARGET_FREE_RAM_FRACTION", 0.1)}
                  onChange={(v) =>
                    queueMemoryUpdate({ APEX_WEIGHT_TARGET_FREE_RAM_FRACTION: v })
                  }
                  min={0}
                  max={1}
                  step={0.01}
                  toFixed={2}
                />
              </div>
            </div>
          </TabsContent>
          <TabsContent value="paths">
            <div className="flex flex-col gap-4 font-poppins text-brand-light h-full">

              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Cache Path</span>
                    {cacheSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {cacheSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={cachePath}
                    onChange={(e) => setCachePath(e.target.value)}
                    placeholder="/path/to/cache"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose cache folder",
                          setCachePath,
                          cachePath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Component Path</span>
                    {componentsSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {componentsSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={componentsPath}
                    onChange={(e) => setComponentsPath(e.target.value)}
                    placeholder="/path/to/components"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose components folder",
                          setComponentsPath,
                          componentsPath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Lora Path</span>
                    {loraSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {loraSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={loraPath}
                    onChange={(e) => setLoraPath(e.target.value)}
                    placeholder="/path/to/lora"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose Lora folder",
                          setLoraPath,
                          loraPath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Config Path</span>
                    {configSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {configSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={configPath}
                    onChange={(e) => setConfigPath(e.target.value)}
                    placeholder="/path/to/components"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose config folder",
                          setConfigPath,
                          configPath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Preprocessor Path</span>
                    {preSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {preSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={preprocessorPath}
                    onChange={(e) => setPreprocessorPath(e.target.value)}
                    placeholder="/path/to/preprocessors"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose preprocessor folder",
                          setPreprocessorPath,
                          preprocessorPath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex flex-row gap-1 items-center justify-between w-full">
                <label className={fieldLabelClass + " w-1/3"}>
                  <div className="flex flex-col w-full">
                    <span>Postprocessor Path</span>
                    {postSizeLabel && (
                      <span className="text-[10px] text-brand-light/60 font-normal">
                        {postSizeLabel}
                      </span>
                    )}
                  </div>
                </label>
                <div className="flex items-center gap-1.5 w-4/5">
                  <Input
                    value={postprocessorPath}
                    onChange={(e) => setPostprocessorPath(e.target.value)}
                    placeholder="/path/to/postprocessors"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
                  {!effectiveIsRemote && (
                    <Button
                      type="button"
                      variant="outline"
                      className="h-7.5 px-3 text-[10.5px] font-medium border-brand-light/20 bg-brand hover:bg-brand-background/80 rounded-[6px]"
                      onClick={() =>
                        handlePickDirectory(
                          "Choose postprocessor folder",
                          setPostprocessorPath,
                          postprocessorPath,
                        )
                      }
                    >
                      <LuFolder className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>
          </div>
        </Tabs>
        <div className="px-4 py-3 border-t bg-brand-background rounded-b-[8px] border-brand-light/10 h-16 flex items-center justify-end gap-2 absolute bottom-0 left-0 right-0">
          <Button
            type="button"
            variant="ghost"
            className="h-8 px-4 text-[11px] w-full font-medium bg-brand-light/5 hover:bg-brand-light/10 rounded-[6px] text-brand-light font-poppins"
            onClick={() => {
              // Cancel should revert any previewed backend URL + local form edits.
              resetToGlobalSettings();
              onOpenChange(false);
            }}
          >
            Cancel
          </Button>
          <Button
            type="button"
            className="h-8 px-5 w-full bg-brand-accent hover:bg-brand-accent-two-shade text-brand-light font-poppins text-[11px] font-medium rounded-[6px] border border-brand-accent-two-shade"
            disabled={saveDisabled}
            onClick={async () => {
              if (saveDisabled) return;
              // Ensure any debounced Memory-tab updates are sent before closing.
              await flushMemoryUpdates();
              const fpsNumeric = Number(projectFps);
              if (Number.isFinite(fpsNumeric) && fpsNumeric > 0) {
                setGlobalFpsWithRescale(fpsNumeric);
              }
              const defaultLen = Number(defaultClipSeconds);
              if (Number.isFinite(defaultLen) && defaultLen > 0) {
                setGlobalDefaultClipLength(defaultLen);
              }

              // If the backend URL changed, persist it first so subsequent config writes
              // go to the correct server.
              if (backendUrlChanged) {
                await setBackendUrlGlobal(backendUrl || null);
              }

              void setCachePathGlobal(cachePath || null);
              void setComponentsPathGlobal(componentsPath || null);
              void setConfigPathGlobal(configPath || null);
              void setLoraPathGlobal(loraPath || null);
              void setPreprocessorPathGlobal(preprocessorPath || null);
              void setPostprocessorPathGlobal(postprocessorPath || null);
              void setHfTokenGlobal(hfToken || null);
              void setCivitaiApiKeyGlobal(civitaiApiKey || null);
              if (!backendUrlChanged) {
                void setBackendUrlGlobal(backendUrl || null);
              }
              void setMaskModelGlobal(maskModel || null);
              void setRenderImageStepsGlobal(renderImageSteps);
              void setRenderVideoStepsGlobal(renderVideoSteps);
              void setUseFastDownloadGlobal(useFastDownload);
              void setAutoUpdateEnabledGlobal(autoUpdateEnabled);
              onOpenChange(false);
            }}
          >
            Save
          </Button>
        </div>
      </DialogContent>
      <button
        onClick={() => onOpenChange(true)}
        className="text-brand-light/90 dark w-8 h-8 flex items-center justify-center rounded-[6px] hover:text-brand-light border border-brand-light/10 bg-brand hover:bg-brand-light/10 transition-all duration-300 cursor-pointer"
      >
        <LuSettings className="w-4 h-4" />
      </button>
    </Dialog>
  )
}
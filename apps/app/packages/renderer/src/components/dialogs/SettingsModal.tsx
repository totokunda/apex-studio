import React, { useEffect, useState } from "react";
import { LuFolder, LuSettings } from "react-icons/lu";
import { LuEye, LuEyeOff } from "react-icons/lu";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { pickMediaPaths } from "@app/preload";
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

  const useFolderSizeEffect = (
    pathValue: string,
    setter: (label: string | null) => void,
  ) => {
    useEffect(() => {
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
    }, [pathValue]);
  };

  useFolderSizeEffect(cachePath, setCacheSizeLabel);
  useFolderSizeEffect(componentsPath, setComponentsSizeLabel);
  useFolderSizeEffect(configPath, setConfigSizeLabel);
  useFolderSizeEffect(loraPath, setLoraSizeLabel);
  useFolderSizeEffect(preprocessorPath, setPreSizeLabel);
  useFolderSizeEffect(postprocessorPath, setPostSizeLabel);

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
              value="tokens"
            >
             API Tokens
            </TabsTrigger>
           
          </TabsList>
          <div className="flex-1 overflow-y-auto pb-14 pr-2.5 custom-scrollbar">
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
                <Input
                  value={backendUrl}
                  onChange={(e) => setBackendUrlLocal(e.target.value)}
                  placeholder="http://127.0.0.1:8765"
                  className="h-7.5 text-[11.5px]! rounded-[6px]"
                />
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
                    placeholder="/path/to/configs"
                    className="h-7.5 text-[11px]! rounded-[6px] disabled:opacity-100 bg-brand! w-full truncate"
                  />
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
                    <LuFolder 
                      className="w-3.5 h-3.5"
                    />
                  </Button>
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
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            className="h-8 px-5 w-full bg-brand-accent hover:bg-brand-accent-two-shade text-brand-light font-poppins text-[11px] font-medium rounded-[6px] border border-brand-accent-two-shade"
            onClick={() => {
              const fpsNumeric = Number(projectFps);
              if (Number.isFinite(fpsNumeric) && fpsNumeric > 0) {
                setGlobalFpsWithRescale(fpsNumeric);
              }
              const defaultLen = Number(defaultClipSeconds);
              if (Number.isFinite(defaultLen) && defaultLen > 0) {
                setGlobalDefaultClipLength(defaultLen);
              }
              void setCachePathGlobal(cachePath || null);
              void setComponentsPathGlobal(componentsPath || null);
              void setConfigPathGlobal(configPath || null);
              void setLoraPathGlobal(loraPath || null);
              void setPreprocessorPathGlobal(preprocessorPath || null);
              void setPostprocessorPathGlobal(postprocessorPath || null);
              void setHfTokenGlobal(hfToken || null);
              void setCivitaiApiKeyGlobal(civitaiApiKey || null);
              void setBackendUrlGlobal(backendUrl || null);
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
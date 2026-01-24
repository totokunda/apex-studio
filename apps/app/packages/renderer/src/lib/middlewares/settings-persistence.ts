import type { StateCreator } from "zustand";
import {
  getAllPathsSetting,
  setAllPathsSetting,
  getHfTokenSetting,
  setHfTokenSetting,
  getCivitaiApiKeySetting,
  setCivitaiApiKeySetting,
  getMaskModelSetting,
  setMaskModelSetting,
  getRenderImageStepsSetting,
  setRenderImageStepsSetting,
  getRenderVideoStepsSetting,
  setRenderVideoStepsSetting,
  getUseFastDownloadSetting,
  setUseFastDownloadSetting,
  getAutoUpdateEnabledSetting,
  setAutoUpdateEnabledSetting,
  getDisableAutoMemoryManagementSetting,
  setDisableAutoMemoryManagementSetting,
} from "@app/preload";
import { getBackendApiUrl, setBackendApiUrl } from "@/lib/config";

export type SettingsState = {
  // Hydration status
  initialized: boolean;
  initializing: boolean;
  error?: string | null;

  // Path settings
  cachePath: string | null;
  componentsPath: string | null;
  configPath: string | null;
  loraPath: string | null;
  preprocessorPath: string | null;
  postprocessorPath: string | null;

  // Auth tokens
  hfToken: string | null;
  civitaiApiKey: string | null;

  // Backend URL
  backendUrl: string | null;

  // Mask model
  maskModel: string | null;

  // Render intermediary steps
  renderImageSteps: boolean;
  renderVideoSteps: boolean;

  // Downloads
  useFastDownload: boolean;

  // API auto updates
  autoUpdateEnabled: boolean;

  // Disable ComfyUI-style auto memory manager hooks
  disableAutoMemoryManagement: boolean;

  // Actions
  hydrate: () => Promise<void>;
  setCachePath: (value: string | null) => Promise<void>;
  setComponentsPath: (value: string | null) => Promise<void>;
  setConfigPath: (value: string | null) => Promise<void>;
  setLoraPath: (value: string | null) => Promise<void>;
  setPreprocessorPath: (value: string | null) => Promise<void>;
  setPostprocessorPath: (value: string | null) => Promise<void>;
  setHfToken: (value: string | null) => Promise<void>;
  setCivitaiApiKey: (value: string | null) => Promise<void>;
  setBackendUrl: (value: string | null) => Promise<void>;
  setMaskModel: (value: string | null) => Promise<void>;
  setRenderImageSteps: (enabled: boolean) => Promise<void>;
  setRenderVideoSteps: (enabled: boolean) => Promise<void>;
  setUseFastDownload: (enabled: boolean) => Promise<void>;
  setAutoUpdateEnabled: (enabled: boolean) => Promise<void>;
  setDisableAutoMemoryManagement: (disabled: boolean) => Promise<void>;
};

export const withSettingsPersistence =
  <T extends SettingsState>(
    config: StateCreator<T, [], []>,
  ): StateCreator<T, [], []> =>
  (set, get, api) => {
    const base = config(set, get, api);

    const hydrate: SettingsState["hydrate"] = async () => {
      const state = get();
      if (state.initializing || state.initialized) return;
      set({ initializing: true, error: null } as Partial<T>);
      try {
        const [
          paths,
          hfToken,
          backendRes,
          civitaiApiKey,
          maskModel,
          renderImageSteps,
          renderVideoSteps,
          useFastDownload,
          autoUpdateEnabled,
          disableAutoMemoryManagement,
        ] = await Promise.all([
          getAllPathsSetting().catch(() => ({} as any)),
          getHfTokenSetting().catch(() => null),
          getBackendApiUrl().catch(() => null),
          getCivitaiApiKeySetting().catch(() => null),
          getMaskModelSetting().catch(() => null),
          getRenderImageStepsSetting().catch(() => false),
          getRenderVideoStepsSetting().catch(() => false),
          getUseFastDownloadSetting().catch(() => true),
          getAutoUpdateEnabledSetting().catch(() => true),
          getDisableAutoMemoryManagementSetting().catch(() => false),
        ]);

        const backendUrl =
          backendRes && backendRes.success && backendRes.data?.url
            ? backendRes.data.url
            : null;

        set({
          cachePath: paths?.cachePath ?? null,
          componentsPath: paths?.componentsPath ?? null,
          configPath: paths?.configPath ?? null,
          loraPath: paths?.loraPath ?? null,
          preprocessorPath: paths?.preprocessorPath ?? null,
          postprocessorPath: paths?.postprocessorPath ?? null,
          hfToken: hfToken ?? null,
          civitaiApiKey: civitaiApiKey ?? null,
          backendUrl,
          maskModel: maskModel ?? null,
          renderImageSteps: Boolean(renderImageSteps),
          renderVideoSteps: Boolean(renderVideoSteps),
          useFastDownload: Boolean(useFastDownload),
          autoUpdateEnabled: Boolean(autoUpdateEnabled),
          disableAutoMemoryManagement: Boolean(disableAutoMemoryManagement),
          initialized: true,
          initializing: false,
          error: null,
        } as Partial<T>);
      } catch (err) {
        set({
          initializing: false,
          error:
            err instanceof Error
              ? err.message
              : "Failed to load settings state",
        } as Partial<T>);
      }
    };

    const enhanced: T = {
      ...base,
      hydrate,
      setCachePath: async (value: string | null) => {
        set({ cachePath: value ?? null } as Partial<T>);
        void setAllPathsSetting({ cachePath: value ?? null }).catch(
          () => undefined,
        );
      },
      setComponentsPath: async (value: string | null) => {
        set({ componentsPath: value ?? null } as Partial<T>);
        void setAllPathsSetting({
          componentsPath: value ?? null,
        }).catch(() => undefined);
      },
      setConfigPath: async (value: string | null) => {
        set({ configPath: value ?? null } as Partial<T>);
        void setAllPathsSetting({
          configPath: value ?? null,
        }).catch(() => undefined);
      },
      setLoraPath: async (value: string | null) => {
        set({ loraPath: value ?? null } as Partial<T>);
        void setAllPathsSetting({ loraPath: value ?? null }).catch(
          () => undefined,
        );
      },
      setPreprocessorPath: async (value: string | null) => {
        set({ preprocessorPath: value ?? null } as Partial<T>);
        void setAllPathsSetting({
          preprocessorPath: value ?? null,
        }).catch(() => undefined);
      },
      setPostprocessorPath: async (value: string | null) => {
        set({ postprocessorPath: value ?? null } as Partial<T>);
        void setAllPathsSetting({
          postprocessorPath: value ?? null,
        }).catch(() => undefined);
      },
      setHfToken: async (value: string | null) => {
        set({ hfToken: value ?? null } as Partial<T>);
        void setAllPathsSetting({ hfToken: value ?? null }).catch(
          () => undefined,
        );
        void setHfTokenSetting(value ?? null).catch(() => undefined);
      },
      setCivitaiApiKey: async (value: string | null) => {
        set({ civitaiApiKey: value ?? null } as Partial<T>);
        void setAllPathsSetting({ civitaiApiKey: value ?? null }).catch(
          () => undefined,
        );
        void setCivitaiApiKeySetting(value ?? null).catch(() => undefined);
      },
      setBackendUrl: async (value: string | null) => {
        const url = (value ?? "").trim() || null;
        set({ backendUrl: url } as Partial<T>);
        if (!url) return;

        // Persist in main process
        const res = await setBackendApiUrl(url).catch(() => null);
        if (!res?.success) return;
      },
      setMaskModel: async (value: string | null) => {
        set({ maskModel: value ?? null } as Partial<T>);
        void setMaskModelSetting(value ?? null).catch(
          () => undefined,
        );
      },
      setRenderImageSteps: async (enabled: boolean) => {
        const v = Boolean(enabled);
        set({ renderImageSteps: v } as Partial<T>);
        void setRenderImageStepsSetting(v).catch(() => undefined);
      },
      setRenderVideoSteps: async (enabled: boolean) => {
        const v = Boolean(enabled);
        set({ renderVideoSteps: v } as Partial<T>);
        void setRenderVideoStepsSetting(v).catch(() => undefined);
      },
      setUseFastDownload: async (enabled: boolean) => {
        const v = Boolean(enabled);
        set({ useFastDownload: v } as Partial<T>);
        void setUseFastDownloadSetting(v).catch(() => undefined);
      },
      setAutoUpdateEnabled: async (enabled: boolean) => {
        const v = Boolean(enabled);
        set({ autoUpdateEnabled: v } as Partial<T>);
        void setAutoUpdateEnabledSetting(v).catch(() => undefined);
      },
      setDisableAutoMemoryManagement: async (disabled: boolean) => {
        const v = Boolean(disabled);
        set({ disableAutoMemoryManagement: v } as Partial<T>);
        void setDisableAutoMemoryManagementSetting(v).catch(() => undefined);
      },
    } as T;

    return enhanced;
  };



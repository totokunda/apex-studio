import { create } from "zustand";
import {
  withSettingsPersistence,
  type SettingsState,
} from "./middlewares/settings-persistence";

export const useSettingsStore = create<SettingsState>(
  withSettingsPersistence<SettingsState>(() => ({
    initialized: false,
    initializing: false,
    error: null,
    maskModel: null,
    cachePath: null,
    componentsPath: null,
    configPath: null,
    loraPath: null,
    preprocessorPath: null,
    postprocessorPath: null,
    hfToken: null,
    civitaiApiKey: null,
    backendUrl: null,
    renderImageSteps: false,
    renderVideoSteps: false,
    useFastDownload: true,

    // These will be replaced by the middleware with real implementations.
    hydrate: async () => {},
    setCachePath: async () => {},
    setComponentsPath: async () => {},
    setConfigPath: async () => {},
    setLoraPath: async () => {},
    setMaskModel: async () => {},
    setPreprocessorPath: async () => {},
    setPostprocessorPath: async () => {},
    setHfToken: async () => {},
    setCivitaiApiKey: async () => {},
    setBackendUrl: async () => {},
    setRenderImageSteps: async () => {},
    setRenderVideoSteps: async () => {},
    setUseFastDownload: async () => {},
  })),
);


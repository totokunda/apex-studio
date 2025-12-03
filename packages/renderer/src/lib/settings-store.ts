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

    cachePath: null,
    componentsPath: null,
    configPath: null,
    loraPath: null,
    preprocessorPath: null,
    postprocessorPath: null,
    hfToken: null,
    backendUrl: null,

    // These will be replaced by the middleware with real implementations.
    hydrate: async () => {},
    setCachePath: async () => {},
    setComponentsPath: async () => {},
    setConfigPath: async () => {},
    setLoraPath: async () => {},
    setPreprocessorPath: async () => {},
    setPostprocessorPath: async () => {},
    setHfToken: async () => {},
    setBackendUrl: async () => {},
  })),
);


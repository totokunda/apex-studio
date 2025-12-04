import type { StateCreator } from "zustand";
import {
  getAllPathsSetting,
  setAllPathsSetting,
  getHfTokenSetting,
  setHfTokenSetting,
  getCivitaiApiKeySetting,
  setCivitaiApiKeySetting,
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
        const [paths, hfToken, backendRes, civitaiApiKey] = await Promise.all([
          getAllPathsSetting().catch(() => ({} as any)),
          getHfTokenSetting().catch(() => null),
          getBackendApiUrl().catch(() => null),
          getCivitaiApiKeySetting().catch(() => null),
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
        if (url) {
          void setBackendApiUrl(url).catch(() => undefined);
        }
      },
    } as T;

    return enhanced;
  };



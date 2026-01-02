import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
type SupportedExtension =
  | "REDUX_DEVTOOLS"
  | "VUEJS_DEVTOOLS"
  | "EMBER_INSPECTOR"
  | "BACKBONE_DEBUGGER"
  | "REACT_DEVELOPER_TOOLS"
  | "JQUERY_DEBUGGER"
  | "MOBX_DEVTOOLS";

export class ChromeDevToolsExtension implements AppModule {
  readonly #extension: SupportedExtension;

  constructor({
    extension,
  }: {
    readonly extension: SupportedExtension;
  }) {
    this.#extension = extension;
  }

  async enable({ app }: ModuleContext): Promise<void> {
    // Only install extensions in development and never fail app startup
    if (!import.meta.env.DEV) {
      return;
    }

    await app.whenReady();

    try {
      // `electron-devtools-installer` is a devDependency and is not shipped in production builds.
      // Use a dynamic import so packaged apps never crash with ERR_MODULE_NOT_FOUND.
      const installer = (await import("electron-devtools-installer")) as any;
      const dict = {
        REDUX_DEVTOOLS: installer.REDUX_DEVTOOLS,
        VUEJS_DEVTOOLS: installer.VUEJS_DEVTOOLS,
        EMBER_INSPECTOR: installer.EMBER_INSPECTOR,
        BACKBONE_DEBUGGER: installer.BACKBONE_DEBUGGER,
        REACT_DEVELOPER_TOOLS: installer.REACT_DEVELOPER_TOOLS,
        JQUERY_DEBUGGER: installer.JQUERY_DEBUGGER,
        MOBX_DEVTOOLS: installer.MOBX_DEVTOOLS,
      } as const;
      const installExtension = installer.default as (
        ext: unknown,
      ) => Promise<unknown>;
      await installExtension(dict[this.#extension]);
    } catch (error) {
      // Extensions are nice-to-have in dev; log and continue instead of crashing
      console.warn(
        "[ChromeDevToolsExtension] Failed to install devtools extension:",
        error,
      );
    }
  }
}

export function chromeDevToolsExtension(
  ...args: ConstructorParameters<typeof ChromeDevToolsExtension>
) {
  return new ChromeDevToolsExtension(...args);
}

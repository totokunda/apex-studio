import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import installer from "electron-devtools-installer";

const {
  REDUX_DEVTOOLS,
  VUEJS_DEVTOOLS,
  EMBER_INSPECTOR,
  BACKBONE_DEBUGGER,
  REACT_DEVELOPER_TOOLS,
  JQUERY_DEBUGGER,
  MOBX_DEVTOOLS,
  default: installExtension,
} = installer;

const extensionsDictionary = {
  REDUX_DEVTOOLS,
  VUEJS_DEVTOOLS,
  EMBER_INSPECTOR,
  BACKBONE_DEBUGGER,
  REACT_DEVELOPER_TOOLS,
  JQUERY_DEBUGGER,
  MOBX_DEVTOOLS,
} as const;

export class ChromeDevToolsExtension implements AppModule {
  readonly #extension: keyof typeof extensionsDictionary;

  constructor({
    extension,
  }: {
    readonly extension: keyof typeof extensionsDictionary;
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
      await installExtension(extensionsDictionary[this.#extension]);
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

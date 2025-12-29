import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { ipcMain, dialog } from "electron";

type OpenDialogOptions = {
  directory?: boolean;
  title?: string;
  filters?: { name: string; extensions: string[] }[];
  defaultPath?: string;
};

export class NativeFileDialog implements AppModule {
  enable(_context: ModuleContext): void | Promise<void> {
    // Register once; remove any previous to avoid duplication during dev HMR
    try {
      ipcMain.removeHandler("dialog:pick-media");
    } catch {}

    ipcMain.handle(
      "dialog:pick-media",
      async (_evt, opts: OpenDialogOptions) => {
        const properties: Array<
          "openFile" | "openDirectory" | "multiSelections" | "dontAddToRecent"
        > = ["multiSelections", "dontAddToRecent"];
        if (opts?.directory) properties.push("openDirectory");
        else properties.push("openFile");
        const res = await dialog.showOpenDialog({
          properties,
          filters: opts?.filters,
          title: opts?.title,
          defaultPath: opts?.defaultPath,
        });
        if (res.canceled) return [] as string[];
        return res.filePaths as string[];
      },
    );
  }
}

export function createNativeFileDialogModule(): NativeFileDialog {
  return new NativeFileDialog();
}

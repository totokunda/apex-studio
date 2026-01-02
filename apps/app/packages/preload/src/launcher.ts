import { ipcRenderer } from "electron";

/**
 * Tell the main process to open the main app window and close the launcher window.
 */
export async function launchMainWindow(): Promise<{ ok: true } | { ok: false; error: string }> {
  try {
    await ipcRenderer.invoke("launcher:launch");
    return { ok: true };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "Failed to launch main window",
    };
  }
}



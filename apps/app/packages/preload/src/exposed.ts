import * as exports from "./index.js";
import { contextBridge } from "electron";

const isExport = (key: string): key is keyof typeof exports =>
  Object.hasOwn(exports, key);

let useContextBridge = true;
for (const exportsKey in exports) {
  if (isExport(exportsKey)) {
    const key = btoa(exportsKey);
    try {
      if (useContextBridge) {
        contextBridge.exposeInMainWorld(key, exports[exportsKey]);
      } else {
        (globalThis as Record<string, unknown>)[key] = exports[exportsKey];
      }
    } catch {
      useContextBridge = false;
      (globalThis as Record<string, unknown>)[key] = exports[exportsKey];
    }
  }
}

// Re-export for tests
export * from "./index.js";

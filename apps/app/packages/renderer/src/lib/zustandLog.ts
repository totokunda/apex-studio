// withStackLog.ts
import type { StateCreator, StoreApi } from "zustand";

/**
 * Type-safe logging middleware for Zustand.
 * - Preserves T, set/get/api types
 * - Prints a stack trace for every set()
 * - Shows shallow prev/next diff (cheap)
 */
export const withStackLog =
  <T extends object>(
    config: StateCreator<T, [], []>,
  ): StateCreator<T, [], []> =>
  (set, get, api) => {
    const setWithLog: typeof set = (partial, replace) => {
      const prev = get();
      console.groupCollapsed("[zustand] set()");
      console.trace(); // precise call site
      try {
        return set(partial as any, replace as any);
      } finally {
        const next = get();
        logShallowDiff(prev, next);
        console.groupEnd();
      }
    };
    return config(setWithLog, get, api as StoreApi<T>);
  };

function logShallowDiff(prev: object, next: object) {
  const changes: string[] = [];
  const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
  for (const k of keys) {
    const a = (prev as any)[k];
    const b = (next as any)[k];
    if (a !== b) changes.push(`${k}: ${fmt(a)} → ${fmt(b)}`);
  }
  if (changes.length) console.log("diff", changes.join(", "));
}

const fmt = (v: unknown) =>
  typeof v === "object" && v !== null ? JSON.stringify(v) : String(v);

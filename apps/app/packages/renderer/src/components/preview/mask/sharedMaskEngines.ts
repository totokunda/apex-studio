import { LassoMask } from "./lasso";
import { ShapeMask } from "./shape";
import { TouchMask } from "./touch";

export interface SharedMaskEngines {
  contextKey: string;
  shape: ShapeMask;
  lasso: LassoMask;
  touch: TouchMask;
}

/**
 * Shared mask engines keyed by a WebGLContextManager key.
 *
 * Why this exists:
 * - Creating per-clip/per-call WebGL contexts quickly hits Chromium's context limit
 *   ("Too many active WebGL contexts"), causing context loss and downstream shader failures.
 * - Masks are applied synchronously on the main thread, so sharing a single GL canvas/context
 *   is safe as long as callers copy the result out immediately (which our mask applicators do).
 */
const sharedByKey = new Map<string, SharedMaskEngines>();

export function getSharedMaskEngines(contextKey: string): SharedMaskEngines {
  const existing = sharedByKey.get(contextKey);
  if (existing) return existing;

  // All mask engines share the same underlying WebGL context (same key).
  const engines: SharedMaskEngines = {
    contextKey,
    shape: new ShapeMask(contextKey),
    lasso: new LassoMask(contextKey),
    touch: new TouchMask(contextKey),
  };

  sharedByKey.set(contextKey, engines);
  return engines;
}

/**
 * Optional manual cleanup (typically only useful in tests or on app shutdown).
 * In normal app usage we intentionally keep the shared engines alive to avoid
 * WebGL context churn (create/lose/create loops).
 */
export function disposeSharedMaskEngines(contextKey?: string): void {
  if (typeof contextKey === "string") {
    const engines = sharedByKey.get(contextKey);
    if (!engines) return;
    try {
      engines.shape.dispose();
    } catch {}
    try {
      engines.lasso.dispose();
    } catch {}
    try {
      engines.touch.dispose();
    } catch {}
    sharedByKey.delete(contextKey);
    return;
  }

  for (const [key, engines] of sharedByKey.entries()) {
    try {
      engines.shape.dispose();
    } catch {}
    try {
      engines.lasso.dispose();
    } catch {}
    try {
      engines.touch.dispose();
    } catch {}
    sharedByKey.delete(key);
  }
}


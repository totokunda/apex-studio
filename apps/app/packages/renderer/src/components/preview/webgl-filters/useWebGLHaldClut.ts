/**
 * Custom hook for managing a singleton instance of WebGLHaldClut
 * Ensures the same instance is used throughout the app for efficient resource management
 */

import { useEffect, useRef, useState } from "react";
import { WebGLHaldClut } from "./hald-clut";

// Module-level singleton instance
let haldClutInstance: WebGLHaldClut | null = null;
let referenceCount = 0;

/**
 * Hook that provides access to a shared WebGLHaldClut instance
 * The instance is created on first use and cleaned up when all components unmount
 */
export function useWebGLHaldClut() {
  const isInitializedRef = useRef(false);
  const [instance, setInstance] = useState<WebGLHaldClut | null>(
    haldClutInstance,
  );

  useEffect(() => {
    // Initialize instance on first mount
    if (!isInitializedRef.current) {
      if (!haldClutInstance) {
        haldClutInstance = new WebGLHaldClut();
      }
      // Always set instance to ensure component has the latest reference
      setInstance(haldClutInstance);
      referenceCount++;
      isInitializedRef.current = true;
    }

    // Cleanup on unmount
    return () => {
      if (isInitializedRef.current) {
        referenceCount--;

        // Only dispose when no components are using it
        if (referenceCount === 0 && haldClutInstance) {
          haldClutInstance.dispose();
          haldClutInstance = null;
          setInstance(null);
        }

        isInitializedRef.current = false;
      }
    };
  }, []);

  return instance;
}

/**
 * Manually dispose the singleton instance
 * Use with caution - only call when you're sure no components are using it
 */
export function disposeHaldClutSingleton() {
  if (haldClutInstance) {
    haldClutInstance.dispose();
    haldClutInstance = null;
    referenceCount = 0;
  }
}

/**
 * Get the current reference count (useful for debugging)
 */
export function getHaldClutReferenceCount(): number {
  return referenceCount;
}

import { WebGLHaldClut } from "../../../renderer/src/components/preview/webgl-filters/hald-clut";

let haldClutInstance: WebGLHaldClut | null = null;
let referenceCount = 0;

export function acquireHaldClut(): WebGLHaldClut {
  if (!haldClutInstance) {
    haldClutInstance = new WebGLHaldClut();
  }
  referenceCount++;
  return haldClutInstance;
}

export function releaseHaldClut(): void {
  if (referenceCount > 0) {
    referenceCount--;
    if (referenceCount === 0 && haldClutInstance) {
      haldClutInstance.dispose();
      haldClutInstance = null;
    }
  }
}

export function getHaldClutInstance(): WebGLHaldClut | null {
  return haldClutInstance;
}

export function disposeHaldClutSingleton(): void {
  if (haldClutInstance) {
    haldClutInstance.dispose();
    haldClutInstance = null;
  }
  referenceCount = 0;
}

export function getHaldClutReferenceCount(): number {
  return referenceCount;
}

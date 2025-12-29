type WebGLContextType = "webgl" | "webgl2";

export interface WebGLContextListener {
  onContextLost?: () => void;
  onContextRestored?: (
    gl: WebGLRenderingContext | WebGL2RenderingContext,
  ) => void;
}

export interface WebGLSharedContextHandle {
  readonly canvas: HTMLCanvasElement;
  getContext(): WebGLRenderingContext | WebGL2RenderingContext | null;
  ensureContext(): WebGLRenderingContext | WebGL2RenderingContext | null;
  subscribe(listener: WebGLContextListener): () => void;
  release(): void;
}

interface ContextRecord {
  key: string;
  canvas: HTMLCanvasElement;
  contextType: WebGLContextType;
  attributes?: WebGLContextAttributes;
  gl: WebGLRenderingContext | WebGL2RenderingContext | null;
  refCount: number;
  listeners: Set<WebGLContextListener>;
  lostHandler: (event: Event) => void;
  restoredHandler: (event: Event) => void;
}

interface AcquireOptions {
  contextType?: WebGLContextType;
  attributes?: WebGLContextAttributes;
}

class WebGLContextHandleImpl implements WebGLSharedContextHandle {
  private released = false;
  private unsubscribeCallbacks: Array<() => void> = [];

  constructor(
    private readonly manager: typeof WebGLContextManager,
    private readonly record: ContextRecord,
  ) {}

  get canvas(): HTMLCanvasElement {
    return this.record.canvas;
  }

  getContext(): WebGLRenderingContext | WebGL2RenderingContext | null {
    return this.record.gl;
  }

  ensureContext(): WebGLRenderingContext | WebGL2RenderingContext | null {
    if (this.released) {
      return null;
    }
    return this.manager.ensureContext(this.record);
  }

  subscribe(listener: WebGLContextListener): () => void {
    if (this.released) {
      return () => undefined;
    }
    this.record.listeners.add(listener);
    const unsubscribe = () => {
      this.record.listeners.delete(listener);
    };
    this.unsubscribeCallbacks.push(unsubscribe);
    return unsubscribe;
  }

  release(): void {
    if (this.released) {
      return;
    }
    this.released = true;
    // Ensure listeners are removed before releasing the record.
    this.unsubscribeCallbacks.forEach((fn) => fn());
    this.unsubscribeCallbacks = [];
    //@ts-ignore
    this.manager.release(this.record);
  }
}

export class WebGLContextManager {
  private static contexts = new Map<string, ContextRecord>();

  static acquire(
    key: string,
    options: AcquireOptions = {},
  ): WebGLSharedContextHandle {
    let record = this.contexts.get(key);
    if (!record) {
      const canvas = document.createElement("canvas");
      const contextType = options.contextType ?? "webgl";
      const attributes = options.attributes;

      record = {
        key,
        canvas,
        contextType,
        attributes,
        gl: null,
        refCount: 0,
        listeners: new Set(),
        lostHandler: () => undefined,
        restoredHandler: () => undefined,
      };

      record.lostHandler = (event: Event) => {
        event.preventDefault();
        // Mark GL as lost so ensureContext can recreate it later.
        record!.gl = null;
        const listeners = Array.from(record!.listeners);
        listeners.forEach((listener) => {
          listener.onContextLost?.();
        });
      };

      record.restoredHandler = () => {
        const gl = this.ensureContext(record!);
        if (!gl) {
          console.error("Failed to restore shared WebGL context");
          return;
        }
        const listeners = Array.from(record!.listeners);
        listeners.forEach((listener) => {
          listener.onContextRestored?.(gl);
        });
      };

      canvas.addEventListener("webglcontextlost", record.lostHandler, false);
      canvas.addEventListener(
        "webglcontextrestored",
        record.restoredHandler,
        false,
      );

      this.ensureContext(record);
      this.contexts.set(key, record);
    }

    record.refCount += 1;
    return new WebGLContextHandleImpl(this, record);
  }

  private static createContext(
    canvas: HTMLCanvasElement,
    type: WebGLContextType,
    attributes?: WebGLContextAttributes,
  ): WebGLRenderingContext | WebGL2RenderingContext | null {
    if (type === "webgl2") {
      const gl2 = canvas.getContext(
        "webgl2",
        attributes,
      ) as WebGL2RenderingContext | null;
      if (gl2) {
        return gl2;
      }
      console.warn(
        "WebGL2 not available, falling back to WebGL1 for shared context",
      );
    }
    const gl = canvas.getContext(
      "webgl",
      attributes,
    ) as WebGLRenderingContext | null;
    if (!gl) {
      console.error("Failed to initialize shared WebGL context");
    }
    return gl;
  }

  static ensureContext(
    record: ContextRecord,
  ): WebGLRenderingContext | WebGL2RenderingContext | null {
    if (
      record.gl &&
      typeof record.gl.isContextLost === "function" &&
      !record.gl.isContextLost()
    ) {
      return record.gl;
    }
    record.gl = WebGLContextManager.createContext(
      record.canvas,
      record.contextType,
      record.attributes,
    );
    return record.gl;
  }

  private static release(record: ContextRecord): void {
    const existing = this.contexts.get(record.key);
    if (!existing) {
      return;
    }
    existing.refCount -= 1;
    if (existing.refCount > 0) {
      return;
    }

    existing.canvas.removeEventListener(
      "webglcontextlost",
      existing.lostHandler,
    );
    existing.canvas.removeEventListener(
      "webglcontextrestored",
      existing.restoredHandler,
    );

    // Attempt to proactively release GPU resources.
    if (existing.gl) {
      const loseExt = existing.gl.getExtension("WEBGL_lose_context");
      loseExt?.loseContext();
      existing.gl = null;
    }

    existing.listeners.clear();
    this.contexts.delete(existing.key);
  }
}

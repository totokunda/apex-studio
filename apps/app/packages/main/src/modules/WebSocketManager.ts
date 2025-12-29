import WebSocket from "ws";

type WebSocketHandlers = {
  onOpen?: () => void;
  onMessage?: (data: string) => void;
  onError?: (error: Error) => void;
  onClose?: (code: number, reason: string) => void;
};

export class WebSocketManager {
  private baseHttpUrl: string;
  private connections: Map<string, WebSocket> = new Map();

  constructor(baseHttpUrl: string) {
    this.baseHttpUrl = baseHttpUrl;
  }

  public setBaseUrl(baseHttpUrl: string): void {
    this.baseHttpUrl = baseHttpUrl;
  }

  private getBaseWsUrl(): string {
    return this.baseHttpUrl
      .replace("http://", "ws://")
      .replace("https://", "wss://");
  }

  private resolveUrl(pathOrUrl: string): string {
    if (pathOrUrl.startsWith("ws://") || pathOrUrl.startsWith("wss://")) {
      return pathOrUrl;
    }
    const base = this.getBaseWsUrl();
    if (pathOrUrl.startsWith("/")) {
      return `${base}${pathOrUrl}`;
    }
    return `${base}/${pathOrUrl}`;
  }

  public has(key: string): boolean {
    return this.connections.has(key);
  }

  public connect(
    key: string,
    pathOrUrl: string,
    handlers: WebSocketHandlers = {},
  ): { success: boolean; error?: string } {
    try {
      // Close existing connection if any
      const existing = this.connections.get(key);
      if (existing && existing.readyState === WebSocket.OPEN) {
        try {
          existing.close();
        } catch {}
      }
      if (existing) {
        this.connections.delete(key);
      }

      const url = this.resolveUrl(pathOrUrl);
      const ws = new WebSocket(url);

      ws.on("open", () => {
        handlers.onOpen?.();
      });

      ws.on("message", (data) => {
        try {
          const asString = data.toString();
          handlers.onMessage?.(asString);
        } catch (err) {
          handlers.onError?.(err as Error);
        }
      });

      ws.on("error", (error) => {
        handlers.onError?.(error as Error);
      });

      ws.on("close", (code, reasonBuffer) => {
        const reason = reasonBuffer?.toString?.() ?? "";
        this.connections.delete(key);
        handlers.onClose?.(code, reason);
      });

      this.connections.set(key, ws);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  public disconnect(key: string): { success: boolean; error?: string } {
    try {
      const ws = this.connections.get(key);
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.close();
        } catch {}
      }
      if (ws) {
        this.connections.delete(key);
      }
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }
}

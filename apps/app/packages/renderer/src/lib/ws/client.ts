import {
  wsConnect,
  wsDisconnect,
  wsStatus,
  onWsUpdate,
  onWsStatus,
  onWsError,
} from "@app/preload";

type Unsubscribe = () => void;

export class WsClient {
  private unsubscribersByKey: Map<string, Unsubscribe[]> = new Map();

  async connect(key: string, pathOrUrl: string): Promise<void> {
    const res = await wsConnect(key, pathOrUrl);
    if (!res.success) throw new Error(res.error || "Failed to connect");
  }

  async disconnect(key: string): Promise<void> {
    try {
      const list = this.unsubscribersByKey.get(key) || [];
      list.forEach((fn) => {
        try {
          fn();
        } catch {}
      });
      this.unsubscribersByKey.delete(key);
    } catch {}
    await wsDisconnect(key).catch(() => {});
  }

  async status(key: string): Promise<boolean> {
    const res = await wsStatus(key);
    return !!(res.success && res.data?.connected);
  }

  onUpdate(key: string, cb: (data: any) => void): Unsubscribe {
    const off = onWsUpdate(key, cb);
    const list = this.unsubscribersByKey.get(key) || [];
    list.push(off);
    this.unsubscribersByKey.set(key, list);
    return off;
  }

  onStatus(key: string, cb: (data: any) => void): Unsubscribe {
    const off = onWsStatus(key, cb);
    const list = this.unsubscribersByKey.get(key) || [];
    list.push(off);
    this.unsubscribersByKey.set(key, list);
    return off;
  }

  onError(key: string, cb: (data: any) => void): Unsubscribe {
    const off = onWsError(key, cb);
    const list = this.unsubscribersByKey.get(key) || [];
    list.push(off);
    this.unsubscribersByKey.set(key, list);
    return off;
  }
}

export const wsClient = new WsClient();

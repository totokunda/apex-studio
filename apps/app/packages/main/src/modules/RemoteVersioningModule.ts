import type { AppModule } from "../AppModule.js";
import { ipcMain } from "electron";

export type HostPlatform = NodeJS.Platform;
export type HostArch = NodeJS.Architecture;

export interface ServerBundleAsset {
  /** Version folder name on the remote host, e.g. "v0.1.0" */
  tag: string;
  /** Parsed semver from tag, e.g. "0.1.0" */
  tagVersion: string;
  /** Optional display name (not currently provided for Hugging Face sources) */
  releaseName?: string;
  /** ISO timestamp (not currently provided for Hugging Face sources) */
  publishedAt?: string;
  prerelease?: boolean;

  /** Asset filename, e.g. "python-api-0.1.0-darwin-arm64-cpu-cp312.tar.zst" */
  assetName: string;
  /** Browser download URL for the asset */
  downloadUrl: string;
  /** Size in bytes (if provided by the remote host) */
  size?: number;

  /** Parsed from the asset filename */
  assetVersion: string;
  platform: string;
  arch: string;
  device: string;
  pythonTag: string;
}

export interface ListServerVersionsResponse {
  host: { platform: HostPlatform; arch: HostArch };
  repo: { owner: string; name: string };
  items: ServerBundleAsset[];
}

function parseSemverTriplet(v: string): [number, number, number] | null {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(v.trim());
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function compareSemverDesc(a: string, b: string): number {
  const pa = parseSemverTriplet(a);
  const pb = parseSemverTriplet(b);
  // Unknown versions sort last.
  if (!pa && !pb) return 0;
  if (!pa) return 1;
  if (!pb) return -1;
  if (pa[0] !== pb[0]) return pb[0] - pa[0];
  if (pa[1] !== pb[1]) return pb[1] - pa[1];
  return pb[2] - pa[2];
}

function normalizePlatformCandidates(hostPlatform: HostPlatform): string[] {
  // Asset naming tends to mirror Node's platform values, but support a couple common aliases.
  const base = hostPlatform;
  const out = new Set<string>([base]);
  if (hostPlatform === "win32") out.add("windows");
  if (hostPlatform === "darwin") out.add("macos");
  return [...out];
}

function normalizeArchCandidates(hostArch: HostArch): string[] {
  const base = hostArch;
  const out = new Set<string>([base]);
  if (hostArch === "x64") {
    out.add("amd64");
    out.add("x86_64");
  }
  if (hostArch === "arm64") out.add("aarch64");
  return [...out];
}

function encodePathSegments(path: string): string {
  return path
    .split("/")
    .filter(Boolean)
    .map((s) => encodeURIComponent(s))
    .join("/");
}

function parsePythonBundleAssetName(filename: string): Omit<
  ServerBundleAsset,
  | "tag"
  | "tagVersion"
  | "releaseName"
  | "publishedAt"
  | "prerelease"
  | "assetName"
  | "downloadUrl"
  | "size"
> | null {
  // Examples:
  // - python-api-0.1.0-darwin-arm64-cpu-cp312.tar.zst
  // - python-code-0.1.0-linux-x86_64-cuda126-cp312.tar.zst
  const re =
    /^python-(?:api|code)-(?<version>[^-]+)-(?<platform>[^-]+)-(?<arch>[^-]+)-(?<device>[^-]+)-(?<python>[^.]+)\.tar\.zst$/i;
  const m = re.exec(filename);
  if (!m || !m.groups) return null;
  return {
    assetVersion: m.groups.version,
    platform: m.groups.platform,
    arch: m.groups.arch,
    device: m.groups.device,
    pythonTag: m.groups.python,
  };
}

type HfRepoTreeItem = {
  type?: "file" | "directory";
  path?: string;
  size?: number;
  lfs?: { size?: number };
};

class RemoteVersioningService {
  readonly #owner: string;
  readonly #repo: string;
  readonly #apiBase = "https://huggingface.co";
  readonly #revision = "main";
  #cache:
    | {
        at: number;
        hostKey: string;
        payload: ListServerVersionsResponse;
      }
    | null = null;

  constructor({ owner, repo }: { owner: string; repo: string }) {
    this.#owner = owner;
    this.#repo = repo;
  }

  async #listTree(path?: string): Promise<HfRepoTreeItem[]> {
    const base = `${this.#apiBase}/api/models/${this.#owner}/${this.#repo}/tree/${this.#revision}`;
    const url = path ? `${base}/${encodePathSegments(path)}` : base;
    const res = await fetch(url, {
      headers: {
        Accept: "application/json",
        "User-Agent": "apex-studio",
      },
    });
    if (!res.ok) {
      throw new Error(
        `Failed to fetch Hugging Face repo tree: ${res.status} ${res.statusText}`,
      );
    }
    const json = (await res.json()) as unknown;
    return Array.isArray(json) ? (json as HfRepoTreeItem[]) : [];
  }

  async listServerVersionsForHost(host: {
    platform: HostPlatform;
    arch: HostArch;
  }): Promise<ListServerVersionsResponse> {
    const hostKey = `${host.platform}/${host.arch}`;
    const now = Date.now();
    if (this.#cache && this.#cache.hostKey === hostKey && now - this.#cache.at < 60_000) {
      return this.#cache.payload;
    }

    const items: ServerBundleAsset[] = [];
    const platformCandidates = normalizePlatformCandidates(host.platform);
    const archCandidates = normalizeArchCandidates(host.arch);

    const root = await this.#listTree();
    const versionDirs = root
      .filter((x) => x?.type === "directory" && typeof x.path === "string")
      .map((x) => String(x.path))
      .filter((p) => /^v?\d+\.\d+\.\d+$/i.test(p.split("/").pop() || ""))
      .sort((a, b) => {
        const va = (a.split("/").pop() || "").replace(/^v/i, "");
        const vb = (b.split("/").pop() || "").replace(/^v/i, "");
        return compareSemverDesc(va, vb);
      });

    for (const dirPath of versionDirs) {
      const dirName = dirPath.split("/").pop() || dirPath;
      const tag = dirName;
      const tagVersion = dirName.replace(/^v/i, "");

      const entries = await this.#listTree(dirPath);
      for (const e of entries) {
        if (e?.type !== "file" || typeof e.path !== "string") continue;
        const fullPath = String(e.path);
        const assetName = fullPath.split("/").pop() || fullPath;
        if (!assetName.toLowerCase().endsWith(".tar.zst")) continue;

        const parsed = parsePythonBundleAssetName(assetName);
        if (!parsed) continue;

        // Quick host verify (platform + arch)
        const platformOk = platformCandidates.includes(parsed.platform);
        const archOk = archCandidates.includes(parsed.arch);
        if (!platformOk || !archOk) continue;

        const downloadUrl = `${this.#apiBase}/${this.#owner}/${this.#repo}/resolve/${this.#revision}/${encodePathSegments(fullPath)}?download=true`;
        const size = e.lfs?.size ?? e.size;

        items.push({
          tag,
          tagVersion,
          prerelease: false,
          assetName,
          downloadUrl,
          size,
          ...parsed,
        });
      }
    }

    // Sort by tag semver desc, then by asset version desc-ish (best-effort), then name.
    items.sort((a, b) => {
      const byTag = compareSemverDesc(a.tagVersion, b.tagVersion);
      if (byTag !== 0) return byTag;
      const byAsset = compareSemverDesc(a.assetVersion, b.assetVersion);
      if (byAsset !== 0) return byAsset;
      return a.assetName.localeCompare(b.assetName);
    });

    const payload: ListServerVersionsResponse = {
      host,
      repo: { owner: this.#owner, name: this.#repo },
      items,
    };
    this.#cache = { at: now, hostKey, payload };
    return payload;
  }
}

export class RemoteVersioningModule implements AppModule {
  readonly #service: RemoteVersioningService;

  constructor({ owner, repo }: { owner: string; repo: string }) {
    this.#service = new RemoteVersioningService({ owner, repo });
  }

  async enable(): Promise<void> {
    // idempotent-ish registration
    if (ipcMain.listenerCount("versions:list-server-releases") > 0) return;

    ipcMain.handle("versions:list-server-releases", async () => {
      try {
        const platform = process.platform;
        const arch = process.arch;
        const data = await this.#service.listServerVersionsForHost({
          platform,
          arch,
        });
        return { success: true, data };
      } catch (e) {
        return {
          success: false,
          error:
            e instanceof Error ? e.message : "Failed to list server versions",
        };
      }
    });
  }
}

export function remoteVersioningModule() {
  // Hugging Face Hub repo where server bundles are published.
  return new RemoteVersioningModule({ owner: "totoku", repo: "apex-studio-server" });
}



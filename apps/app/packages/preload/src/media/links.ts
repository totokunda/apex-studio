import { promises as fsp } from "node:fs";
import fs from "node:fs";
import { join, dirname, extname, basename } from "node:path";

export type LinksIndex = {
  originalToConverted: Record<string, string>;
  convertedToOriginal: Record<string, string>;
};

export function linksFilePath(mediaRootAbs: string): string {
  return join(mediaRootAbs, ".links.json");
}

export async function loadLinks(mediaRootAbs: string): Promise<LinksIndex> {
  try {
    const p = linksFilePath(mediaRootAbs);
    const s = await fsp.readFile(p, "utf8");
    const parsed = JSON.parse(s);
    const originalToConverted =
      parsed.originalToConverted ?? parsed.original_to_converted ?? {};
    const convertedToOriginal =
      parsed.convertedToOriginal ?? parsed.converted_to_original ?? {};
    return { originalToConverted, convertedToOriginal } as LinksIndex;
  } catch {
    return { originalToConverted: {}, convertedToOriginal: {} };
  }
}

export async function saveLinks(
  mediaRootAbs: string,
  idx: LinksIndex,
): Promise<void> {
  const p = linksFilePath(mediaRootAbs);
  const data = JSON.stringify(
    {
      original_to_converted: idx.originalToConverted,
      converted_to_original: idx.convertedToOriginal,
    },
    null,
    2,
  );
  await fsp.writeFile(p, data, "utf8");
}

export async function insertLink(
  mediaRootAbs: string,
  originalName: string,
  convertedName: string,
): Promise<void> {
  const idx = await loadLinks(mediaRootAbs);
  idx.originalToConverted[originalName] = convertedName;
  idx.convertedToOriginal[convertedName] = originalName;
  await saveLinks(mediaRootAbs, idx);
}

export async function updateLinkOnRename(
  mediaRootAbs: string,
  oldConverted: string,
  newConverted: string,
  oldOriginal: string,
  newOriginal: string,
): Promise<void> {
  const idx = await loadLinks(mediaRootAbs);
  delete idx.convertedToOriginal[oldConverted];
  const origRef = Object.entries(idx.originalToConverted).find(
    ([, v]) => v === oldConverted,
  )?.[0];
  if (origRef) delete idx.originalToConverted[origRef];
  delete idx.originalToConverted[oldOriginal];
  const convRef = idx.originalToConverted[oldOriginal];
  if (convRef) delete idx.convertedToOriginal[convRef];
  idx.originalToConverted[newOriginal] = newConverted;
  idx.convertedToOriginal[newConverted] = newOriginal;
  await saveLinks(mediaRootAbs, idx);
}

export async function removeLinkByConverted(
  mediaRootAbs: string,
  convertedName: string,
): Promise<string | undefined> {
  const idx = await loadLinks(mediaRootAbs);
  const orig = idx.convertedToOriginal[convertedName];
  delete idx.convertedToOriginal[convertedName];
  if (orig) delete idx.originalToConverted[orig];
  await saveLinks(mediaRootAbs, idx);
  return orig;
}

export function ensureUniqueNameSync(
  dirAbs: string,
  desiredName: string,
): string {
  const desiredPath = join(dirAbs, desiredName);
  if (!fs.existsSync(desiredPath)) return desiredName;
  const dot = desiredName.lastIndexOf(".");
  const base = dot >= 0 ? desiredName.slice(0, dot) : desiredName;
  const ext = dot >= 0 ? desiredName.slice(dot) : "";
  let i = 1;
  while (true) {
    const candidate = `${base} (${i})${ext}`;
    if (!fs.existsSync(join(dirAbs, candidate))) return candidate;
    i += 1;
  }
}

export function findFileByStemSync(
  dirAbs: string,
  stem: string,
): string | undefined {
  try {
    const entries = fs.readdirSync(dirAbs, { withFileTypes: true });
    for (const e of entries) {
      if (!e.isFile()) continue;
      const name = e.name;
      const dot = name.lastIndexOf(".");
      const s = dot >= 0 ? name.slice(0, dot) : name;
      if (s === stem) return name;
    }
  } catch {}
  return undefined;
}

export async function createSymlinkWithFallback(
  srcAbs: string,
  dstAbs: string,
): Promise<void> {
  try {
    await fsp.rm(dstAbs, { force: true });
    await fsp.symlink(srcAbs, dstAbs);
  } catch {
    await fsp.copyFile(srcAbs, dstAbs);
  }
}

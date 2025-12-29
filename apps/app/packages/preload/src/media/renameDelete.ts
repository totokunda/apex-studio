import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { join } from "node:path";
import { symlinksDir, proxyDir } from "./paths.js";

export async function renameMediaPairInRoot(
  mediaRootAbs: string,
  convertedOldName: string,
  convertedNewName: string,
): Promise<void> {
  const symlinksPath = symlinksDir(mediaRootAbs);
  const oldSymlinkAbs = join(symlinksPath, convertedOldName);
  const newSymlinkAbs = join(symlinksPath, convertedNewName);

  if (oldSymlinkAbs !== newSymlinkAbs && fs.existsSync(oldSymlinkAbs)) {
    await fsp.rename(oldSymlinkAbs, newSymlinkAbs);
  }

  // Update proxy index if the file has a proxy
  try {
    const proxyIndexPath = join(mediaRootAbs, ".proxy-index.json");
    if (fs.existsSync(proxyIndexPath)) {
      const content = await fsp.readFile(proxyIndexPath, "utf8");
      const proxyIndex = JSON.parse(content);

      if (proxyIndex[convertedOldName]) {
        proxyIndex[convertedNewName] = proxyIndex[convertedOldName];
        delete proxyIndex[convertedOldName];
        await fsp.writeFile(
          proxyIndexPath,
          JSON.stringify(proxyIndex, null, 2),
          "utf8",
        );
      }
    }
  } catch (e) {
    console.error("Error updating proxy index on rename:", e);
  }
}

export async function deleteMediaPairInRoot(
  mediaRootAbs: string,
  convertedName: string,
): Promise<void> {
  const symlinksPath = symlinksDir(mediaRootAbs);
  const symlinkAbs = join(symlinksPath, convertedName);

  if (fs.existsSync(symlinkAbs)) {
    await fsp.rm(symlinkAbs, { force: true });
  }

  // Also delete proxy if it exists
  try {
    const proxyIndexPath = join(mediaRootAbs, ".proxy-index.json");
    if (fs.existsSync(proxyIndexPath)) {
      const content = await fsp.readFile(proxyIndexPath, "utf8");
      const proxyIndex = JSON.parse(content);

      if (proxyIndex[convertedName]) {
        const proxyPath = join(
          proxyDir(mediaRootAbs),
          proxyIndex[convertedName],
        );
        if (fs.existsSync(proxyPath)) {
          await fsp.rm(proxyPath, { force: true });
        }
        delete proxyIndex[convertedName];
        await fsp.writeFile(
          proxyIndexPath,
          JSON.stringify(proxyIndex, null, 2),
          "utf8",
        );
      }
    }
  } catch (e) {
    console.error("Error cleaning up proxy:", e);
  }
}

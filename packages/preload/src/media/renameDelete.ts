import fs from 'node:fs';
import {promises as fsp} from 'node:fs';
import {join} from 'node:path';
import {converted24Dir, originalDir as originalDirFromRoot} from './paths.js';
import {ensureUniqueNameSync, findFileByStemSync, loadLinks, removeLinkByConverted, updateLinkOnRename} from './links.js';

export async function renameMediaPairInRoot(mediaRootAbs: string, convertedOldName: string, convertedNewName: string): Promise<void> {
  const convDir = converted24Dir(mediaRootAbs);
  const origDir = originalDirFromRoot(mediaRootAbs);

  const oldConvAbs = join(convDir, convertedOldName);
  const newConvAbs = join(convDir, convertedNewName);
  if (oldConvAbs !== newConvAbs) {
    await fsp.rename(oldConvAbs, newConvAbs);
  }

  const idx = await loadLinks(mediaRootAbs);
  const originalOldName = idx.convertedToOriginal[convertedOldName] ?? findFileByStemSync(origDir, convertedOldName.replace(/\.[^.]+$/, ''));
  if (originalOldName) {
    const newBase = convertedNewName.replace(/\.[^.]+$/, '');
    const ext = originalOldName.includes('.') ? originalOldName.slice(originalOldName.lastIndexOf('.')) : '';
    const desired = `${newBase}${ext}`;
    const finalOriginalNew = ensureUniqueNameSync(origDir, desired);
    const oldOrigAbs = join(origDir, originalOldName);
    const newOrigAbs = join(origDir, finalOriginalNew);
    if (oldOrigAbs !== newOrigAbs) {
      try { await fsp.rename(oldOrigAbs, newOrigAbs); } catch {}
    }
    await updateLinkOnRename(mediaRootAbs, convertedOldName, convertedNewName, originalOldName, finalOriginalNew);
  } else {
    const idx2 = await loadLinks(mediaRootAbs);
    delete idx2.convertedToOriginal[convertedOldName];
    idx2.convertedToOriginal[convertedNewName] = '';
    await fsp.writeFile(join(mediaRootAbs, '.links.json'), JSON.stringify(idx2, null, 2), 'utf8');
  }
}

export async function deleteMediaPairInRoot(mediaRootAbs: string, convertedName: string): Promise<void> {
  const convDir = converted24Dir(mediaRootAbs);
  const origDir = originalDirFromRoot(mediaRootAbs);

  const convAbs = join(convDir, convertedName);
  if (fs.existsSync(convAbs)) {
    await fsp.rm(convAbs, {force: true});
  }

  const origName = await removeLinkByConverted(mediaRootAbs, convertedName);
  if (origName) {
    if (origName.length > 0) {
      const origAbs = join(origDir, origName);
      if (fs.existsSync(origAbs)) {
        try { await fsp.rm(origAbs, {force: true}); } catch {}
      }
    } else {
      const stem = convertedName.replace(/\.[^.]+$/, '');
      const found = findFileByStemSync(origDir, stem);
      if (found) {
        const abs = join(origDir, found);
        try { await fsp.rm(abs, {force: true}); } catch {}
      }
    }
  }
}



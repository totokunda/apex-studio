import { contextBridge } from 'electron';
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

function resolveAddonPath() {
  const nodeName = 'native_decoder.node';
  const moduleDir = dirname(fileURLToPath(import.meta.url));
  const devCandidate = resolve(moduleDir, 'apps', 'app', 'packages', 'native-decoder', 'build', 'Release', nodeName);
  const candidates = [
    devCandidate,
    join(process.resourcesPath ?? '', 'native-decoder', nodeName),
    join(process.resourcesPath ?? '', 'app.asar.unpacked', 'node_modules', '@app', 'native-decoder', 'build', 'Release', nodeName),
  ];
  for (const p of candidates) if (existsSync(p)) return p;
  return null;
}

let addon = null;
function loadAddon() {
  if (addon) return addon;
  const p = resolveAddonPath();
  if (!p) throw new Error('addon not found');
  addon = require(p);
  return addon;
}

contextBridge.exposeInMainWorld('nativeOnlyCreate', () => {
  const a = loadAddon();
  return a.createDecoder();
});

contextBridge.exposeInMainWorld('nativeOnlyDispose', (h) => {
  const a = loadAddon();
  a.dispose(h);
});

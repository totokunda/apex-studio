import { log } from 'console';
import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { BrowserWindow, globalShortcut, App, contentTracing } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import inspector from 'node:inspector';

const session = new inspector.Session();
session.connect();

export async function dumpMainCPU(ms = 8000, app: App) {
  await session.post('Profiler.enable');
  await session.post('Profiler.start');
  setTimeout(() => {
    session.post('Profiler.stop', (err, { profile }) => {
      if (!err) {
        const out = path.join(app.getPath('userData'), `main-${Date.now()}.cpuprofile`);
        fs.writeFileSync(out, JSON.stringify(profile));
        console.log(`Main CPU profile saved: ${out}`);
      }
    });
  }, ms);
}

export async function dumpRendererCPU(win: BrowserWindow, ms = 8000, app: App) {
  const dbg = win.webContents.debugger;
  if (!dbg.isAttached()) dbg.attach('1.3');
  await dbg.sendCommand('Profiler.enable');
  await dbg.sendCommand('Profiler.start');
  setTimeout(async () => {
    const { profile } = await dbg.sendCommand('Profiler.stop');
    const out = path.join(app.getPath('userData'), `renderer-${Date.now()}.cpuprofile`);
    fs.writeFileSync(out, JSON.stringify(profile));
    console.log(`Renderer CPU profile saved: ${out}`);
  }, ms);
}


export async function dumpTrace(ms = 8000, app:App) {
  const outPath = path.join(app.getPath('userData'), `trace-${Date.now()}.json`);

  // Start tracing with all categories enabled
  await contentTracing.startRecording({
    included_categories: ['*'] // You can filter if you only need some categories
  });

  console.log(`Tracing started, will run for ${ms}ms...`);

  setTimeout(async () => {
    const traceFile = await contentTracing.stopRecording();
    // The API returns the file path automatically; we can rename/move it if needed
    if (traceFile !== outPath) {
      fs.copyFileSync(traceFile, outPath);
    }
    console.log(`Trace saved to: ${outPath}`);
  }, ms);
}

export class ApexApi implements AppModule {
  async enable(_context: ModuleContext): Promise<void> {
    const app = _context.app;
    await app.whenReady();
    globalShortcut.register('CommandOrControl+Alt+P', async () => {
      log('Hot dump requested');
      await dumpMainCPU(6000, app);
      const win = BrowserWindow.getFocusedWindow();
      if (win) await dumpRendererCPU(win, 6000, app);
      await dumpTrace(6000, app);
    });
  }
}

export function apexApi(...args: ConstructorParameters<typeof ApexApi>) {
  return new ApexApi(...args);
}
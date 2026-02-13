import { app, BrowserWindow } from 'electron';

const preloadPath = '/Users/tosinkuye/apex-workspace/apex-studio/.tmp-preload-native-only.mjs';

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.webContents.on('render-process-gone', (_e, d) => console.error('render-process-gone', d));

  await win.loadURL('about:blank');
  try {
    const h = await win.webContents.executeJavaScript('window.nativeOnlyCreate()');
    console.log('handle', h);
    await win.webContents.executeJavaScript(`window.nativeOnlyDispose(${Number(h)})`);
    console.log('ok');
    app.exit(0);
  } catch (e) {
    console.error('err', e?.stack || e);
    app.exit(1);
  }
});

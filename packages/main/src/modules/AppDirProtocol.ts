import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { protocol } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import mime from 'mime';


class AppDirProtocol implements AppModule {
  async enable({app}: ModuleContext): Promise<void> {
    protocol.registerSchemesAsPrivileged([{
        scheme: 'app',
        privileges: {
          standard: true,
          secure: true,
          stream: true,
          supportFetchAPI: true, // important for window.fetch
          corsEnabled: true
        }
      }]);
    await app.whenReady();
    protocol.handle('app', (request) => {
        // Map app://user-data/... to a real file under userData
        const u = new URL(request.url);
        if (u.hostname !== 'user-data') {
          return new Response(null, { status: 404 });
        }
        
        const userData = app.getPath('userData');
        const decodedPathname = decodeURIComponent(u.pathname);
        const normalizedPathname = decodedPathname.startsWith('/') ? decodedPathname.slice(1) : decodedPathname;
        const filePath = normalizedPathname.startsWith(userData) 
          ? normalizedPathname 
          : path.join(userData, normalizedPathname);
        // Check if file exists and is readable
        try {
          if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            return new Response(null, { status: 404 });
          }
        } catch {
          return new Response(null, { status: 404 });
        }
        
        const stream = fs.createReadStream(filePath);
        const contentType = mime.getType(filePath) || 'application/octet-stream';
        
        return new Response(stream as any, {
          status: 200,
          headers: { 'Content-Type': contentType }
        });
      });
  }
}

export function appDirProtocol(): AppDirProtocol {
  return new AppDirProtocol();
}
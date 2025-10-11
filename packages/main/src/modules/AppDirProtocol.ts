import {AppModule} from '../AppModule.js';
import {ModuleContext} from '../ModuleContext.js';
import { protocol } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import mime from 'mime';

// Helper to convert Node.js stream to Web ReadableStream with proper error handling
function nodeStreamToWebStream(nodeStream: fs.ReadStream): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      nodeStream.on('data', (chunk: string | Buffer) => {
        try {
          const uint8Array = typeof chunk === 'string' 
            ? new TextEncoder().encode(chunk) 
            : new Uint8Array(chunk);
          controller.enqueue(uint8Array);
        } catch (err) {
          // Stream might be closed, ignore
        }
      });

      nodeStream.on('end', () => {
        try {
          controller.close();
        } catch (err) {
          // Already closed, ignore
        }
      });

      nodeStream.on('error', (err) => {
        try {
          controller.error(err);
        } catch {
          // Already errored, ignore
        }
        nodeStream.destroy();
      });
    },
    cancel() {
      nodeStream.destroy();
    }
  });
}

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
        
        const contentType = mime.getType(filePath) || 'application/octet-stream';
        const fileSize = fs.statSync(filePath).size;
        
        // Handle Range requests
        const rangeHeader = request.headers.get('range');
        if (rangeHeader) {
          const parts = rangeHeader.replace(/bytes=/, '').split('-');
          const start = parseInt(parts[0], 10);
          const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
          const chunkSize = (end - start) + 1;
          
          const nodeStream = fs.createReadStream(filePath, { start, end });
          const webStream = nodeStreamToWebStream(nodeStream);
          
          return new Response(webStream, {
            status: 206,
            headers: {
              'Content-Type': contentType,
              'Content-Length': chunkSize.toString(),
              'Content-Range': `bytes ${start}-${end}/${fileSize}`,
              'Accept-Ranges': 'bytes'
            }
          });
        }
        
        // Full content response
        const nodeStream = fs.createReadStream(filePath);
        const webStream = nodeStreamToWebStream(nodeStream);
        
        return new Response(webStream, {
          status: 200,
          headers: {
            'Content-Type': contentType,
            'Content-Length': fileSize.toString(),
            'Accept-Ranges': 'bytes'
          }
        });
      });
  }
}

export function appDirProtocol(): AppDirProtocol {
  return new AppDirProtocol();
}
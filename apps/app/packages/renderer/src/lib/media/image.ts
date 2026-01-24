import { InputImageTrack } from "../types";
import { MediaInfo } from "../types";
import { WrappedCanvas } from "mediabunny";
import { buildImageKey } from "./cache";
import { getCachedImage } from "./cache";
import { FramesCache } from "./cache";
import { readFileBuffer } from "@app/preload";

function detectMime(buf: ArrayBuffer): string | undefined {
  const b = new Uint8Array(buf);
  const ascii = (s: string, o = 0) =>
    [...s].every((ch, i) => b[o + i] === ch.charCodeAt(0));

  if (b[0] === 0xff && b[1] === 0xd8) return "image/jpeg";
  if (b[0] === 0x89 && ascii("PNG", 1)) return "image/png";
  if (ascii("GIF87a") || ascii("GIF89a")) return "image/gif";
  if (ascii("RIFF", 0) && ascii("WEBP", 8)) return "image/webp";
  if (ascii("BM")) return "image/bmp";
  if (ascii("\x00\x00\x01\x00")) return "image/x-icon";
  // AVIF: 'ftyp' box with 'avif' brand within first 32 bytes
  if (ascii("ftyp", 4) && (ascii("avif", 8) || ascii("avis", 8)))
    return "image/avif";
  // Very loose SVG sniff (text XML)
  const text = new TextDecoder("utf-8", { fatal: false }).decode(
    b.slice(0, 64),
  );
  if (/\<svg[\s>]/i.test(text)) return "image/svg+xml";
  // TIFF (little/big endian)
  if (ascii("II*\x00") || ascii("MM\x00*")) return "image/tiff";
  return undefined;
}

// Tiny JPEG EXIF orientation reader (fast scan of APP1)
async function readJpegOrientation(
  blob: Blob,
): Promise<InputImageTrack["orientation"]> {
  // Read up to 128 KB (enough for typical APP1)
  const buf = await blob.slice(0, 128 * 1024).arrayBuffer();
  const v = new DataView(buf);
  let off = 2; // skip SOI (FFD8)
  while (off + 4 <= v.byteLength) {
    if (v.getUint8(off) !== 0xff) break;
    const marker = v.getUint8(off + 1);
    off += 2;
    if (marker === 0xd9 || marker === 0xda) break; // EOI / SOS
    const len = v.getUint16(off, false);
    if (marker === 0xe1 && len >= 10) {
      // APP1
      // Check Exif header
      const exif = new Uint8Array(buf, off + 2, 6);
      if (String.fromCharCode(...exif) === "Exif\0\0") {
        const tiffOff = off + 2 + 6;
        const little = v.getUint16(tiffOff, false) === 0x4949; // 'II' or 'MM'
        const get16 = (p: number) => v.getUint16(p, little);
        const get32 = (p: number) => v.getUint32(p, little);
        if (
          get16(tiffOff) === (little ? 0x4949 : 0x4d4d) &&
          get16(tiffOff + 2) === 0x002a
        ) {
          const ifd0 = tiffOff + get32(tiffOff + 4);
          if (ifd0 + 2 <= v.byteLength) {
            const count = get16(ifd0);
            for (let i = 0; i < count; i++) {
              const entry = ifd0 + 2 + i * 12;
              const tag = get16(entry);
              if (tag === 0x0112) {
                // Orientation stored in the value or pointed-to (SHORT, count=1)
                const type = get16(entry + 2);
                const countVal = get32(entry + 4);
                if (type === 3 && countVal === 1) {
                  const val = little
                    ? v.getUint16(entry + 8, true)
                    : v.getUint16(entry + 8, false);
                  if (val >= 1 && val <= 8) return val as any;
                } else {
                  const valOff = tiffOff + get32(entry + 8);
                  const val = get16(valOff);
                  if (val >= 1 && val <= 8) return val as any;
                }
                break;
              }
            }
          }
        }
      }
    }
    off += len;
  }
  return undefined;
}

async function toBlob(
  input: Blob | File | ArrayBuffer | Uint8Array | string,
): Promise<Blob> {
  if (input instanceof Blob) return input;
  if (typeof input === "string") {
    // Requires CORS for cross-origin URLs.
    const res = await readFileBuffer(input);
    return new Blob([res as unknown as ArrayBuffer]);
  }
  if (input instanceof ArrayBuffer) return new Blob([input]);
  if (input instanceof Uint8Array) {
    // Ensure we pass an ArrayBuffer compatible with BlobPart
    let arrayBuffer: ArrayBuffer;
    if (input.buffer instanceof ArrayBuffer) {
      // Slice to the exact view window
      arrayBuffer = input.buffer.slice(
        input.byteOffset,
        input.byteOffset + input.byteLength,
      );
    } else {
      // Fallback: copy into a real ArrayBuffer (avoids SharedArrayBuffer)
      arrayBuffer = new ArrayBuffer(input.byteLength);
      new Uint8Array(arrayBuffer).set(input);
    }
    return new Blob([arrayBuffer]);
  }
  throw new TypeError("Unsupported input type");
}

export async function readImageMetadataFast(
  input: Blob | File | ArrayBuffer | Uint8Array | string,
): Promise<InputImageTrack> {

  const blob = await toBlob(input);
  const size = blob.size;
  const head = await blob.slice(0, 512).arrayBuffer();
  const mime = detectMime(head) || blob.type || undefined;

  // Prefer ImageDecoder if available (Chrome/Edge/Firefox, Safari 17+)
  if (typeof (globalThis as any).ImageDecoder === "function" && mime) {
    const arrayBuffer = await blob.arrayBuffer();

    const decoder = new ImageDecoder({ data: arrayBuffer, type: mime });
    await decoder.tracks.ready;
    const track = decoder.tracks.selectedTrack;
    const result = await decoder.decode({
      frameIndex: 0,
      completeFramesOnly: true,
    });
    const image = result.image;
    const width = image.displayWidth;
    const height = image.displayHeight;
    image.close();
    decoder.close?.();
    const meta: InputImageTrack = {
      width,
      height,
      mime,
      size,
      animated: !!track?.animated,
      input:input
    };

    if (mime === "image/jpeg") {
      meta.orientation = await readJpegOrientation(blob);
    }
    return meta;
  }

  if (typeof createImageBitmap === "function") {
    const bmp = await createImageBitmap(blob);
    const meta: InputImageTrack = {
      width: bmp.width,
      height: bmp.height,
      mime,
      size,
      input: blob,
    };
    bmp.close();
    if (mime === "image/jpeg")
      meta.orientation = await readJpegOrientation(blob);
    return meta;
  }

  // Final fallback: HTMLImageElement
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const meta: InputImageTrack = {
      width: img.naturalWidth,
      height: img.naturalHeight,
      mime,
      size,
      input
    }
    if (mime === "image/jpeg")
      meta.orientation = await readJpegOrientation(blob);
    return meta;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export const fetchImage = async (
  path: string,
  width?: number,
  height?: number,
  options?: { mediaInfo?: MediaInfo },
): Promise<WrappedCanvas> => {
  width = width || (options?.mediaInfo?.image?.width as number);
  height = height || (options?.mediaInfo?.image?.height as number);

  const imageKey = buildImageKey(path, width, height);

  const cachedImage = getCachedImage(path, width, height);
  if (cachedImage) return cachedImage as WrappedCanvas;

  const blob = await toBlob(path);
  const url = URL.createObjectURL(blob);
  const img = new Image(width, height);
  img.src = url;
  img.decoding = "async";

  await img.decode();

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  if (ctx) {
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
  }

  URL.revokeObjectURL(url);

  const imageCache = FramesCache.getState();

  const contentToCache = {
    canvas: canvas,
    duration: 1,
    timestamp: 0,
  };

  imageCache.put(imageKey, contentToCache);

  return contentToCache;
};

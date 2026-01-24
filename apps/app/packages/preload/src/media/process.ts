import fs from "node:fs";
import { promises as fsp } from "node:fs";
import { join } from "node:path";
import {
  getLowercaseExtension,
  isAudioExt,
  isImageExt,
  isVideoExt,
} from "./fileExts.js";
import { converted24Dir, resolveMediaRootFromOriginalFile } from "./paths.js";
import {
  createSymlinkWithFallback,
  ensureUniqueNameSync,
  insertLink,
} from "./links.js";
import {
  ffprobeAudioSampleRate,
  ffprobeDimensions,
  ffprobeHasAudio,
} from "./probe.js";
import {
  convertAudioToWav48k,
  convertImageScale,
  convertVideoTo24Fps,
} from "./ffmpeg.js";

export type ProcessMediaOptions = {
  inputAbsPath: string;
  resolution?: string;
};

function parseTargetHeight(resolution?: string): number {
  const s = (resolution ?? "480p").trim().toLowerCase();
  if (s === "hd" || s === "720" || s === "720p") return 720;
  if (s === "1080" || s === "1080p" || s === "fullhd" || s === "full_hd")
    return 1080;
  const raw = s.endsWith("p") ? s.slice(0, -1) : s;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 480;
}

export async function processMediaTo24(
  options: ProcessMediaOptions,
): Promise<string> {
  const inputAbs = options.inputAbsPath;
  if (!fs.existsSync(inputAbs)) throw new Error("Input path does not exist");
  const targetH = parseTargetHeight(options.resolution);

  const fileName = inputAbs.split(/[/\\]/g).pop();
  if (!fileName) throw new Error("Invalid input filename");

  const mediaRoot = resolveMediaRootFromOriginalFile(inputAbs);
  const convertedDir = converted24Dir(mediaRoot);
  if (!fs.existsSync(convertedDir))
    fs.mkdirSync(convertedDir, { recursive: true });

  let outputName = fileName;
  let outputAbs = join(convertedDir, outputName);
  if (fs.existsSync(outputAbs)) {
    outputName = ensureUniqueNameSync(convertedDir, outputName);
    outputAbs = join(convertedDir, outputName);
  }
  const ext = getLowercaseExtension(fileName);

  if (isImageExt(ext)) {
    const dims = await ffprobeDimensions(inputAbs);
    const needResize = dims ? dims.height > targetH : true;
    if (!needResize) {
      await createSymlinkWithFallback(inputAbs, outputAbs);
      await insertLink(mediaRoot, fileName, outputName);
      return outputAbs;
    }
    await convertImageScale(inputAbs, outputAbs, targetH);
    await insertLink(mediaRoot, fileName, outputName);
    return outputAbs;
  }

  if (isAudioExt(ext)) {
    const sr = await ffprobeAudioSampleRate(inputAbs);
    if (ext === "wav" && sr === 48000) {
      await createSymlinkWithFallback(inputAbs, outputAbs);
      await insertLink(mediaRoot, fileName, outputName);
      return outputAbs;
    }
    const dot = fileName.lastIndexOf(".");
    const base = dot >= 0 ? fileName.slice(0, dot) : fileName;
    const desired = `${base}.wav`;
    const finalName = ensureUniqueNameSync(convertedDir, desired);
    const finalAbs = join(convertedDir, finalName);
    await convertAudioToWav48k(inputAbs, finalAbs);
    await insertLink(mediaRoot, fileName, finalName);
    return finalAbs;
  }

  if (isVideoExt(ext)) {
    const hasAudio = await ffprobeHasAudio(inputAbs);
    await convertVideoTo24Fps(inputAbs, outputAbs, targetH, hasAudio);
    await insertLink(mediaRoot, fileName, outputName);
    return outputAbs;
  }

  await fsp.copyFile(inputAbs, outputAbs);
  await insertLink(mediaRoot, fileName, outputName);
  return outputAbs;
}

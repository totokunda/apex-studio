import { spawn } from "node:child_process";
import { resolveFfmpegCommand } from "./ffmpegBin.js";

function run(
  cmd: string,
  args: string[],
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  return new Promise((resolve) => {
    const resolved =
      cmd === "ffmpeg" || cmd === "ffprobe"
        ? resolveFfmpegCommand(cmd)
        : cmd;
    const ps = spawn(resolved, args);
    let out = "";
    let err = "";
    ps.stdout.setEncoding("utf8");
    ps.stderr.setEncoding("utf8");
    ps.stdout.on("data", (d) => {
      out += String(d);
    });
    ps.stderr.on("data", (d) => {
      err += String(d);
    });
    ps.on("close", (code) => resolve({ stdout: out, stderr: err, code }));
    ps.on("error", () =>
      resolve({ stdout: "", stderr: "spawn error", code: 1 }),
    );
  });
}

export function buildScaleExprEscaped(targetH: number): string {
  const core = `if(gt(ih,${targetH}),-2,iw):if(gt(ih,${targetH}),${targetH},ih)`;
  return core.replace(/,/g, "\\,");
}

export async function convertImageScale(
  inputAbs: string,
  outputAbs: string,
  targetH: number,
): Promise<void> {
  const scaleFilter = `scale=${buildScaleExprEscaped(targetH)}`;
  const args = ["-y", "-i", inputAbs, "-vf", scaleFilter, outputAbs];
  const res = await run("ffmpeg", args);
  if (res.code !== 0)
    throw new Error(`ffmpeg image scaling failed: ${res.stderr}`);
}

export async function convertAudioToWav48k(
  inputAbs: string,
  outputAbs: string,
): Promise<void> {
  const args = [
    "-y",
    "-i",
    inputAbs,
    "-vn",
    "-c:a",
    "pcm_s16le",
    "-ar",
    "48000",
    outputAbs,
  ];
  const res = await run("ffmpeg", args);
  if (res.code !== 0)
    throw new Error(`ffmpeg audio resample failed: ${res.stderr}`);
}

export async function convertVideoTo24Fps(
  inputAbs: string,
  outputAbs: string,
  targetH: number,
  hasAudio: boolean,
): Promise<void> {
  const vf = `fps=24,scale=${buildScaleExprEscaped(targetH)}`;
  const args = [
    "-y",
    "-i",
    inputAbs,
    "-r",
    "24",
    "-vf",
    vf,
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
  ];
  if (hasAudio) {
    args.push(
      "-c:a",
      "aac",
      "-profile:a",
      "aac_low",
      "-b:a",
      "192k",
      "-ar",
      "48000",
      "-ac",
      "2",
    );
  } else {
    args.push("-an");
  }
  args.push(outputAbs);
  const res = await run("ffmpeg", args);
  if (res.code !== 0)
    throw new Error(`ffmpeg video convert failed: ${res.stderr}`);
}

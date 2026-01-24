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

export async function ffprobeDimensions(
  absPath: string,
): Promise<{ width: number; height: number } | undefined> {
  const args = [
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height",
    "-of",
    "csv=s=x:p=0",
    absPath,
  ];
  const res = await run("ffprobe", args);
  if (res.code !== 0) return undefined;
  const s = res.stdout.trim();
  const [wStr, hStr] = s.split("x");
  const width = Number(wStr);
  const height = Number(hStr);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return undefined;
  return { width, height };
}

export async function ffprobeHasAudio(absPath: string): Promise<boolean> {
  const args = [
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=index",
    "-of",
    "csv=p=0",
    absPath,
  ];
  const res = await run("ffprobe", args);
  if (res.code !== 0) return false;
  return res.stdout.trim().length > 0;
}

export async function ffprobeAudioSampleRate(
  absPath: string,
): Promise<number | undefined> {
  const args = [
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    "stream=sample_rate",
    "-of",
    "csv=p=0",
    absPath,
  ];
  const res = await run("ffprobe", args);
  if (res.code !== 0) return undefined;
  const num = Number(res.stdout.trim());
  return Number.isFinite(num) ? num : undefined;
}

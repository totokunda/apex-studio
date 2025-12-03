import { v4 as uuidv4 } from "uuid";
import { getMediaInfoCached } from "@/lib/media/utils";
import { toFrameRange } from "@/lib/media/fps";
import type { AnyClipProps, Asset, ImageClipProps, VideoClipProps } from "@/lib/types";
import { runPreprocessor } from "@/lib/preprocessor/api";

export interface PreprocessorRunContext {
  selectedPreprocessorId: string | null;
  fps: number;
  getPreprocessorById: (id: string) => any;
  getClipFromPreprocessorId: (id: string) => AnyClipProps | null;
  updatePreprocessor: (
    clipId: string,
    preprocessorId: string,
    patch: any,
  ) => void;
  clearJob: (jobId: string) => void;
  toast: {
    info: (msg: string) => void;
    success: (msg: string) => void;
    error: (msg: string) => void;
  };
  setIsPreparingPreprocessor: (v: boolean) => void;
  getAssetById: (id: string) => Asset | undefined;
}

export const runPreprocessorJob = async (ctx: PreprocessorRunContext) => {
  const {
    selectedPreprocessorId,
    fps,
    getPreprocessorById,
    getClipFromPreprocessorId,
    updatePreprocessor,
    getAssetById,
    clearJob,
    toast,
    setIsPreparingPreprocessor,
  } = ctx;

  if (!selectedPreprocessorId) return;
  const preprocessor = getPreprocessorById(selectedPreprocessorId);
  if (!preprocessor) return;
  const clip = getClipFromPreprocessorId(selectedPreprocessorId) as VideoClipProps | ImageClipProps;
  const asset = getAssetById(clip.assetId);
  if (!asset || !asset.path) return;

  if (preprocessor?.activeJobId) {
    clearJob(preprocessor.activeJobId);
  }

  setIsPreparingPreprocessor(true);
  try {
    const { getBackendIsRemote, getFileShouldUpload } =
      await import("@app/preload");
    const remoteRes = await getBackendIsRemote();
    const isRemote = !!(
      remoteRes &&
      remoteRes.success &&
      remoteRes.data?.isRemote
    );
    if (isRemote) {
      const su = await getFileShouldUpload(String(asset.path || ""));
      const shouldUpload = !!(su && su.success && su.data?.shouldUpload);
      if (shouldUpload) {
        toast.info("Uploading source media to server…");
      }
    }
  } catch {}

  const clipMediaInfo = getMediaInfoCached(asset.path);
  const clipFps = clipMediaInfo?.stats.video?.averagePacketRate ?? 24;
  if (
    preprocessor.startFrame === undefined ||
    preprocessor.endFrame === undefined
  ) {
    setIsPreparingPreprocessor(false);
    return;
  }
  let { start: startFrameReal, end: endFrameReal } = toFrameRange(
    preprocessor.startFrame,
    preprocessor.endFrame,
    fps,
    clipFps,
    clipMediaInfo?.duration ?? 0,
  );

  const clipMediaStartFrame = Math.round(
    ((clipMediaInfo?.startFrame ?? 0) / fps) * clipFps,
  );

  startFrameReal += clipMediaStartFrame;
  endFrameReal += clipMediaStartFrame;

  const activeJobId = uuidv4();
  try {
    updatePreprocessor(clip.clipId, preprocessor.id, {
      status: "running",
      activeJobId,
      jobIds: [...(preprocessor.jobIds || []), activeJobId],
    });
  } catch {}

  const response = await runPreprocessor({
    start_frame: startFrameReal,
    end_frame: endFrameReal,
    preprocessor_name: preprocessor.preprocessor.id,
    input_path: asset.path,
    job_id: activeJobId,
    download_if_needed: true,
    params: preprocessor.values,
  });
  setIsPreparingPreprocessor(false);
  if (response.success) {
    toast.success(
      `Preprocessor ${preprocessor.preprocessor.name} run started successfully`,
    );
    updatePreprocessor(clip.clipId, preprocessor.id, { status: "running" });
  } else {
    toast.error(`Failed to run preprocessor ${preprocessor.preprocessor.name}`);
  }
};

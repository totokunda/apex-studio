import { useManifestStore } from "@/lib/manifest/store";
import React from "react";
import { LuChevronLeft, LuPlus } from "react-icons/lu";
import { ScrollArea } from "../ui/scroll-area";
import ComponentCard from "./ComponentCard";
import LoRACard from "./LoRACard";
import { useControlsStore } from "@/lib/control";
import {
  useClipStore,
  getTimelineHeightForClip,
  getTimelineTypeForClip,
  isValidTimelineForClip,
} from "@/lib/clip";
import { v4 as uuidv4 } from "uuid";

interface ModelPageProps {
  manifestId: string;
}

const ModelPage: React.FC<ModelPageProps> = ({ manifestId }) => {
  const { clearSelectedManifestId, getLoadedManifest } = useManifestStore();
  const manifest = getLoadedManifest(manifestId);
  if (!manifest) return null;

  const isVideoDemo = React.useMemo(() => {
    const value = (manifest.metadata?.demo_path || "").toLowerCase();
    try {
      const url = new URL(value);
      const pathname = url.pathname;
      const ext = pathname.split(".").pop() || "";
      return ["mp4", "webm", "mov", "m4v", "ogg", "m3u8"].includes(ext);
    } catch {
      return (
        value.endsWith(".mp4") ||
        value.endsWith(".webm") ||
        value.endsWith(".mov") ||
        value.endsWith(".m4v") ||
        value.endsWith(".ogg") ||
        value.endsWith(".m3u8")
      );
    }
  }, [manifest.metadata?.demo_path]);

  const components = manifest?.spec?.components || [];

  return (
    <div className="flex flex-col h-full w-full">
      <ScrollArea className="flex-1">
        <div className="p-7 pt-3 pb-28">
          <div className="flex items-center gap-x-2">
            <button
              onClick={async () => {
                clearSelectedManifestId();
              }}
              className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded transition-colors cursor-pointer"
            >
              <LuChevronLeft className="w-3 h-3" />
            </button>
            <span className="text-brand-light/90 text-[11px] font-medium">
              Back
            </span>
          </div>
          <div className="mt-4 flex flex-row gap-x-4 w-full">
            <div className="rounded-md overflow-hidden flex items-center w-44 aspect-square justify-start flex-shrink-0">
              {isVideoDemo ? (
                <video
                  src={manifest.demo_path}
                  className="h-full w-full object-cover rounded-md"
                  autoPlay
                  muted
                  loop
                  playsInline
                />
              ) : (
                <img
                  src={manifest.metadata.demo_path}
                  alt={manifest.metadata.name}
                  className="h-full object-cover rounded-md"
                />
              )}
            </div>

            <div className="flex flex-col gap-y-1 w-full justify-start">
              <h2 className="text-brand-light text-[18px] font-semibold text-start">
                {manifest.metadata.name}
              </h2>
              <p className="text-brand-light/90 text-[12px] text-start">
                {manifest.metadata.description}
              </p>

              <div className="flex flex-col mt-1 items-start gap-y-0.5">
                <span className="text-brand-light text-[12px] font-medium">
                  {manifest.metadata.license}
                </span>
                <span className="text-brand-light/80 text-[11px]">
                  {manifest.metadata.author}
                </span>
              </div>

              <div className="flex flex-row items-center gap-x-1.5 mt-2">
                {manifest.metadata?.tags?.map((tag) => (
                  <span
                    key={tag}
                    className="text-brand-light text-[11px] bg-brand border shadow border-brand-light/10 rounded px-2 py-0.5"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
          {manifest.downloaded ? (
            <div className="mt-5 ">
              <button
                type="button"
                className="text-[11px] font-medium w-full flex items-center transition-all duration-200 justify-center gap-x-1.5 rounded-[6px] px-12 py-2 flex-shrink-0 text-brand-light hover:text-brand-light/90 bg-brand-accent-two-shade hover:bg-brand-accent-two-shade/90"
                title="Add clip at playhead"
                onClick={() => {
                  try {
                    const controls = useControlsStore.getState();
                    const clipStore = useClipStore.getState();
                    const fps = Math.max(1, controls.fps || 1);
                    const focusFrame = Math.max(0, controls.focusFrame || 0);
                    const desiredFrames = Math.max(
                      1,
                      manifest.metadata?.desired_duration ?? controls.defaultClipLength * fps,
                    );
                    const startFrame = focusFrame;
                    const endFrame = startFrame + desiredFrames;

                    // Choose an existing compatible timeline with free space
                    const mediaTimelines = clipStore.timelines.filter((t) =>
                      isValidTimelineForClip(t, { type: "model" } as any),
                    );
                    const intervalOverlaps = (
                      loA: number,
                      hiA: number,
                      loB: number,
                      hiB: number,
                    ) => loA < hiB && hiA > loB;
                    let targetTimelineId: string | undefined;
                    for (const t of mediaTimelines) {
                      const existing = clipStore
                        .getClipsForTimeline(t.timelineId)
                        .map((c) => ({
                          lo: c.startFrame || 0,
                          hi: c.endFrame || 0,
                        }))
                        .filter((iv) => iv.hi > iv.lo);
                      const hasConflict = existing.some((iv) =>
                        intervalOverlaps(startFrame, endFrame, iv.lo, iv.hi),
                      );
                      if (!hasConflict) {
                        targetTimelineId = t.timelineId;
                        break;
                      }
                    }
                    // If no space found, create a new timeline
                    if (!targetTimelineId) {
                      const timelineId = uuidv4();
                      const last =
                        clipStore.timelines[clipStore.timelines.length - 1];
                      clipStore.addTimeline({
                        timelineId,
                        type: getTimelineTypeForClip("model"),
                        timelineHeight: getTimelineHeightForClip("model"),
                        timelineWidth: last?.timelineWidth ?? 0,
                        timelineY:
                          (last?.timelineY ?? 0) + (last?.timelineHeight ?? 54),
                        timelinePadding: last?.timelinePadding ?? 24,
                        muted: false,
                        hidden: false,
                      });
                      targetTimelineId = timelineId;
                    }
                    // Build and add clip
                    const newClipId = uuidv4();
                    const clipBase: any = {
                      timelineId: targetTimelineId,
                      clipId: newClipId,
                      startFrame,
                      endFrame,
                      // @ts-ignore
                      type: "model",
                      trimEnd: -Infinity,
                      trimStart: Infinity,
                      speed: 1.0,
                      manifest: manifest,
                    };
                    useClipStore.getState().addClip(clipBase);
                  } catch {}
                }}
              >
                <LuPlus className="w-4 h-4" />
                <span className="">Add Clip</span>
              </button>
            </div>
          ) : null}
          <div className="mt-5 ">
            <div className="flex items-center justify-between">
              <h3 className="text-brand-light text-[13.5px] font-semibold text-start">
                Model Architecture
              </h3>
            </div>

            <div className="space-y-2 mt-3.5">
              {components.map((component, index) => (
                <ComponentCard
                  key={index}
                  component={component}
                  manifestId={manifestId}
                  index={index}
                />
              ))}
              {components.length === 0 && (
                <div className="text-brand-light/60 text-[12px] text-center py-8">
                  No components available
                </div>
              )}
            </div>
          </div>
          {manifest.spec.loras && manifest.spec.loras.length > 0 && (
            <div className="mt-6">
              <div className="flex items-center justify-between">
                <h3 className="text-brand-light text-[13.5px] font-semibold text-start">
                  LoRAs
                </h3>
              </div>
              <div className="space-y-2 mt-3.5">
                {manifest.spec.loras.map((l, idx) => (
                  <LoRACard key={idx} item={l} manifestId={manifestId} />
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ModelPage;

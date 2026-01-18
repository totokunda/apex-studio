import { useCallback, useEffect } from "react";
import type { UIInput } from "@/lib/manifest/api";

type GetInputById = (id: string) => UIInput | undefined;

type UpdateModelInput = (
  clipId: string,
  inputId: string,
  patch: { value: string },
) => void;

export function useResolutionAspectSync(opts: {
  clipId: string;
  inputs: UIInput[];
  getInputById: GetInputById;
  updateModelInput: UpdateModelInput;
}) {
  const { clipId, inputs, getInputById, updateModelInput } = opts;

  // Helpers to support dependent updates (resolution/aspect_ratio -> height/width)
  const getSelectedOption = useCallback(
    (input: UIInput | undefined, valueOverride?: string) => {
      const opts: any[] = (input as any)?.options || [];
      const val =
        valueOverride ??
        String((input as any)?.value ?? (input as any)?.default ?? "");
      return opts.find((o) => String(o?.value) === String(val));
    },
    [],
  );

  const snapToStep = useCallback(
    (num: number, step?: number, min?: number, max?: number) => {
      if (!Number.isFinite(num)) return num;
      let out = num;
      if (step && step > 0) {
        out = Math.round(out / step) * step;
      }
      if (Number.isFinite(min)) out = Math.max(min as number, out);
      if (Number.isFinite(max)) out = Math.min(max as number, out);
      return out;
    },
    [],
  );

  const computeDimsFromResolutionAndAR = useCallback(
    (resBaseline: number, aspectVal: string) => {
      const heightInput = getInputById("height") as any;
      const widthInput = getInputById("width") as any;
      const arInput = getInputById("aspect_ratio") as any;
      if (!heightInput || !widthInput) return null;

      // If AR is custom, respect manual sizing (auto-sync should not run in that mode)
      if (String(aspectVal) === "custom") return null;

      const arOpt: any = getSelectedOption(arInput, aspectVal);
      const rw = Number(arOpt?.ratio_w ?? 16) || 16;
      const rh = Number(arOpt?.ratio_h ?? 9) || 9;
      const aspect = rw / rh;

      // Special case: SD (512) should keep the *longer* side fixed at 512.
      // This avoids width/height max-clamps forcing the UI back to 512x512.
      const useLongerSideBaseline = Math.round(resBaseline) === 512;

      let targetW: number;
      let targetH: number;

      if (useLongerSideBaseline) {
        // Longer side = baseline
        if (aspect >= 1) {
          // Landscape: width longer
          targetW = resBaseline;
          targetH = resBaseline / aspect;
        } else {
          // Portrait: height longer
          targetH = resBaseline;
          targetW = resBaseline * aspect;
        }
      } else {
        // Default behavior: baseline is for the *shorter* side
        if (aspect >= 1) {
          // Landscape: height shorter
          targetH = resBaseline;
          targetW = resBaseline * aspect;
        } else {
          // Portrait: width shorter
          targetW = resBaseline;
          targetH = resBaseline / aspect;
        }
      }

      const snappedH = snapToStep(
        targetH,
        heightInput?.step,
        heightInput?.min,
        heightInput?.max,
      );
      const snappedW = snapToStep(
        targetW,
        widthInput?.step,
        widthInput?.min,
        widthInput?.max,
      );

      return {
        height: Math.max(1, Math.round(snappedH)),
        width: Math.max(1, Math.round(snappedW)),
      };
    },
    [getInputById, getSelectedOption, snapToStep],
  );

  const handleResolutionChange = useCallback(
    (newVal: string) => {
      const resInput = getInputById("resolution") as any;
      const heightInput = getInputById("height") as any;
      const widthInput = getInputById("width") as any;
      if (!heightInput || !widthInput) return;
      const selected = getSelectedOption(resInput, newVal) as any;

      // Typically resolution is the target for the *shorter* side, but SD (512)
      // should keep the *longer* side fixed at 512 to avoid max-clamps.
      const resBaselineRaw = Number(
        selected?.value ?? newVal ?? heightInput?.default ?? 1024,
      );

      const resBaseline =
        Number.isFinite(resBaselineRaw) && resBaselineRaw > 0
          ? resBaselineRaw
          : 1024;
      const useLongerSideBaseline = Math.round(resBaseline) === 512;

      // Current dimensions (fall back to defaults)
      const currentHeightRaw = Number(
        heightInput?.value ?? heightInput?.default ?? NaN,
      );
      const currentWidthRaw = Number(
        widthInput?.value ?? widthInput?.default ?? NaN,
      );
      const hasValidCurrentDims =
        Number.isFinite(currentHeightRaw) &&
        currentHeightRaw > 0 &&
        Number.isFinite(currentWidthRaw) &&
        currentWidthRaw > 0;

      let heightInt: number;
      let widthInt: number;

      // Check current AR setting
      const arInput = getInputById("aspect_ratio") as any;
      const currentAR = String(arInput?.value ?? arInput?.default ?? "custom");

      // If we have a standard aspect ratio selected, prioritize recalculating dimensions
      // to match that ratio exactly at the new resolution, rather than scaling old dimensions
      // (which might have drifted or been snapped).
      if (currentAR !== "custom") {
        const dims =
          computeDimsFromResolutionAndAR(resBaseline, currentAR) ??
          (() => {
            const heightSnapped = snapToStep(
              resBaseline,
              heightInput?.step,
              heightInput?.min,
              heightInput?.max,
            );
            const widthSnapped = snapToStep(
              resBaseline,
              widthInput?.step,
              widthInput?.min,
              widthInput?.max,
            );
            return {
              height: Math.max(1, Math.round(heightSnapped)),
              width: Math.max(1, Math.round(widthSnapped)),
            };
          })();
        heightInt = dims.height;
        widthInt = dims.width;
      } else if (hasValidCurrentDims) {
        // Preserve current aspect ratio and scale so that either:
        // - default: the *shorter* side matches the selected resolution baseline
        // - SD (512): the *longer* side matches the selected resolution baseline
        const longer = Math.max(currentWidthRaw, currentHeightRaw);
        const shorter = Math.min(currentWidthRaw, currentHeightRaw);
        if (longer <= 0 || shorter <= 0) return;

        const isHeightLonger = currentHeightRaw >= currentWidthRaw;
        const useHeightAsBaseline = useLongerSideBaseline
          ? isHeightLonger
          : !isHeightLonger;

        const baselineInput = useHeightAsBaseline ? heightInput : widthInput;
        const targetBaseline = snapToStep(
          resBaseline,
          baselineInput?.step,
          baselineInput?.min,
          baselineInput?.max,
        );

        const scaleDenom = useLongerSideBaseline ? longer : shorter;
        const scale = targetBaseline / scaleDenom;

        const targetLonger = longer * scale;
        const targetShorter = shorter * scale;

        const targetHeight = isHeightLonger ? targetLonger : targetShorter;
        const targetWidth = isHeightLonger ? targetShorter : targetLonger;

        const snappedHeight = snapToStep(
          targetHeight,
          heightInput?.step,
          heightInput?.min,
          heightInput?.max,
        );
        const snappedWidth = snapToStep(
          targetWidth,
          widthInput?.step,
          widthInput?.min,
          widthInput?.max,
        );

        heightInt = Math.max(1, Math.round(snappedHeight));
        widthInt = Math.max(1, Math.round(snappedWidth));
      } else {
        // Fallback: derive from aspect ratio selector using resolution baseline
        // Note: if we reached here, currentAR is "custom" but hasValidCurrentDims is false,
        // OR currentAR is missing. We try to use whatever is in the AR input.
        const arVal = String(arInput?.value ?? arInput?.default ?? "");
        const dims =
          computeDimsFromResolutionAndAR(resBaseline, arVal) ??
          (() => {
            const heightSnapped = snapToStep(
              resBaseline,
              heightInput?.step,
              heightInput?.min,
              heightInput?.max,
            );
            const widthSnapped = snapToStep(
              resBaseline,
              widthInput?.step,
              widthInput?.min,
              widthInput?.max,
            );
            return {
              height: Math.max(1, Math.round(heightSnapped)),
              width: Math.max(1, Math.round(widthSnapped)),
            };
          })();
        heightInt = dims.height;
        widthInt = dims.width;
      }

      // Check if the new dimensions match a specific aspect ratio option
      const arOptions: any[] = arInput?.options || [];
      let bestAR = "custom";

      for (const opt of arOptions) {
        if (opt.value === "custom") continue;
        const dims = computeDimsFromResolutionAndAR(resBaseline, opt.value);
        if (dims && dims.width === widthInt && dims.height === heightInt) {
          bestAR = opt.value;
          break;
        }
      }

      // Prevent auto-sync effect from overriding our maintained ratio
      updateModelInput(clipId, "aspect_ratio", { value: bestAR });
      updateModelInput(clipId, "height", { value: String(heightInt) });
      updateModelInput(clipId, "width", { value: String(widthInt) });
    },
    [
      clipId,
      computeDimsFromResolutionAndAR,
      getInputById,
      getSelectedOption,
      snapToStep,
      updateModelInput,
    ],
  );

  const handleAspectRatioChange = useCallback(
    (newVal: string) => {
      const resInput = getInputById("resolution") as any;
      const heightInput = getInputById("height") as any;
      const widthInput = getInputById("width") as any;
      if (!heightInput || !widthInput) return;

      const resOpt = getSelectedOption(resInput);
      const resBaselineRaw = Number(
        (resOpt as any)?.value ??
          resInput?.value ??
          resInput?.default ??
          heightInput?.default ??
          1024,
      );
      const resBaseline =
        Number.isFinite(resBaselineRaw) && resBaselineRaw > 0
          ? resBaselineRaw
          : 1024;

      const dims = computeDimsFromResolutionAndAR(resBaseline, newVal);
      if (!dims) return;

      updateModelInput(clipId, "height", { value: String(dims.height) });
      updateModelInput(clipId, "width", { value: String(dims.width) });

      // Check if new dimensions match a specific resolution
      const resOptions: any[] = resInput?.options || [];
      let bestRes = "custom";

      for (const opt of resOptions) {
        if (opt.value === "custom") continue;
        // We need the numeric baseline for computeDims
        const baseline = Number(opt.value);
        if (!Number.isFinite(baseline)) continue;

        const checkDims = computeDimsFromResolutionAndAR(baseline, newVal);
        if (
          checkDims &&
          checkDims.width === dims.width &&
          checkDims.height === dims.height
        ) {
          bestRes = opt.value;
          break;
        }
      }
      updateModelInput(clipId, "resolution", { value: bestRes });
    },
    [
      clipId,
      computeDimsFromResolutionAndAR,
      getInputById,
      getSelectedOption,
      updateModelInput,
    ],
  );

  // Sync effect: ensure height and width reflect selected resolution and aspect ratio.
  // Default behavior: resolution targets the *shorter* side.
  // SD (512) special-case: resolution targets the *longer* side (keeps longer side at 512).
  useEffect(() => {
    const resInput = getInputById("resolution") as any;
    const arInput = getInputById("aspect_ratio") as any;
    const heightInput = getInputById("height") as any;
    const widthInput = getInputById("width") as any;
    if (!heightInput || !widthInput) return;

    const resVal = String(resInput?.value ?? resInput?.default ?? "");
    const arVal = String(arInput?.value ?? arInput?.default ?? "");
    // If custom is selected, do not auto-sync; manual overrides are in effect
    if (resVal === "custom" || arVal === "custom") return;

    const resOpt = getSelectedOption(resInput);
    const resBaselineRaw = Number(
      (resOpt as any)?.value ??
        resVal ??
        heightInput?.value ??
        heightInput?.default ??
        1024,
    );
    const resBaseline =
      Number.isFinite(resBaselineRaw) && resBaselineRaw > 0
        ? resBaselineRaw
        : 1024;

    const currentHeight = Number(
      heightInput?.value ?? heightInput?.default ?? NaN,
    );
    const currentWidth = Number(
      widthInput?.value ?? widthInput?.default ?? NaN,
    );

    const dims = computeDimsFromResolutionAndAR(resBaseline, arVal);
    if (!dims) return;

    if (Number.isFinite(dims.height) && dims.height !== currentHeight) {
      updateModelInput(clipId, "height", { value: String(dims.height) });
    }
    if (Number.isFinite(dims.width) && dims.width !== currentWidth) {
      updateModelInput(clipId, "width", { value: String(dims.width) });
    }
  }, [
    clipId,
    computeDimsFromResolutionAndAR,
    getInputById,
    getSelectedOption,
    snapToStep,
    updateModelInput,
    inputs,
  ]);

  return {
    handleResolutionChange,
    handleAspectRatioChange,
  };
}

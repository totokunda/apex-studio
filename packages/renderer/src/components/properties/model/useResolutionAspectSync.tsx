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

  const computeWidthFromAR = useCallback(
    (heightVal: number, aspectOverride?: string) => {
      const widthInput = getInputById("width") as any;
      const arInput = getInputById("aspect_ratio") as any;
      const arOpt: any = getSelectedOption(arInput, aspectOverride);
      if ((aspectOverride ?? String(arInput?.value)) === "custom") {
        const currentWidth = Number(
          (widthInput as any)?.value ?? (widthInput as any)?.default ?? NaN,
        );
        return Number.isFinite(currentWidth)
          ? Math.round(currentWidth)
          : Math.round(heightVal);
      }
      const rw = Number(arOpt?.ratio_w ?? 16) || 16;
      const rh = Number(arOpt?.ratio_h ?? 9) || 9;
      const rawWidth = heightVal * (rw / rh);
      const snapped = snapToStep(
        rawWidth,
        widthInput?.step,
        widthInput?.min,
        widthInput?.max,
      );
      return Math.max(1, Math.round(snapped));
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

      // Interpret resolution as the target for the *shorter* side.
      const resBaselineRaw = Number(
        selected?.value ?? newVal ?? heightInput?.default ?? 1024,
      );

      const resBaseline =
        Number.isFinite(resBaselineRaw) && resBaselineRaw > 0
          ? resBaselineRaw
          : 1024;

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

      if (hasValidCurrentDims) {
        // Preserve current aspect ratio and scale so that the *shorter* side
        // matches the selected resolution baseline.
        const longer = Math.max(currentWidthRaw, currentHeightRaw);
        const shorter = Math.min(currentWidthRaw, currentHeightRaw);
        if (longer <= 0 || shorter <= 0) return;

        const scale = resBaseline / shorter;
        const targetShorter = resBaseline;
        const targetLonger = longer * scale;

        const snappedShorter = snapToStep(
          targetShorter,
          heightInput?.step,
          heightInput?.min,
          heightInput?.max,
        );
        const snappedLonger = snapToStep(
          targetLonger,
          widthInput?.step,
          widthInput?.min,
          widthInput?.max,
        );

        const finalShorter = Math.max(1, Math.round(snappedShorter));
        const finalLonger = Math.max(1, Math.round(snappedLonger));

        const isHeightLonger = currentHeightRaw >= currentWidthRaw;

        heightInt = isHeightLonger ? finalLonger : finalShorter;
        widthInt = isHeightLonger ? finalShorter : finalLonger;
      } else {
        // Fallback: derive from aspect ratio selector using resolution baseline
        const heightSnapped = snapToStep(
          resBaseline,
          heightInput?.step,
          heightInput?.min,
          heightInput?.max,
        );
        heightInt = Math.max(1, Math.round(heightSnapped));
        widthInt = computeWidthFromAR(heightInt);
      }

      // Prevent auto-sync effect from overriding our maintained ratio
      updateModelInput(clipId, "aspect_ratio", { value: "custom" });
      updateModelInput(clipId, "height", { value: String(heightInt) });
      updateModelInput(clipId, "width", { value: String(widthInt) });
    },
    [clipId, computeWidthFromAR, getInputById, getSelectedOption, snapToStep, updateModelInput],
  );

  const handleAspectRatioChange = useCallback(
    (newVal: string) => {
      const heightInput = getInputById("height") as any;
      const resInput = getInputById("resolution") as any;
      let heightNow = Number(
        heightInput?.value ?? heightInput?.default ?? 1024,
      );
      if (!Number.isFinite(heightNow)) {
        const resSelected = getSelectedOption(resInput);
        heightNow =
          Number(
            (resSelected as any)?.height ??
              resInput?.value ??
              resInput?.default ??
              1024,
          ) || 1024;
      }
      const heightSnapped = snapToStep(
        heightNow,
        heightInput?.step,
        heightInput?.min,
        heightInput?.max,
      );
      const heightInt = Math.max(1, Math.round(heightSnapped));
      // Ensure height is updated so the UI reflects recalculation alongside width
      updateModelInput(clipId, "height", { value: String(heightInt) });
      const widthInt = computeWidthFromAR(heightInt, newVal);
      updateModelInput(clipId, "width", { value: String(widthInt) });
    },
    [clipId, computeWidthFromAR, getInputById, getSelectedOption, snapToStep, updateModelInput],
  );

  // Sync effect: ensure height and width reflect selected resolution and aspect ratio,
  // scaling by whichever side (height or width) is currently *shorter*.
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

    // If we don't yet have valid explicit dimensions, fall back to
    // aspect-ratio-based computation using the resolution baseline.
    if (!hasValidCurrentDims) {
      const heightSnapped = snapToStep(
        resBaseline,
        heightInput?.step,
        heightInput?.min,
        heightInput?.max,
      );
      const fallbackHeight = Math.max(1, Math.round(heightSnapped));
      const fallbackWidth = computeWidthFromAR(fallbackHeight, arVal);

      const currentHeight = Number(
        heightInput?.value ?? heightInput?.default ?? NaN,
      );
      const currentWidth = Number(
        widthInput?.value ?? widthInput?.default ?? NaN,
      );

      if (
        Number.isFinite(fallbackHeight) &&
        fallbackHeight !== currentHeight
      ) {
        updateModelInput(clipId, "height", { value: String(fallbackHeight) });
      }
      if (Number.isFinite(fallbackWidth) && fallbackWidth !== currentWidth) {
        updateModelInput(clipId, "width", { value: String(fallbackWidth) });
      }
      return;
    }

    // Scale by whichever side is currently shorter so that the shorter side
    // matches the selected resolution baseline.
    const longer = Math.max(currentHeightRaw, currentWidthRaw);
    const shorter = Math.min(currentHeightRaw, currentWidthRaw);
    if (longer <= 0 || shorter <= 0) return;

    const scale = resBaseline / shorter;
    const targetShorter = resBaseline;
    const targetLonger = longer * scale;

    const isHeightLonger = currentHeightRaw >= currentWidthRaw;

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

    const nextHeightInt = Math.max(1, Math.round(snappedHeight));
    const nextWidthInt = Math.max(1, Math.round(snappedWidth));

    const currentHeight = Number(
      heightInput?.value ?? heightInput?.default ?? NaN,
    );
    const currentWidth = Number(
      widthInput?.value ?? widthInput?.default ?? NaN,
    );

    if (Number.isFinite(nextHeightInt) && nextHeightInt !== currentHeight) {
      updateModelInput(clipId, "height", { value: String(nextHeightInt) });
    }
    if (Number.isFinite(nextWidthInt) && nextWidthInt !== currentWidth) {
      updateModelInput(clipId, "width", { value: String(nextWidthInt) });
    }
  }, [
    clipId,
    computeWidthFromAR,
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



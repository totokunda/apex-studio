import React, { useCallback, useEffect, useMemo, useState } from "react";
import { UIPanel, UIInput } from "@/lib/manifest/api";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import TextInput from "./inputs/TextInput";
import { useClipStore } from "@/lib/clip";
import type { IconType } from "react-icons";
import SelectInput from "./inputs/SelectInput";
import NumberInput from "./inputs/NumberInput";
import NumberInputSlider from "./inputs/NumberInputSlider";
import RandomInput from "./inputs/RandomInput";
import BooleanInput from "./inputs/BooleanInput";
import ImageInput from "./inputs/ImageInput";
import VideoInput from "./inputs/VideoInput";
import NumberListInput from "./inputs/NumberListInput";
import AudioInput from "./inputs/AudioInput";
import SchedulerPanel from "./SchedulerPanel";
import AttentionPanel from "./AttentionPanel";
import ImageInputList from "./inputs/ImageInputList";
import TextInputList from "./inputs/TextInputList";
import { useControlsStore } from "@/lib/control";
import { useResolutionAspectSync } from "./useResolutionAspectSync";
import type { ManifestComponent } from "@/lib/manifest/api";
import { getSchedulerComponentKey } from "@/lib/manifest/componentKey";

export const ModelInputsPanel: React.FC<{
  panel: UIPanel;
  inputs: UIInput[];
  clipId: string;
  panelSize: number;
}> = ({ panel, inputs, clipId, panelSize }) => {
  const updateModelInput = useClipStore((s) => s.updateModelInput);
  const getClipById = useClipStore((s) => s.getClipById);
  const [iconEl, setIconEl] = useState<React.ReactNode>(null);
  const { fps } = useControlsStore();

  const getInputById = useCallback(
    (id: string) => {
      return inputs.find((input) => input.id === id) as UIInput;
    },
    [inputs],
  );

  const collapsible = useMemo(() => {
    return panel.collapsible === true;
  }, [panel]);

  const [collapsed, setCollapsed] = useState(() =>
    panel.collapsible === true ? panel.default_open === false : false,
  );

  useEffect(() => {
    let cancelled = false;
    async function loadIcon() {
      if (!panel.icon) {
        setIconEl(null);
        return;
      }
      const [libraryRaw, iconName] = panel.icon.split("/");
      const library = libraryRaw?.toLowerCase();
      const loaders: Record<string, () => Promise<Record<string, IconType>>> = {
        lu: () =>
          import("react-icons/lu") as unknown as Promise<
            Record<string, IconType>
          >,
        tfi: () =>
          import("react-icons/tfi") as unknown as Promise<
            Record<string, IconType>
          >,
        md: () =>
          import("react-icons/md") as unknown as Promise<
            Record<string, IconType>
          >,
        tb: () =>
          import("react-icons/tb") as unknown as Promise<
            Record<string, IconType>
          >,
        fa: () =>
          import("react-icons/fa") as unknown as Promise<
            Record<string, IconType>
          >,
        fa6: () =>
          import("react-icons/fa6") as unknown as Promise<
            Record<string, IconType>
          >,
        ri: () =>
          import("react-icons/ri") as unknown as Promise<
            Record<string, IconType>
          >,
        go: () =>
          import("react-icons/go") as unknown as Promise<
            Record<string, IconType>
          >,
        sl: () =>
          import("react-icons/sl") as unknown as Promise<
            Record<string, IconType>
          >,
        fi: () =>
          import("react-icons/fi") as unknown as Promise<
            Record<string, IconType>
          >,
        cg: () =>
          import("react-icons/cg") as unknown as Promise<
            Record<string, IconType>
          >,
        rx: () =>
          import("react-icons/rx") as unknown as Promise<
            Record<string, IconType>
          >,
        io5: () =>
          import("react-icons/io5") as unknown as Promise<
            Record<string, IconType>
          >,
        bs: () =>
          import("react-icons/bs") as unknown as Promise<
            Record<string, IconType>
          >,
        hi: () =>
          import("react-icons/hi") as unknown as Promise<
            Record<string, IconType>
          >,
        vsc: () =>
          import("react-icons/vsc") as unknown as Promise<
            Record<string, IconType>
          >,
        bi: () =>
          import("react-icons/bi") as unknown as Promise<
            Record<string, IconType>
          >,
        pi: () =>
          import("react-icons/pi") as unknown as Promise<
            Record<string, IconType>
          >,
        io: () =>
          import("react-icons/io") as unknown as Promise<
            Record<string, IconType>
          >,
      };
      const loader = loaders[library || ""];
      if (!loader || !iconName) {
        setIconEl(null);
        return;
      }
      try {
        const mod = await loader();
        const Exported = (mod as unknown as Record<string, any>)[iconName];
        if (!cancelled) {
          if (Exported) {
            let ComponentToRender: any | null = null;
            if (React.isValidElement(Exported)) {
              ComponentToRender = (Exported as React.ReactElement).type as any;
            } else if (
              typeof Exported === "function" ||
              (typeof Exported === "object" && Exported !== null)
            ) {
              ComponentToRender = Exported as any;
            }
            if (ComponentToRender) {
              setIconEl(
                React.createElement(ComponentToRender, {
                  className: "h-3.5 w-3.5 text-brand-light/80",
                }),
              );
            } else {
              setIconEl(null);
            }
          } else {
            setIconEl(null);
          }
        }
      } catch {
        if (!cancelled) setIconEl(null);
      }
    }
    loadIcon();
    return () => {
      cancelled = true;
    };
  }, [panel.icon]);
  const { handleResolutionChange, handleAspectRatioChange } =
    useResolutionAspectSync({
      clipId,
      inputs,
      getInputById,
      updateModelInput: (cId, inputId, patch) =>
        updateModelInput(cId, inputId, patch as any),
    });

  // Scheduler components (from manifest spec.components -> scheduler_options)
  const clip: any = getClipById(clipId);
  const schedulerComponents: ManifestComponent[] = useMemo(() => {
    const comps = (clip?.manifest?.spec?.components || []) as ManifestComponent[];
    return comps.filter((c: any) => String(c?.type) === "scheduler");
  }, [clip]);

  // Only render scheduler selector when this panel is named 'scheduler'
  const shouldShowScheduler = useMemo(() => {
    if (!schedulerComponents || schedulerComponents.length === 0) return false;
    return String(panel.name || "").toLowerCase() === "scheduler";
  }, [panel.name, schedulerComponents]);

  // Only render attention selector when this panel is named 'attention' and options exist
  const shouldShowAttention = useMemo(() => {
    const attentionOptions = (clip?.manifest?.spec?.attention_types_detail ||
      []) as any[];
    if (!attentionOptions || attentionOptions.length === 0) return false;
    return String(panel.name || "").toLowerCase() === "attention";
  }, [panel.name, clip?.manifest?.spec?.attention_types_detail]);

  const rowSub = 56;

  return (
    <div className="">
      {panel.label && (
        <div
          onClick={() => setCollapsed((v) => !v)}
          className={cn("flex items-center justify-between  py-2.5 px-3", {
            "rounded-b": collapsed,
            "rounded-b-none ": !collapsed || !collapsible,
            "cursor-pointer": collapsible,
          })}
        >
          <h3 className="text-brand-light text-[11.5px] text-start font-medium flex items-center gap-x-2">
            {iconEl}
            {panel.label}
          </h3>
          {collapsible && (
            <button
              type="button"
              aria-label={collapsed ? "Expand panel" : "Collapse panel"}
              className="p-1 text-brand-light/60 hover:text-brand-light transition-colors"
            >
              <ChevronDown
                className={`h-3.5 w-3.5 transition-transform ${collapsed ? "-rotate-90" : "rotate-0"}`}
              />
            </button>
          )}
        </div>
      )}
      {(!collapsible || !collapsed) && (
        <div
          className="px-3 py-4  pt-2"
          style={{
            display: "flex",
            flexDirection: panel.layout.flow as "row" | "column",
            gap: "12px",
          }}
        >
          {shouldShowScheduler && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "12px",
                width: "100%",
                minWidth: 0,
              }}
            >
              {schedulerComponents.map((comp, idx) => (
                <SchedulerPanel
                  key={getSchedulerComponentKey(comp, idx)}
                  clipId={clipId}
                  component={comp}
                  schedulerIndex={idx}
                />
              ))}
            </div>
          )}
          {shouldShowAttention && (
            <div
              style={{
                display: "flex",
                flexDirection: "row",
                gap: "12px",
                width: "100%",
                minWidth: 0,
              }}
            >
              <AttentionPanel clipId={clipId} />
            </div>
          )}

          {panel.layout.rows.map((row) => {
            // Determine if we should override layout for small panels:
            // If each item's computed width would be too small, stack items vertically.
            const estimatedPerItem = panelSize / row.length - rowSub;
            const shouldStackRowItems = estimatedPerItem < 220;
            const perItemPanelSize = shouldStackRowItems
              ? panelSize - rowSub
              : estimatedPerItem;
            return (
              <div
                key={row.join("-")}
                style={{
                  display: "flex",
                  flexDirection: shouldStackRowItems ? "column" : "row",
                  columnGap: "10px",
                  rowGap: shouldStackRowItems ? "10px" : "0px",
                  alignItems: "stretch",
                }}
              >
                {row.map((inputId) => {
                  const input = getInputById(inputId);
                  switch (input?.type) {
                    case "text":
                      if (input.value === undefined) {
                        updateModelInput(clipId, inputId, {
                          value: input.default || "",
                        });
                      }
                      return (
                        <TextInput
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={input?.value || ""}
                          defaultValue={input?.default}
                          onChange={(value) =>
                            updateModelInput(clipId, inputId, { value })
                          }
                          placeholder={input?.placeholder}
                        />
                      );
                    case "text_list": {
                      const parseTextListValue = (v: any) => {
                        if (!v) return [] as string[];
                        if (Array.isArray(v)) {
                          return v.map((x) => (x == null ? "" : String(x)));
                        }
                        if (typeof v === "string") {
                          try {
                            const parsed = JSON.parse(v);
                            if (Array.isArray(parsed)) {
                              return parsed.map((x) =>
                                x == null ? "" : String(x),
                              );
                            }
                          } catch {
                            // ignore
                          }
                          // fallback: treat as newline-separated
                          return v
                            .split("\n")
                            .map((s) => s.trim())
                            .filter((s) => s.length > 0);
                        }
                        return [String(v)];
                      };

                      const currentVal: string[] = parseTextListValue(
                        (input as any)?.value ?? (input as any)?.default,
                      );
                      const maxItems =
                        (input as any)?.max_items ?? (input as any)?.maxItems;

                      return (
                        <TextInputList
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={currentVal}
                          defaultValue={
                            Array.isArray((input as any)?.default)
                              ? ((input as any).default as string[])
                              : undefined
                          }
                          maxItems={
                            typeof maxItems === "number" ? maxItems : undefined
                          }
                          placeholder={input?.placeholder}
                          onChange={(vals) =>
                            updateModelInput(clipId, inputId, {
                              value:
                                vals && vals.length ? JSON.stringify(vals) : "",
                            })
                          }
                        />
                      );
                    }
                    case "image+mask": {
                      const parseImageValue = (v: any) => {
                        if (!v) return null;
                        if (typeof v === "object" && v !== null) {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return v;
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return v;
                        }
                        if (typeof v === "string") {
                          try {
                            const obj = JSON.parse(v);
                            if (
                              obj &&
                              (obj.kind === "media" || obj.kind === "clip")
                            )
                              return obj;
                            if (obj && (obj.clipId || obj.assetId || obj.type))
                              return obj;
                          } catch {}
                          return { kind: "media", assetUrl: v } as any;
                        }
                        return null;
                      };
                      const currentVal: any = parseImageValue(
                        (input as any)?.value ?? (input as any)?.default,
                      );
                      const mapToId = (input as any)?.map_to as
                        | string
                        | undefined;
                      const panelSizeToUse = perItemPanelSize;
                      return (
                        <ImageInput
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          onChange={(v: any) => {
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            });
                            if (!v && mapToId) {
                              updateModelInput(clipId, mapToId, { value: "" });
                            } else if (v && mapToId) {
                              const hasMasks =
                                Array.isArray((v as any)?.masks) &&
                                (v as any).masks.length > 0;
                              if (hasMasks) {
                                const mapped = {
                                  ...(v as any),
                                  masks: [],
                                  disableTimelineSync: true,
                                };
                                updateModelInput(clipId, mapToId, {
                                  value: JSON.stringify(mapped),
                                });
                              }
                            }
                          }}
                        />
                      );
                    }
                    case "image+preprocessor": {
                      // Value may be either a raw selection or a composite object with selection/apply/preprocessor_ref
                      const parseComposite = (v: any) => {
                        if (!v)
                          return { selection: null, apply: undefined } as any;
                        if (typeof v === "string") {
                          try {
                            const obj = JSON.parse(v);
                            if (
                              obj &&
                              (obj.selection ||
                                obj.apply !== undefined ||
                                obj.apply_preprocessor !== undefined ||
                                obj.preprocessor_ref)
                            )
                              return obj as any;
                            if (
                              obj &&
                              (obj.kind === "media" || obj.kind === "clip")
                            )
                              return { selection: obj } as any;
                            if (obj && (obj.clipId || obj.assetId || obj.type))
                              return { selection: obj } as any;
                          } catch {}
                          return {
                            selection: { kind: "media", assetUrl: v },
                          } as any;
                        }
                        if (typeof v === "object") {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return { selection: v } as any;
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return { selection: v } as any;
                          return v as any;
                        }
                        return { selection: null } as any;
                      };
                      const composite = parseComposite((input as any)?.value);
                      const selectionVal: any = composite?.selection ?? null;
                      const preprocRef = (input as any)?.preprocessor_ref as
                        | string
                        | undefined;
                      const preprocName = (input as any)?.preprocessor_name as
                        | string
                        | undefined;
                      const panelSizeToUse = perItemPanelSize;
                      return (
                        <ImageInput
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label}
                          description={input?.description}
                          value={selectionVal}
                          panelSize={panelSizeToUse}
                          preprocessorRef={preprocRef}
                          preprocessorName={preprocName}
                          applyPreprocessorInitial={
                            typeof composite?.apply === "boolean"
                              ? composite.apply
                              : typeof composite?.apply_preprocessor ===
                                  "boolean"
                                ? composite.apply_preprocessor
                                : undefined
                          }
                          onChangeComposite={(v: any) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            })
                          }
                          onChange={(v: any) => {
                            // Fallback: if child calls onChange (selection-only), wrap into composite
                            const payload = {
                              selection: v ?? null,
                              preprocessor_ref: preprocRef,
                              preprocessor_name: preprocName,
                              apply_preprocessor: true,
                              apply: true,
                            };
                            updateModelInput(clipId, inputId, {
                              value: JSON.stringify(payload),
                            });
                          }}
                        />
                      );
                    }
                    case "select":
                      return (
                        <SelectInput
                          key={inputId}
                          label={input?.label}
                          defaultOption={input?.default}
                          description={input?.description}
                          value={input?.value || ""}
                          onChange={(value) => {
                            updateModelInput(clipId, inputId, { value });
                            if (inputId === "resolution") {
                              handleResolutionChange(value);
                            } else if (inputId === "aspect_ratio") {
                              handleAspectRatioChange(value);
                            }
                          }}
                          options={input?.options || []}
                        />
                      );
                    case "boolean": {
                      if (input.value === undefined) {
                        updateModelInput(clipId, inputId, {
                          value:
                            String(input?.default ?? "false").toLowerCase() ===
                            "true"
                              ? "true"
                              : "false",
                        });
                      }
                      const boolVal =
                        String(
                          input?.value ?? input?.default ?? "false",
                        ).toLowerCase() === "true";
                      return (
                        <BooleanInput
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={boolVal}
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? "true" : "false",
                            })
                          }
                        />
                      );
                    }
                    case "number+slider": {
                      const numVal = Number(
                        input?.value ?? input?.default ?? 0,
                      );
                      const toFixed = input.step
                        ? ((input.step.toString().split(".")[1]?.length ?? 0) + 1)
                        : 0;

                      return (
                        <NumberInputSlider
                          key={inputId}
                          label={input?.label || ""}
                          description={input?.description}
                          value={Number.isFinite(numVal) ? numVal : 0}
                          min={input?.min}
                          max={input?.max}
                          step={input?.step}
                          toFixed={toFixed}
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, {
                              value: v.toString(),
                            })
                          }
                        />
                      );
                    }

                    case "number": {
                      const strVal = String(
                        input?.value ?? input?.default ?? "",
                      );

                      return (
                        <NumberInput
                          startLogo={
                            input?.label
                              ? input?.label.charAt(0).toUpperCase()
                              : ""
                          }
                          key={inputId}
                          label={input?.label}
                          toFixed={input?.value_type === "integer" ? 0 : 2}
                          description={input?.description}
                          value={strVal}
                          min={input?.min}
                          max={input?.max}
                          step={input?.step}
                          onChange={(v) => {
                            updateModelInput(clipId, inputId, { value: v });
                            if (inputId === "height" || inputId === "width") {
                              // Manual override: set selects to custom to disable auto-sync
                              updateModelInput(clipId, "resolution", {
                                value: "custom",
                              });
                              updateModelInput(clipId, "aspect_ratio", {
                                value: "custom",
                              });
                            }
                          }}
                        />
                      );
                    }
                    case "video+preprocessor": {
                      const parseComposite = (v: any) => {
                        if (!v)
                          return { selection: null, apply: undefined } as any;
                        if (typeof v === "string") {
                          try {
                            const obj = JSON.parse(v);
                            if (
                              obj &&
                              (obj.selection ||
                                obj.apply !== undefined ||
                                obj.apply_preprocessor !== undefined ||
                                obj.preprocessor_ref)
                            )
                              return obj as any;
                            if (obj && (obj.clipId || obj.assetId || obj.type))
                              return { selection: obj } as any;
                          } catch {}
                        }
                        return v as any;
                      };
                      const composite = parseComposite((input as any)?.value);
                      const selectionVal: any = (() => {
                        const raw =
                          composite?.selection ??
                          (input as any)?.value ??
                          (input as any)?.default;
                        if (!raw) return null;
                        if (typeof raw === "string") {
                          try {
                            const parsed = JSON.parse(raw);
                            if (
                              parsed &&
                              (parsed.kind === "media" ||
                                parsed.kind === "clip")
                            )
                              return parsed;
                          } catch {}
                          return null;
                        }
                        return raw;
                      })();
                      const preprocRef = (input as any)?.preprocessor_ref as
                        | string
                        | undefined;
                      const preprocName = (input as any)?.preprocessor_name as
                        | string
                        | undefined;
                      const panelSizeToUse = perItemPanelSize;
                      let maxDuration = (input )?.max_duration_secs;
                      if (typeof maxDuration === "number" && maxDuration > 0) {
                        // convert to frames
                        maxDuration = Math.max(1, Math.floor(maxDuration * fps));
                      }
                      return (
                        <VideoInput
                          inputId={inputId}
                          clipId={clipId}
                          key={inputId}
                          label={input?.label || "Video"}
                          description={input?.description}
                          value={selectionVal}
                          panelSize={panelSizeToUse}
                          maxDuration={maxDuration}
                          preprocessorRef={preprocRef}
                          preprocessorName={preprocName}
                          applyPreprocessorInitial={
                            typeof composite?.apply === "boolean"
                              ? composite.apply
                              : typeof composite?.apply_preprocessor ===
                                  "boolean"
                                ? composite.apply_preprocessor
                                : undefined
                          }
                          onChangeComposite={(v: any) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            })
                          }
                          onChange={(v) => {
                            const payload = {
                              selection: v ?? null,
                              preprocessor_ref: preprocRef,
                              preprocessor_name: preprocName,
                              apply_preprocessor: true,
                              apply: true,
                            };
                            updateModelInput(clipId, inputId, {
                              value: JSON.stringify(payload),
                            });
                          }}
                        />
                      );
                    }
                    case "number_list": {
                      const strVal = String(
                        input?.value ?? input?.default ?? "",
                      );
                      const maxItems =
                        (input as any)?.max_items ?? (input as any)?.maxItems;
                      return (
                        <NumberListInput
                          startLogo={
                            input?.label
                              ? input?.label.charAt(0).toUpperCase()
                              : ""
                          }
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={strVal}
                          min={(input as any)?.min}
                          max={(input as any)?.max}
                          step={(input as any)?.step}
                          valueType={(input as any)?.value_type}
                          maxItems={
                            typeof maxItems === "number" ? maxItems : undefined
                          }
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, { value: v })
                          }
                        />
                      );
                    }
                    case "random": {
                      const strVal = String(
                        input?.value ?? input?.default ?? "-1",
                      );
                      return (
                        <RandomInput
                          startLogo="🎲"
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={strVal}
                          min={input?.min}
                          max={input?.max}
                          step={input?.step}
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, { value: v })
                          }
                        />
                      );
                    }

                    case "image": {
                      const parseImageValue = (v: any) => {
                        if (!v) return null;
                        if (typeof v === "object" && v !== null) {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return v;
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return v;
                        }
                        if (typeof v === "string") {
                          try {
                            const obj = JSON.parse(v);
                            if (
                              obj &&
                              (obj.kind === "media" || obj.kind === "clip")
                            )
                              return obj;
                            if (obj && (obj.clipId || obj.assetId || obj.type))
                              return obj;
                          } catch {}
                          return { kind: "media", assetUrl: v } as any;
                        }
                        return null;
                      };
                      const currentVal: any = parseImageValue(input?.value);
                      const panelSizeToUse = perItemPanelSize;
                      return (
                        <ImageInput
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          onChange={(v: any) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            })
                          }
                        />
                      );
                    }
                    case "image_list": {
                      const parseImageListValue = (v: any) => {
                        if (!v) return [] as any[];
                        const toSelection = (item: any): any => {
                          if (!item) return null;
                          if (typeof item === "object") {
                            if (
                              (item as any).kind === "media" ||
                              (item as any).kind === "clip"
                            )
                              return item;
                            if (
                              (item as any).clipId ||
                              (item as any).assetId ||
                              (item as any).type
                            )
                              return item;
                            return null;
                          }
                          if (typeof item === "string") {
                            try {
                              const obj = JSON.parse(item);
                              if (
                                obj &&
                                (obj.kind === "media" || obj.kind === "clip")
                              )
                                return obj;
                              if (obj && (obj.clipId || obj.assetId || obj.type))
                                return obj;
                            } catch {}
                            return { kind: "media", assetUrl: item } as any;
                          }
                          return null;
                        };
                        let arr: any[] = [];
                        if (Array.isArray(v)) {
                          arr = v;
                        } else if (typeof v === "string") {
                          try {
                            const parsed = JSON.parse(v);
                            if (Array.isArray(parsed)) {
                              arr = parsed;
                            } else if (parsed && typeof parsed === "object") {
                              arr = [parsed];
                            }
                          } catch {
                            // ignore parse errors and fall back to empty
                          }
                        } else if (typeof v === "object") {
                          arr = [v];
                        }
                        // Preserve null/empty entries so they render as empty slots in ImageInputList
                        return arr.map(toSelection);
                      };
                      const currentVal: any[] = parseImageListValue(
                        (input as any)?.value ?? (input as any)?.default,
                      );
                      const panelSizeToUse = perItemPanelSize;
                      const min = (input as any)?.min;
                      const max = (input as any)?.max;
                      return (
                        <ImageInputList
                          key={inputId}
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          min={typeof min === "number" ? min : undefined}
                          max={typeof max === "number" ? max : undefined}
                          onChange={(vals) => {
                            updateModelInput(clipId, inputId, {
                              value:
                                vals && vals.length ? JSON.stringify(vals) : "",
                            });
                          }}
                        />
                      );
                    }
                    case "video": {
                      const parseVideoValue = (v: any) => {
                        if (!v) return null;
                        const coerceRange = (obj: any) => {
                          const start = Math.max(
                            0,
                            Math.round(Number(obj?.startFrame ?? 0)),
                          );
                          const end = Math.max(
                            start + 1,
                            Math.round(Number(obj?.endFrame ?? start + 1)),
                          );
                          return { ...obj, startFrame: start, endFrame: end };
                        };
                        if (typeof v === "object" && v !== null) {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return coerceRange(v);
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return coerceRange(v);
                        }
                        if (typeof v === "string") {
                          try {
                            const parsed = JSON.parse(v);
                            if (
                              parsed &&
                              (parsed.kind === "media" ||
                                parsed.kind === "clip")
                            )
                              return coerceRange(parsed);
                            if (
                              parsed &&
                              (parsed.clipId || parsed.assetId || parsed.type)
                            )
                              return coerceRange(parsed);
                          } catch {}
                        }
                        return null;
                      };
                      const currentVal: any = parseVideoValue(
                        input?.value ?? input?.default,
                      );
                      let maxDuration = (input)?.max_duration_secs
                      const panelSizeToUse = perItemPanelSize;
                      if (typeof maxDuration === "number" && maxDuration > 0) {
                        // convert to frames
                        maxDuration = Math.max(1, Math.floor(maxDuration * fps));
                      }

                     
                      return (
                        <VideoInput
                          inputId={inputId}
                          clipId={clipId}
                          maxDuration={maxDuration}
                          label={input?.label || "Video"}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            })
                          }
                        />
                      );
                    }
                    case "video+mask": {
                      const parseVideoValue = (v: any) => {
                        if (!v) return null;
                        const coerceRange = (obj: any) => {
                          const start = Math.max(
                            0,
                            Math.round(Number(obj?.startFrame ?? 0)),
                          );
                          const end = Math.max(
                            start + 1,
                            Math.round(Number(obj?.endFrame ?? start + 1)),
                          );
                          return { ...obj, startFrame: start, endFrame: end };
                        };
                        if (typeof v === "object" && v !== null) {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return coerceRange(v);
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return coerceRange(v);
                        }
                        if (typeof v === "string") {
                          try {
                            const parsed = JSON.parse(v);
                            if (
                              parsed &&
                              (parsed.kind === "media" ||
                                parsed.kind === "clip")
                            )
                              return coerceRange(parsed);
                            if (
                              parsed &&
                              (parsed.clipId || parsed.assetId || parsed.type)
                            )
                              return coerceRange(parsed);
                          } catch {}
                        }
                        return null;
                      };
                      const currentVal: any = parseVideoValue(
                        (input as any)?.value ?? (input as any)?.default,
                      );
                      const mapToId = (input as any)?.map_to as
                        | string
                        | undefined;
                      const panelSizeToUse = perItemPanelSize;
                      let maxDuration = (input)?.max_duration_secs;
                      if (typeof maxDuration === "number" && maxDuration > 0) {
                        // convert to frames
                        maxDuration = Math.max(1, Math.floor(maxDuration * fps));
                      }
                      return (
                        <VideoInput
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label || "Video"}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          maxDuration={maxDuration}
                          onChange={(v) => {
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            });
                            if (!v && mapToId) {
                              updateModelInput(clipId, mapToId, { value: "" });
                            } else if (v && mapToId) {
                              const hasMasks =
                                Array.isArray((v as any)?.masks) &&
                                (v as any).masks.length > 0;
                              if (hasMasks) {
                                const mapped = {
                                  ...(v as any),
                                  masks: [],
                                  disableTimelineSync: true,
                                };
                                updateModelInput(clipId, mapToId, {
                                  value: JSON.stringify(mapped),
                                });
                              }
                            }
                          }}
                        />
                      );
                    }
                    case "audio": {
                      const parseAudioValue = (v: any) => {
                        if (!v) return null;
                        const coerceRange = (obj: any) => {
                          const start = Math.max(
                            0,
                            Math.round(Number(obj?.startFrame ?? 0)),
                          );
                          const end = Math.max(
                            start + 1,
                            Math.round(Number(obj?.endFrame ?? start + 1)),
                          );
                          return { ...obj, startFrame: start, endFrame: end };
                        };
                        if (typeof v === "object" && v !== null) {
                          if (
                            (v as any).kind === "media" ||
                            (v as any).kind === "clip"
                          )
                            return coerceRange(v);
                          if (
                            (v as any).clipId ||
                            (v as any).assetId ||
                            (v as any).type
                          )
                            return coerceRange(v);
                        }
                        if (typeof v === "string") {
                          try {
                            const parsed = JSON.parse(v);
                            if (
                              parsed &&
                              (parsed.kind === "media" ||
                                parsed.kind === "clip")
                            )
                              return coerceRange(parsed);
                            if (
                              parsed &&
                              (parsed.clipId || parsed.assetId || parsed.type)
                            ) 
                              return coerceRange(parsed);
                          } catch {}
                          return null;
                        }
                        return null;
                      };

                      const currentVal: any = parseAudioValue(
                        input?.value ?? input?.default,
                      );
                      const panelSizeToUse = perItemPanelSize;
                      return (
                        <AudioInput
                          inputId={inputId}
                          clipId={clipId}
                          label={input?.label || "Audio"}
                          description={input?.description}
                          value={currentVal}
                          panelSize={panelSizeToUse}
                          onChange={(v) =>
                            updateModelInput(clipId, inputId, {
                              value: v ? JSON.stringify(v) : "",
                            })
                          }
                        />
                      );
                    }

                    default:
                      return null;
                  }
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

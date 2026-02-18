export type ModelDownloadProfile = "auto" | "maximum_performance";

export const normalizeModelDownloadProfile = (
  value: unknown,
): ModelDownloadProfile => {
  const raw = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/-/g, "_")
    .replace(/\s+/g, "_");
  if (
    raw === "maximum_performance" ||
    raw === "max_performance" ||
    raw === "max" ||
    raw === "performance" ||
    raw === "full" ||
    raw === "full_models" ||
    raw === "full_model"
  ) {
    return "maximum_performance";
  }
  return "auto";
};

type ModelPathLike = {
  path: string;
  variant?: string;
  precision?: string;
  type?: string;
  file_size?: number;
  resource_requirements?: {
    min_vram_gb?: number;
    recommended_vram_gb?: number;
    [key: string]: any;
  };
  custom?: boolean;
  [key: string]: any;
};

const tierForModelPathItem = (item: ModelPathLike): string => {
  const variant = String(item.variant || "").toLowerCase();
  const precision = String(item.precision || "").toLowerCase();
  const modelType = String(item.type || "").toLowerCase();
  const path = String(item.path || "").toLowerCase();
  const joined = `${variant} ${precision} ${modelType} ${path}`;

  if (joined.includes("fp8")) return "fp8";
  if (joined.includes("q8")) return "q8";
  if (joined.includes("q6") || joined.includes("q5")) return "q6";
  if (
    joined.includes("q4") ||
    joined.includes("q3") ||
    joined.includes("q2")
  ) {
    return "q4";
  }

  if (
    variant === "default" ||
    variant === "full" ||
    joined.includes("bf16") ||
    joined.includes("fp16") ||
    joined.includes("float16") ||
    joined.includes("float32") ||
    joined.includes("fp32")
  ) {
    return "full";
  }

  return "other";
};

const isFlux2DevTextEncoder = (options: {
  componentType?: string;
  manifestMetadata?: Record<string, any> | null;
}): boolean => {
  if (String(options.componentType || "").trim().toLowerCase() !== "text_encoder") {
    return false;
  }
  const md = options.manifestMetadata || {};
  const model = String(md.model || "").trim().toLowerCase();
  const id = String(md.id || "").trim().toLowerCase();
  const name = String(md.name || "").trim().toLowerCase();
  const joined = `${model} ${id} ${name}`.replace(/_/g, "-");
  return (
    joined.includes("flux2-dev") ||
    joined.includes("flux dev 2") ||
    (model === "flux2" && joined.includes("dev"))
  );
};

const tierOrder = (options: {
  profile: ModelDownloadProfile;
  componentType?: string;
  manifestMetadata?: Record<string, any> | null;
}): string[] => {
  if (options.profile === "maximum_performance") {
    return ["full", "fp8", "q8", "q6", "q4", "other"];
  }
  if (
    isFlux2DevTextEncoder({
      componentType: options.componentType,
      manifestMetadata: options.manifestMetadata,
    })
  ) {
    return ["q6", "fp8", "q8", "q4", "full", "other"];
  }
  return ["fp8", "q8", "q6", "q4", "full", "other"];
};

export const selectPreferredModelPathItem = (
  items: ModelPathLike[],
  options?: {
    profile?: string;
    componentType?: string;
    manifestMetadata?: Record<string, any> | null;
    includeCustom?: boolean;
  },
): ModelPathLike | undefined => {
  if (!Array.isArray(items) || items.length === 0) return undefined;
  const includeCustom = Boolean(options?.includeCustom);
  const profile = normalizeModelDownloadProfile(options?.profile || "auto");

  const candidates = items.filter((it) => {
    if (!it || typeof it.path !== "string" || !it.path) return false;
    if (!includeCustom && it.custom) return false;
    return true;
  });
  if (!candidates.length) return undefined;

  const order = tierOrder({
    profile,
    componentType: options?.componentType,
    manifestMetadata: options?.manifestMetadata || null,
  });
  const orderMap = new Map(order.map((tier, idx) => [tier, idx]));

  return candidates
    .map((item) => ({
      item,
      tier: tierForModelPathItem(item),
      isDefault: String(item.variant || "").toLowerCase() === "default",
      size: typeof item.file_size === "number" && item.file_size > 0 ? item.file_size : null,
    }))
    .sort((a, b) => {
      const aOrder = orderMap.get(a.tier) ?? 999;
      const bOrder = orderMap.get(b.tier) ?? 999;
      if (aOrder !== bOrder) return aOrder - bOrder;

      if (profile === "maximum_performance") {
        if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1;
        const aSize = a.size == null ? -Infinity : a.size;
        const bSize = b.size == null ? -Infinity : b.size;
        return bSize - aSize;
      }

      const aSize = a.size == null ? Infinity : a.size;
      const bSize = b.size == null ? Infinity : b.size;
      if (aSize !== bSize) return aSize - bSize;
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1;
      return 0;
    })[0]?.item;
};

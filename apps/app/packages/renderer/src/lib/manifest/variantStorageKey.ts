import type { ManifestDocument, ManifestGroup } from "@/lib/manifest/api";

type ManifestLike = ManifestDocument | null | undefined;
type GroupLike = ManifestGroup | null | undefined;

const normalizeId = (value: unknown): string =>
  String(value ?? "").trim();

const pushUnique = (arr: string[], value: string) => {
  if (!value) return;
  if (!arr.includes(value)) arr.push(value);
};

const manifestIdentityCandidates = (manifest: ManifestLike): string[] => {
  const out: string[] = [];
  pushUnique(out, normalizeId((manifest as any)?.metadata?.id));
  pushUnique(out, normalizeId((manifest as any)?.id));
  return out;
};

const variantIdentityCandidates = (variant: any): string[] => {
  const out: string[] = [];
  pushUnique(out, normalizeId(variant?.id));
  pushUnique(out, normalizeId(variant?.manifest?.metadata?.id));
  pushUnique(out, normalizeId(variant?.manifest?.id));
  return out;
};

export const getManifestLegacyStorageKey = (manifest: ManifestLike): string => {
  const [first] = manifestIdentityCandidates(manifest);
  return first || "__default__";
};

export const resolveManifestVariantId = (args: {
  group?: GroupLike;
  manifest?: ManifestLike;
  preferredVariantId?: string | null;
}): string | undefined => {
  const preferredVariantId = normalizeId(args.preferredVariantId);
  if (preferredVariantId) {
    const variants = (args.group?.variants || []) as any[];
    if (!variants.length) return preferredVariantId;
    const preferred = variants.find(
      (variant) => normalizeId(variant?.id) === preferredVariantId,
    );
    if (preferred) return preferredVariantId;
  }

  const variants = (args.group?.variants || []) as any[];
  if (!variants.length) return undefined;

  const manifestIds = manifestIdentityCandidates(args.manifest);
  if (!manifestIds.length) return undefined;
  const manifestIdSet = new Set(manifestIds);

  const matches = variants.filter((variant) =>
    variantIdentityCandidates(variant).some((id) => manifestIdSet.has(id)),
  );

  if (matches.length === 1) {
    const resolved = normalizeId(matches[0]?.id);
    return resolved || undefined;
  }

  return undefined;
};

export const getVariantStorageKey = (args: {
  group?: GroupLike;
  manifest?: ManifestLike;
  preferredVariantId?: string | null;
}): string => {
  const variantId = resolveManifestVariantId(args);
  if (variantId) return `variant:${variantId}`;
  return `manifest:${getManifestLegacyStorageKey(args.manifest)}`;
};

export const getVariantStorageLookupKeys = (args: {
  group?: GroupLike;
  manifest?: ManifestLike;
  preferredVariantId?: string | null;
  includeLegacy?: boolean;
}): string[] => {
  const keys: string[] = [];
  pushUnique(keys, getVariantStorageKey(args));

  if (args.includeLegacy === false) {
    return keys;
  }

  const legacy = getManifestLegacyStorageKey(args.manifest);
  pushUnique(keys, `manifest:${legacy}`);
  // Backward compatibility with older snapshots that stored the raw manifest key.
  pushUnique(keys, legacy);
  return keys;
};

export const getVariantScopedValue = <T>(
  valuesByVariant: Record<string, T> | undefined,
  lookupKeys: string[],
): T | undefined => {
  if (!valuesByVariant) return undefined;
  for (const key of lookupKeys) {
    if (Object.prototype.hasOwnProperty.call(valuesByVariant, key)) {
      return valuesByVariant[key];
    }
  }
  return undefined;
};

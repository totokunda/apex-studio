#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-apps/api/manifest/v0.1.2}"

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Target directory does not exist: ${TARGET_DIR}" >&2
  exit 1
fi

before_legacy_count="$(
  (
    grep -R -h -E '^[[:space:]]*variant:[[:space:]]*["'\'']?(FP8_E4M3FN|GGUF_Q8_0|GGUF_Q6_K|GGUF_Q4_K_M)["'\'']?([[:space:]]*(#.*)?)$' \
      "${TARGET_DIR}" || true
  ) | wc -l | tr -d '[:space:]'
)"

while IFS= read -r -d '' file; do
  perl -i -pe '
    s/^(\s*variant:\s*)(["\047]?)FP8_E4M3FN\2(\s*(?:#.*)?)$/${1}${2}FP8${2}${3}/;
    s/^(\s*variant:\s*)(["\047]?)GGUF_Q8_0\2(\s*(?:#.*)?)$/${1}${2}Q8${2}${3}/;
    s/^(\s*variant:\s*)(["\047]?)GGUF_Q6_K\2(\s*(?:#.*)?)$/${1}${2}Q6${2}${3}/;
    s/^(\s*variant:\s*)(["\047]?)GGUF_Q4_K_M\2(\s*(?:#.*)?)$/${1}${2}Q4${2}${3}/;
  ' "${file}"
done < <(find "${TARGET_DIR}" -type f \( -name '*.yml' -o -name '*.yaml' \) -print0)

after_legacy_count="$(
  (
    grep -R -h -E '^[[:space:]]*variant:[[:space:]]*["'\'']?(FP8_E4M3FN|GGUF_Q8_0|GGUF_Q6_K|GGUF_Q4_K_M)["'\'']?([[:space:]]*(#.*)?)$' \
      "${TARGET_DIR}" || true
  ) | wc -l | tr -d '[:space:]'
)"

updated_short_count="$(
  (
    grep -R -h -E '^[[:space:]]*variant:[[:space:]]*["'\'']?(FP8|Q8|Q6|Q4)["'\'']?([[:space:]]*(#.*)?)$' \
      "${TARGET_DIR}" || true
  ) | wc -l | tr -d '[:space:]'
)"

echo "Target: ${TARGET_DIR}"
echo "Legacy variant lines before: ${before_legacy_count}"
echo "Legacy variant lines after:  ${after_legacy_count}"
echo "Normalized variant lines now: ${updated_short_count}"

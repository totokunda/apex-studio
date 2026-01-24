PY312="$(command -v python3.12 || true)"
if [[ -z "$PY312" ]]; then
  echo "ERROR: python3.12 is required for linux CUDA bundles (ray has no cp313 wheels). Install Python 3.12 or activate a py312 env." >&2
  exit 1
fi

"$PY312" scripts/bundle_python.py \
    --platform linux \
    --cuda cuda128 \
    --output ./dist \
    --python "$PY312" \
    --tar-zst \
    --tar-zst-level 12 \
    --bundle-version 0.1.0

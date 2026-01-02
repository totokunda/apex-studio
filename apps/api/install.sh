cd /Users/tosinkuye/apex-workspace/apex-studio/apps/api \
    && python3 scripts/bundle_python.py \
    --platform darwin \
    --cuda cpu \
    --output ./dist \
    --tar-zst \
    --tar-zst-level 12
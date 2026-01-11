### Scripts folder layout

This directory contains dev + release utilities for the Apex Engine API (`apps/api`).

We keep a few **stable entrypoints** at the root for backwards compatibility:
- `scripts/bundle_python.py`
- `scripts/setup.py`
- `scripts/install.sh`
- `scripts/apply_code_update.py`
- `scripts/apply_patches.sh`

Everything else is organized into subfolders:
- `scripts/bundling/`: python-api / python-code bundling utilities
- `scripts/setup/`: setup + environment/bootstrap helpers
- `scripts/updates/`: patching and code-update tooling
- `scripts/dev/`: dev-only installers/helpers
- `scripts/models/`: model conversion / quantization utilities
- `scripts/release/`: upload/publish scripts
- `scripts/maintenance/`: manifest maintenance, housekeeping
- `scripts/benchmarks/`: benchmarking utilities
- `scripts/migration/`: migration-related shell scripts
- `scripts/monitoring/`: GPU/VRAM monitoring helpers
- `scripts/smoke_tests/`: smoke test runner and tests



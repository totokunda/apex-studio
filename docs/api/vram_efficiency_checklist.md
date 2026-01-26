# VRAM efficiency checklist (video models)

Use this checklist to validate peak VRAM reductions after enabling the new
efficiency profile / RoPE-on-CPU options.

## 1) Enable step memory logging

Set the environment variable before running inference:

```
export APEX_STEP_MEM=1
```

This enables per-step memory logging from `src/utils/step_mem.py` inside model
forward passes that are instrumented.

## 2) Run a baseline

Run a short denoise (5-10 steps) with defaults:

- `chunking_profile=none`
- `rope_on_cpu=false`

Capture the peak memory line from the logs for comparison.

## 3) Run the efficiency profile

Run the same prompt with:

- `efficiency_profile=balanced` (or `aggressive`)
- `rope_on_cpu=true`

Compare peak allocations to the baseline.

## 4) Validate output sanity

Confirm that:

- output shapes are unchanged
- no NaNs are present
- sample quality is visually consistent (within expected numerical drift)

## 5) Record results

Log the following for each model family:

- Model: WAN base / WAN multitalk / HunyuanVideo15
- Resolution and duration
- Peak VRAM baseline vs optimized
- Profile used (`balanced` or `aggressive`)

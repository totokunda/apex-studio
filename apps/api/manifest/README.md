## Apex Manifests

This directory contains YAML manifests that drive what appears in the Apex API/UI and how engines/components are wired up at runtime.

There are **two manifest families** in this folder today:

- **Model manifests (v1)**: `manifest/{image,video,upscalers}/*.yml` (most files). These describe a runnable model/pipeline: components, defaults, compute requirements, UI form, etc.
- **Preprocessor manifests**: `manifest/preprocessor/*.yml`. These describe a preprocessor entry for the preprocessor registry (module/class + parameter metadata).

### Directory layout (current)

- `manifest/image/*.yml`: image models (mostly `*.v1.yml`, but the extension is not significant)
- `manifest/video/*.yml`: video models (mostly `*.v1.yml`)
- `manifest/upscalers/*.yml`: upscalers (still the same v1 structure)
- `manifest/preprocessor/*.yml`: preprocessor registry entries (different schema from model manifests)

Notes:
- The API scans **all** `manifest/**/*.yml` files.
- `/manifest/list` only returns manifests that have **`spec.ui` present** and (by default) are **compatible with the current system’s compute capabilities**. Use `include_incompatible=true` to list everything.
- Manifests are addressed externally by **`metadata.id`** (see `/manifest/{manifest_id}`).

---

### v1 model manifest (what most manifests look like now)

Canonical top-level shape (snake_case):

```yaml
api_version: apex/v1
kind: Model

metadata:
  id: wan-2-2-a14b-image-to-video          # required in practice (API identifier)
  model: wan                               # model family label (used for grouping)
  name: Wan 2.2 A14B Image to Video        # required by schema
  version: 1.0.0                           # semver string
  description: ...
  tags: [wan, i2v, 14b, "2.2"]
  author: Wan-AI
  license: Apache-2.0
  demo_path: models/wan-2.2-i2v-14b.mp4
  categories:                              # used for UI filtering (list or string)
    - image-to-video

spec:
  engine: wan                              # required: src/engine/<engine>/*
  model_type: i2v                          # required (t2i, t2v, i2v, ti2v, upscale, ...)
  engine_type: torch                       # optional (torch | mlx)
  denoise_type: moe                        # optional (engine-specific)

  compute_requirements:                    # optional; used to filter incompatible models
    min_cuda_compute_capability: 7.5
    supported_compute_types: [cuda, cpu, metal]

  attention_types: [sdpa, flash, flash3, sage, metal_flash, xformers]
  fps: 16                                  # optional (video/upscale)
  default_duration_secs: 5.0               # optional (video)
  min_duration_secs: 1.0                   # optional (video)

  components:                              # required for execution
    - type: scheduler
      label: Scheduler
      default: UniPCMultistepScheduler
      scheduler_options:
        - name: UniPCMultistepScheduler
          base: diffusers.UniPCMultistepScheduler
          config_path: Wan-AI/.../scheduler_config.json

    - type: transformer
      name: transformer
      label: Transformer
      base: wan.base
      config_path: Wan-AI/.../transformer/config.json
      model_path:                          # usually a list of variants
        - path: Wan-AI/.../transformer
          variant: default
          precision: fp16
          type: safetensors
          file_size: 57154175533
          resource_requirements:
            min_vram_gb: 51
            recommended_vram_gb: 68

  loras:                                   # optional; list[str] or list[object]
    - source: someorg/somerepo/file.safetensors
      name: my_lora
      label: My LoRA
      scale: 1.0
      verified: true
      component: transformer                # optional: target component name

  defaults:                                # optional; merged into runtime kwargs
    run:
      num_inference_steps: 8
      guidance_scale: 5.0
      return_latents: false

  ui:                                      # required for the manifest to appear in /manifest/list
    panels:
      - name: prompting
        label: Prompting
        collapsible: true
        default_open: true
        layout:
          flow: column
          rows:
            - [prompt]
      - name: sampling
        label: Sampling
        collapsible: true
        default_open: false
        layout:
          flow: column
          rows:
            - [num_inference_steps]
            - [guidance_scale, seed]
    inputs:
      - id: prompt
        label: Prompt
        description: Describe what you want to generate.
        placeholder: A beautiful landscape at golden hour.
        type: text
        panel: prompting
        required: true
      - id: num_inference_steps
        label: Inference Steps
        type: number+slider
        value_type: integer
        panel: sampling
        default: 8
        min: 1
        max: 100
        step: 1
      - id: seed
        label: Seed
        type: random
        value_type: integer
        panel: sampling
        default: -1
```

### Field reference (v1 model manifests)

- **`api_version`**: should be `apex/v1` (snake_case is canonical). The loader is tolerant of `apiVersion`, but the schema and current manifests use `api_version`.
- **`kind`**: `Model` (most manifests) or `Pipeline` (supported by schema; uncommon in this repo).
- **`metadata.id`**: stable identifier used by the API (`/manifest/{manifest_id}`). Treat as unique across the repo.
- **`metadata.model`**: grouping label (e.g. `wan`, `flux`, `ltx2`).
- **`metadata.categories`**: user-facing category keys used by the API for filtering (e.g. `text-to-image`, `image-to-video`, `upscale`).
- **`spec.engine` / `spec.model_type`**: primary routing keys into `src/engine/<engine>/<model_type>.py`.
- **`spec.compute_requirements`**: optional; used to compute `compute_compatible` and filter `/manifest/list`.
- **`spec.attention_types`**: optional allow-list of attention backends for the UI/runtime.

### Components (v1 model manifests)

`spec.components[]` is the engine wiring. Common fields you’ll see:

- **`type`**: `scheduler` | `vae` | `text_encoder` | `transformer` | `helper` | `extra_model_path`
- **`name`**: component identifier. If omitted, the loader normalizes it to the component `type`.
- **`base`**: component implementation selector (engine-specific, e.g. `wan.base`, `flux.base`).
- **`config_path`**: optional config reference (often a HF repo path like `org/repo/subdir/config.json`).
- **`model_path`**: typically **a list of variant objects**, each with:
  - `path` (string): HF repo path, URL, or local path
  - `variant` (string): display name like `default`, `FP8`, `GGUF_Q8_0`
  - `precision` (string): e.g. `fp16`, `bf16`, `fp8`, `q4_0`, ...
  - `type` (string): e.g. `safetensors`, `gguf`
  - `file_size` (number): bytes
  - `resource_requirements` (object): e.g. `min_vram_gb`, `recommended_vram_gb`
- **`scheduler_options`** (scheduler-only): list of named scheduler choices.
- **`type: extra_model_path`**: attaches an additional downloadable file to an existing component:
  - `component`: target component name (e.g. `transformer`)
  - `model_path`: list of downloadable paths (same structure as above)

### UI (v1 model manifests)

Current manifests use a **panel + input list** format under `spec.ui`:

- **`ui.panels[]`**: controls grouping/layout (panel name, label, collapsible, default_open, and a simple layout grid via `rows`).
- **`ui.inputs[]`**: parameter definitions. Common keys used in this repo include:
  - `id`, `label`, `description`, `placeholder`, `panel`, `required`, `default`
  - `type`: examples in this repo include `text`, `image`, `video`, `audio`, `select`, `number`, `number+slider`, `boolean`, `random`
  - `value_type`: `integer` or `float` for numeric inputs
  - `min`, `max`, `step`, `options`
  - media helpers: `map_h`, `map_w`, `scale_by`, `max_duration_seconds`

---

### Preprocessor manifests (`manifest/preprocessor/*.yml`)

Preprocessor manifests are **not** v1 model manifests. They describe a preprocessor entry and its UI/parameter metadata for the preprocessor registry.

Shape used in this repo:

```yaml
name: Canny Edge Detection
category: Line
description: Classic multi-stage edge detection algorithm...
module: src.preprocess.canny
class: CannyDetector
supports_image: true
supports_video: true

parameters:
  - name: low_threshold
    display_name: Low Threshold
    type: int
    default: 100
    min: 0
    max: 500
    description: ...
  - name: detect_resolution
    display_name: Detection Resolution
    type: category
    default: 512
    options:
      - { name: Standard, value: 512 }
      - { name: High Definition, value: 1024 }
      - { name: Current Image, value: 0 }
```

### Validation

v1 model manifests can be validated against a JSON Schema (see `src/manifest/schema_v1.py`) when `jsonschema` is installed. The runtime loader is also intentionally permissive to avoid breaking existing manifests, so you may see fields in YAML that are “extra” but still supported.


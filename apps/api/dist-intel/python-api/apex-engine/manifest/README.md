## Apex Manifest

This directory defines how models and pipelines are described and loaded within Apex.

### Goals
- Versioned, registry-ready manifests (inspired by Dockerfile/Compose)
- Strong validation with clear error messages
- Consistent organization across engines and tasks
- Built-in UI descriptors to enable dynamic form generation

### File Layout
- `manifest/<engine>/*.yml`: Legacy manifests (supported)
- `manifest/<engine>/*.v1.yml`: New manifests using the v1 schema
- `manifest/shared_*.yml`: Reusable component groups referenced via `!include`
- `manifest/examples/*.yml`: Example manifests

### v1 Manifest Structure
Top-level keys:

```yaml
apiVersion: apex/v1
kind: Model
metadata:
  name: Wan 2.2 A14B Image to Video
  version: 1.0.0
  description: 14B parameter model for image-to-video generation
  tags: [wan, i2v, 14b]

spec:
  engine: wan             # required
  modelType: i2v          # required (t2v, i2v, v2v, ...)
  engineType: torch       # optional (torch, mlx)
  denoiseType: moe        # optional
  shared:                 # optional; files enabling !include aliases
    - shared_wan.yml

  components:             # required for model execution
    - type: scheduler
      base: diffusers.UniPCMultistepScheduler
      config_path: https://.../scheduler_config.json
    - type: vae
      base: wan
      model_path: Wan-AI/Wan2.2-I2V-A14B-Diffusers/vae
    - type: text_encoder
      base: T5EncoderModel
      model_path: Wan-AI/.../text_encoder
    - type: transformer
      base: wan.base
      model_path: Wan-AI/.../transformer
      tag: wan_i2v_14b_22

  preprocessors: []       # optional
  postprocessors: []      # optional

  defaults:               # optional; grouped defaults like docker-compose service config
    run:
      num_inference_steps: 30
      guidance_scale: 5.0
      return_latents: false

  loras: []               # optional; list of strings or {source|path|url, scale, name}
  save: {}                # optional; kwargs forwarded to component save

  ui:                     # optional; see UI Spec below
    mode: simple
    simple:
      inputs:
        - id: prompt
          label: Prompt
          description: Text prompt
          component: text
          default: "a woman singing on stage"
          required: true
          mapping: {target: engine.run, param: prompt}
        - id: num_inference_steps
          label: Steps
          component: slider
          min: 1
          max: 100
          step: 1
          default: 30
          mapping: {target: engine.run, param: num_inference_steps}
    advanced:
      expose: all
```

### Legacy Manifests
Legacy manifests remain supported. They do not require `apiVersion` or `kind` and may include the following root-level keys:
- `name`, `description`, `engine`, `type`, `engine_type`, `denoise_type`
- `components`, `preprocessors`, `postprocessors`, `defaults`, `loras`, `helpers`
- `shared` (for `!include` alias lookup)
- `ui`/`UI` (optional)

### Validation
v1 manifests are validated against a JSON Schema. If `jsonschema` is available, invalid manifests will raise with precise error messages. If not installed, validation is skipped with a warning.

### Includes (`!include`)
Use `shared:` (legacy) or `spec.shared` (v1) to register shared manifest files by alias. Example:

```yaml
shared:
  - shared_wan.yml   # registers alias "wan"

components:
  - !include shared:wan/vae_2.2

v1 shared component files are supported as well. Place them as `shared_<engine>.v1.yml` or `<engine>/shared.v1.yml` with structure:

```yaml
apiVersion: apex/v1
kind: SharedComponents
metadata:
  name: wan-shared
spec:
  components:
    - type: vae
      name: wan/vae_2.2
      base: wan
      model_path: Wan-AI/Wan2.2/vae
```
You can still reference them with `!include shared:wan/vae_2.2`.
```

### UI Spec
- **mode**: `simple` | `advanced` | `complex` (alias of `advanced`)
- **simple.inputs[]**: declarative UI controls mapped to runtime params
  - `id` (string, required): unique input name
  - `label` (string): display name
  - `description` (string)
  - `component` (enum): `text`, `number`, `float`, `bool`, `list`, `file`, `select`, `slider`
  - `default`, `required` (bool)
  - `options` (for select), `min`/`max`/`step` (for number/slider)
  - `mapping`: where to send the value
    - `target`: `engine.run` or `preprocessor.<name>`
    - `param`: name of the parameter to set
    - `path` (advanced): dot-path if mapping into nested kwargs
- **advanced.expose**: `all` or a list of param names to expose globally
- **advanced.inputs[]**: same shape as `simple.inputs`

At runtime, these declarations can be turned into `UINode`s using `src/ui/manifest.py` helpers, enabling a generic rendering engine.

### Authoring Guidelines
- Prefer `https://` URLs for `config_path` and model sources
- Use semantic versioning in `metadata.version`
- Provide `description` and `tags` for registry discovery
- Keep `defaults.run` reasonable for quick testing
- Favor shared components with `!include` where possible

### Migration
- Existing manifests continue to work as-is
- To migrate to v1:
  1. Wrap core fields under `apiVersion`, `kind`, `metadata`, `spec`
  2. Move `shared` to `spec.shared` (or keep at root; both are supported)
  3. Add a `ui` block if you want dynamic UI rendering
  4. Validate against the schema (install `jsonschema`)

### Naming and Tagging (Docker-style)
- File placement (recommended): `manifest/<engine>/<modelType>/` for clarity. The resolver scans all `manifest/**/*.yml` so structure is flexible.
- v1 filenames: `{slug}-{version}.v1.yml` (version optional, since `metadata.version` is authoritative). Example: `manifest/wan/i2v/wan-2-2-a14b-vid-1.0.0.v1.yml`.
- Shared files: `shared_<engine>.yml` (e.g., `shared_wan.yml`) to keep `!include shared:<engine>/...` working.

Tag references supported by the loader (no need to pass a file path):
- `engine/modelType/slug:version`
- `engine/slug:version`
- `slug:version`
- `:latest` works for each of the above forms, selecting the highest semver.

Examples:

```python
from src.engine.registry import create_engine

# Fully qualified
engine = create_engine("wan", "wan/i2v/wan-2-2-a14b:1.0.0", "i2v")

# Short with latest
engine = create_engine("wan", "wan-2-2-a14b:latest", "i2v")

# Disambiguate with engine if slug is reused across engines
engine = create_engine("wan", "wan/wan-2-2-a14b:1.0.1", "i2v")
```

Slug rules:
- Built from `metadata.name` → lowercased, spaces/punct → `-`, condensed (e.g., `Wan 2.2 A14B` → `wan-2-2-a14b`).
- Prefer concise slugs; include variant if needed (e.g., `wan-2-2-a14b-vid`).

Versioning rules:
- Set `metadata.version` using semver (`MAJOR.MINOR.PATCH` with optional pre-release, e.g., `1.2.0-rc.1`).
- `latest` resolves to highest semver across manifests with the same (engine, modelType, slug).
- Suggested bumps: breaking defaults/IO → MAJOR; additive defaults/params → MINOR; doc/typo/tuning-only → PATCH.


from src.engine import UniversalEngine
import csv
import datetime as _dt
import json
import os

json_path = "/home/tosin_coverquick_co/apex-studio/apps/api/apex-test-suite/image/flux2-dev-text-to-image-edit-1.0.0.v1.json"

def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(x) < 1024.0:
            return f"{x:.2f}{unit}"
        x /= 1024.0
    return f"{x:.2f}PiB"

def _bytes_to_mib(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)

def _infer_duration_seconds(payload: dict) -> float | None:
    # Image models won't include duration; video models may specify it directly or via frames+fps.
    for k in ("duration_s", "duration", "video_duration_s", "video_duration"):
        v = payload.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            return None

    num_frames = payload.get("num_frames")
    fps = payload.get("fps")
    if num_frames is not None and fps:
        try:
            return float(num_frames) / float(fps)
        except Exception:
            return None

    return None

engine = UniversalEngine(
    yaml_path="/home/tosin_coverquick_co/apex-studio/apps/api/new_manifest/image/flux2-dev-text-to-image-edit-1.0.0.v1.yml",
    selected_components={
        "vae": {
            "variant": "default"
        },
        "text_encoder": {
            "variant": "GGUF_Q4_K_M"
        },
        "transformer": {
            "variant": "GGUF_Q4_K_M"
        }
    },
    attention_type="sage",
    debug_component_vram=True,
    disable_text_encoder_cache=True
)

asset_path = "/home/tosin_coverquick_co/apex-studio/apps/api/apex-test-suite"
inputs = json.load(open(json_path))

for key, value in inputs.items():
    if isinstance(value, str) and "assets" in value:
        inputs[key] = os.path.join(asset_path, value)
    if isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str) and "assets" in v:
                inputs[key][i] = os.path.join(asset_path, v)

inputs["height"] = 1024
inputs["width"] = 1024


image = engine.run(
    **inputs
)

image[0].save("output_flux2_dev.png")

profile = getattr(engine, "_apex_component_vram_profile", None) or {}
items = list(profile.items())
items.sort(key=lambda kv: int(kv[1].get("peak_alloc_delta_bytes_max", 0)), reverse=True)
print("\n=== component VRAM/RAM profile (peak deltas per forward) ===")
for label, s in items:
    print(
        f"- {label}: "
        f"calls={s.get('calls')} "
        f"weights={_fmt_bytes(int(s.get('weight_bytes', 0)))} "
        f"peak_alloc+={_fmt_bytes(int(s.get('peak_alloc_delta_bytes_max', 0)))} "
        f"peak_res+={_fmt_bytes(int(s.get('peak_reserved_delta_bytes_max', 0)))} "
        f"peak_rss+={_fmt_bytes(int(s.get('rss_delta_bytes_max', 0)))} "
        f"device={s.get('device')}"
    )

# Write the profiling results to a CSV for easy analysis.
csv_path = "component_vram_ram_profile.csv"
run_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
height = inputs.get("height")
width = inputs.get("width")
duration_s = _infer_duration_seconds(inputs)
with open(csv_path, "a+", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "run_utc",
            "label",
            "device",
            "height",
            "width",
            "duration_s",
            "calls",
            "weight_bytes",
            "weight_mib",
            "peak_alloc_delta_bytes_max",
            "peak_alloc_delta_mib_max",
            "peak_reserved_delta_bytes_max",
            "peak_reserved_delta_mib_max",
            "rss_delta_bytes_max",
            "rss_delta_mib_max",
        ],
    )

    for label, s in items:
        weight_bytes = int(s.get("weight_bytes", 0) or 0)
        peak_alloc = int(s.get("peak_alloc_delta_bytes_max", 0) or 0)
        peak_reserved = int(s.get("peak_reserved_delta_bytes_max", 0) or 0)
        rss = int(s.get("rss_delta_bytes_max", 0) or 0)
        w.writerow(
            {
                "run_utc": run_utc,
                "label": label,
                "device": s.get("device"),
                "height": height,
                "width": width,
                "duration_s": "" if duration_s is None else f"{duration_s:.6f}",
                "calls": s.get("calls"),
                "weight_bytes": weight_bytes,
                "weight_mib": f"{_bytes_to_mib(weight_bytes):.3f}",
                "peak_alloc_delta_bytes_max": peak_alloc,
                "peak_alloc_delta_mib_max": f"{_bytes_to_mib(peak_alloc):.3f}",
                "peak_reserved_delta_bytes_max": peak_reserved,
                "peak_reserved_delta_mib_max": f"{_bytes_to_mib(peak_reserved):.3f}",
                "rss_delta_bytes_max": rss,
                "rss_delta_mib_max": f"{_bytes_to_mib(rss):.3f}",
            }
        )
print(f"\nWrote CSV: {csv_path}")

# Extra debug: show which components were tracked and how many profiled calls
# were observed (useful when a component is invoked via non-forward methods).
mgr = getattr(engine, "_component_memory_manager", None)
try:
    comps = getattr(mgr, "_components", None) or {}
    print("\n=== tracked components (calls observed) ===")
    for label, comp in comps.items():
        calls = getattr(comp, "forward_calls", 0)
        print(f"- {label}: calls={calls}")
except Exception:
    pass
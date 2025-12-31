#!/usr/bin/env python3
"""
Generate one "model cover" image per row in `manifest/verified/prompts.csv`
using a single text-to-image engine defined by the zimage-turbo manifest.

Typical usage:
  python scripts/generate_model_images.py

This will:
  - load `manifest/verified/image/zimage-turbo-1.0.0.v1.yml`
  - iterate all rows in `manifest/verified/prompts.csv`
  - run T2I for each prompt
  - save images to `model_images/<model_id>.png`
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image
import torch
from tqdm import tqdm

# Ensure `import src...` works when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine.registry import UniversalEngine  # noqa: E402
from src.manifest.resolver import resolve_manifest_reference  # noqa: E402


_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_filename(stem: str) -> str:
    stem = (stem or "").strip()
    stem = _FILENAME_SAFE_RE.sub("_", stem)
    stem = stem.strip("._-")
    return stem or "output"


def _coerce_first_pil(output: Any) -> Image.Image:
    """
    Engines in this repo commonly return:
      - List[PIL.Image]
      - PIL.Image
      - numpy array / torch tensor (rare for top-level run)
    This helper returns a single PIL.Image for saving.
    """
    if isinstance(output, Image.Image):
        return output

    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, Image.Image):
            return first
        # Sometimes it's nested lists (e.g. videos) — we don't support those here.
        raise TypeError(f"Unexpected list output element type: {type(first)}")

    raise TypeError(f"Unexpected engine output type: {type(output)}")


def _read_prompts_csv(csv_path: Path) -> Sequence[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            model_id = (row.get("model_id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not model_id or not prompt:
                continue
            rows.append({"model_id": model_id, "prompt": prompt})
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate images for prompts.csv using zimage-turbo manifest."
    )
    parser.add_argument(
        "--prompts-csv",
        default=str(Path("manifest/verified/prompts.csv")),
        help="Path to prompts CSV with columns: model_id,prompt",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("manifest/verified/image/zimage-turbo-1.0.0.v1.yml")),
        help="Manifest YAML path (or manifest reference) for the T2I engine to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_images",
        help="Directory to write images into.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override num_inference_steps (otherwise manifest defaults apply).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Each row uses (seed + row_index) for determinism.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to repo default device.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of rows to run (for quick tests).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale for the T2I engine.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose output file already exists.",
    )
    args = parser.parse_args()

    prompts_csv = Path(args.prompts_csv)
    if not prompts_csv.exists():
        raise FileNotFoundError(f"prompts CSV not found: {prompts_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve manifest reference (e.g. "zimage-turbo") to an absolute YAML file path if possible.
    manifest_ref = str(args.manifest)
    manifest_path = resolve_manifest_reference(manifest_ref) or manifest_ref

    device = torch.device(args.device) if args.device else None

    # Some configs check compute capability; allow opting into CPU explicitly.
    if device is not None:
        engine = UniversalEngine(yaml_path=manifest_path, device=device)
    else:
        engine = UniversalEngine(yaml_path=manifest_path)

    rows = list(_read_prompts_csv(prompts_csv))
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]

    # Helps keep VRAM use a little more predictable in batch generation.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for idx, row in tqdm(enumerate(rows)):
        tqdm.write(f"Processing row {idx+1}/{len(rows)}")
        model_id = row["model_id"]
        prompt = row["prompt"]

        filename = _safe_filename(model_id) + ".png"
        out_path = output_dir / filename

        if args.skip_existing and out_path.exists():
            continue

        run_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": int(args.height),
            "width": int(args.width),
            "seed": int(args.seed) + int(idx),
            "guidance_scale": float(args.guidance_scale),
            "offload": False,
        }

        if args.num_inference_steps is not None:
            run_kwargs["num_inference_steps"] = int(args.num_inference_steps)

        try:
            out = engine.run(**run_kwargs)
            img = _coerce_first_pil(out)
            img.save(out_path)
            print(f"[{idx+1}/{len(rows)}] saved {out_path}")
        except Exception as e:
            traceback.print_exc()
            print(f"[{idx+1}/{len(rows)}] FAILED model_id={model_id!r}: {e}")

    # Optional: offload at end.
    try:
        engine.offload_engine()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

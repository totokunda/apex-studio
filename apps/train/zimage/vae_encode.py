import argparse
import csv
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "apps" / "api"
sys.path.append(str(API_DIR))

from src.engine import UniversalEngine  # noqa: E402


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / "training_inputs"
    default_yaml = API_DIR / "manifest" / "image" / "zimage-1.0.0.v1.yml"

    p = argparse.ArgumentParser(description="Encode images referenced in captions.csv into a safetensors file.")
    p.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        type=Path,
        required=True,
        help="Dataset folder containing captions.csv",
    )
    p.add_argument(
        "--captions-csv",
        dest="captions_csv",
        type=Path,
        default=None,
        help="Path to captions.csv. Defaults to <dataset-dir>/captions.csv",
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=default_out_dir,
        help="Directory to write encodings into.",
    )
    p.add_argument(
        "--out-file",
        dest="out_file",
        type=str,
        default="vae_encodings.safetensors",
        help="Output filename (written inside --out-dir).",
    )
    p.add_argument(
        "--yaml-path",
        dest="yaml_path",
        type=Path,
        default=default_yaml,
        help="Path to the API engine YAML manifest.",
    )
    p.add_argument(
        "--max-area",
        dest="max_area",
        type=int,
        default=720 * 1280,
        help="Max pixel area for aspect ratio resize.",
    )
    p.add_argument(
        "--mod-value",
        dest="mod_value",
        type=int,
        default=16,
        help="Mod value for image resize.",
    )
    p.add_argument(
        "--vae-offload",
        dest="vae_offload",
        action="store_true",
        help="Enable VAE offloading (passed as offload=True).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    captions_csv = args.captions_csv
    if captions_csv is None:
        captions_csv = dataset_dir / "captions.csv"
    captions_csv = captions_csv.expanduser().resolve()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_file

    engine = UniversalEngine(yaml_path=str(args.yaml_path), components_to_load=["vae"])

    out_dict: dict[str, torch.Tensor] = {}
    with open(captions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Encoding images"):
            image_path = row["image_path"]
            image = engine._load_image(image_path)
            image = engine._aspect_ratio_resize(image, max_area=args.max_area, mod_value=args.mod_value)[0]
            image_tensor = engine.image_processor.preprocess(image)
            vae_image = engine.vae_encode(image_tensor, offload=args.vae_offload)[0]
            out_dict[image_path] = vae_image.detach().cpu()

    save_file(out_dict, str(out_path))


if __name__ == "__main__":
    main()
        
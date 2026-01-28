import argparse
import csv
import os
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
PROMPT = "Write a brief caption for this image in a formal tone."


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Generate captions for a folder of images.")
    p.add_argument(
        "--dataset-dir",
        "--images-dir",
        dest="dataset_dir",
        type=Path,
        required=True,
        help="Folder containing images to caption.",
    )
    p.add_argument(
        "--out-csv",
        dest="out_csv",
        type=Path,
        default=None,
        help="Where to write captions CSV. Defaults to <dataset-dir>/captions.csv",
    )
    p.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="*",
        help="Glob pattern (relative to dataset dir) to select images.",
    )
    p.add_argument("--model", dest="model_name", type=str, default=MODEL_NAME, help="HF model name.")
    p.add_argument("--prompt", dest="prompt", type=str, default=PROMPT, help="Prompt to use for captioning.")
    p.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate per image.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"Dataset dir does not exist or is not a directory: {dataset_dir}")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = dataset_dir / "captions.csv"
    out_csv = out_csv.expanduser().resolve()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # JoyCaption's LLM is native bfloat16; fall back to fp16 if bf16 isn't supported.
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(args.model_name)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map=0 if use_cuda else None,
    )
    llava_model.eval()

    image_paths = sorted(glob(str(dataset_dir / args.glob_pattern)))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
        writer.writeheader()

        for image_path in tqdm(image_paths, desc="Generating captions", total=len(image_paths)):
            if not os.path.isfile(image_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                # Skip non-images (or corrupted files) silently.
                continue

            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": args.prompt},
            ]

            # NOTE: HF chat templating with Llava models is fragile; this specific combination is known-good.
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

            device = "cuda" if use_cuda else "cpu"
            inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

            with torch.inference_mode():
                generate_ids = llava_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    suppress_tokens=None,
                    use_cache=True,
                    temperature=0.6,
                    top_k=None,
                    top_p=0.9,
                )[0]

            # Trim off the prompt
            generate_ids = generate_ids[inputs["input_ids"].shape[1] :]
            caption = processor.tokenizer.decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            writer.writerow({"image_path": image_path, "caption": caption})
            print(f"{image_path}\t{caption}")


if __name__ == "__main__":
    main()


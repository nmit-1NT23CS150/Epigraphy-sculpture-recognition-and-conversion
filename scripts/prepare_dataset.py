from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.text import Vocabulary, normalize_target


DATASET_ID = "historyHulk/MoDeTrans"
DATASET_URL = (
    "https://huggingface.co/datasets/historyHulk/MoDeTrans/resolve/main/"
    "data/train-00000-of-00001.parquet?download=true"
)


def split_for_group(group_id: int, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def download_parquet(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    with requests.get(DATASET_URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(path)


def decode_image(value: object) -> Image.Image:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw))
        source_path = value.get("path")
        if source_path:
            return Image.open(source_path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value)))
    raise TypeError(f"Unsupported Parquet image representation: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    parquet_path = args.output / "raw" / "modetrans.parquet"
    download_parquet(parquet_path)
    dataset = pd.read_parquet(parquet_path)
    if args.max_samples is not None:
        dataset = dataset.iloc[: args.max_samples].copy()

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, row in dataset.iterrows():
        original_name = str(row.get("filename") or f"{index + 1}.jpg")
        stem_digits = "".join(char for char in Path(original_name).stem if char.isdigit())
        numeric_id = int(stem_digits) if stem_digits else index + 1
        group_id = (numeric_id - 1) // args.group_size
        filename = f"mode_{numeric_id:05d}.png"
        image_path = image_dir / filename
        with decode_image(row["image"]) as image:
            image.convert("L").save(image_path, format="PNG", optimize=True)
        rows.append(
            {
                "filename": filename,
                "original_filename": original_name,
                "source_group": group_id,
                "split": split_for_group(group_id, args.seed),
                "image_path": str(image_path.resolve()),
                "text": normalize_target(str(row["text"])),
            }
        )

    table = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    manifest_path = args.output / "manifest.csv"
    table.to_csv(manifest_path, index=False, encoding="utf-8")
    vocabulary = Vocabulary.from_texts(table["text"].tolist())
    vocabulary.save(args.output / "vocabulary.json")
    summary = {
        "dataset_id": DATASET_ID,
        "dataset_license": "MIT (as declared by the dataset card)",
        "samples": len(table),
        "source_group_rule": f"floor((numeric_filename_id - 1) / {args.group_size})",
        "split_seed": args.seed,
        "split_counts": table["split"].value_counts().sort_index().to_dict(),
        "vocabulary_size_including_special_tokens": len(vocabulary.tokens),
        "target_length_min": int(table["text"].str.len().min()),
        "target_length_max": int(table["text"].str.len().max()),
        "target_length_mean": float(table["text"].str.len().mean()),
        "warning": "Public metadata lacks reliable manuscript IDs; the split is block-disjoint, not proven document-disjoint.",
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

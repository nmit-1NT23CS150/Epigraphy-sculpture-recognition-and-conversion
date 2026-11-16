from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.data import ImageShape
from modi_transliterator.metrics import sequence_metrics
from modi_transliterator.runtime import load_checkpoint, make_loader, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generation-length", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, vocabulary, model = load_checkpoint(args.checkpoint, device)
    height, width = checkpoint["metadata"]["image_shape"]
    loader = make_loader(
        args.manifest,
        args.split,
        vocabulary,
        ImageShape(height, width),
        int(checkpoint["model_config"]["max_text_length"]),
        args.batch_size,
        args.workers,
        False,
        args.max_samples,
    )
    references: list[str] = []
    hypotheses: list[str] = []
    filenames: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            predictions = model.generate(
                images, vocabulary.bos_id, vocabulary.eos_id, args.generation_length
            )
            references.extend(batch["texts"])
            hypotheses.extend(vocabulary.decode(row.tolist()) for row in predictions.cpu())
            filenames.extend(batch["filenames"])
    metrics = sequence_metrics(references, hypotheses)
    output_dir = args.checkpoint.parent / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filename", "reference", "prediction"))
        writer.writeheader()
        writer.writerows(
            {"filename": name, "reference": reference, "prediction": prediction}
            for name, reference, prediction in zip(filenames, references, hypotheses, strict=True)
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

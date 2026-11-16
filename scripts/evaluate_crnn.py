from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.crnn import CTCVocabulary, ModiLineCRNN, ModiLineDataset, collate_lines
from modi_transliterator.metrics import sequence_metrics
from modi_transliterator.runtime import save_json


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("pretrained") is not False or metadata.get("initialization") != "random":
        raise ValueError("Checkpoint is not certified as a random-initialized local model")
    vocabulary = CTCVocabulary(tuple(checkpoint["vocabulary"]))
    model = ModiLineCRNN(vocabulary.size_with_blank, **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint, vocabulary, model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "line_manifest.csv")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, vocabulary, model = load_model(args.checkpoint, device)
    dataset = ModiLineDataset(
        args.manifest,
        args.split,
        vocabulary,
        int(checkpoint["metadata"]["image_height"]),
        int(checkpoint["metadata"]["max_width"]),
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_lines)
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device, non_blocking=True))
            predictions = logits.argmax(dim=-1).transpose(0, 1).cpu()
            decoded = [vocabulary.decode_greedy(row.tolist()) for row in predictions]
            records.extend(
                {
                    "parent_filename": parent,
                    "line_index": line_index,
                    "line_filename": line_filename,
                    "reference": reference,
                    "prediction": prediction,
                }
                for parent, line_index, line_filename, reference, prediction in zip(
                    batch["parent_filenames"], batch["line_indices"], batch["line_filenames"],
                    batch["texts"], decoded, strict=True
                )
            )
    line_metrics = sequence_metrics(
        [str(row["reference"]) for row in records], [str(row["prediction"]) for row in records]
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row["parent_filename"])].append(row)
    parent_references: list[str] = []
    parent_predictions: list[str] = []
    for parent in sorted(grouped):
        ordered = sorted(grouped[parent], key=lambda row: int(row["line_index"]))
        parent_references.append(" ".join(str(row["reference"]) for row in ordered))
        parent_predictions.append(" ".join(str(row["prediction"]) for row in ordered))
    segment_metrics = sequence_metrics(parent_references, parent_predictions)
    metrics = {"line_level": line_metrics, "reconstructed_segment_level": segment_metrics}
    output_dir = args.checkpoint.parent / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

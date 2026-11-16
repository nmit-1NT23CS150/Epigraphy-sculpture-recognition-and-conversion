from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.data import ImageShape
from modi_transliterator.metrics import sequence_metrics
from modi_transliterator.runtime import DEFAULT_MODEL_CONFIG, make_loader, make_model, save_json, seed_everything
from modi_transliterator.text import Vocabulary


@torch.no_grad()
def validate(model, loader, vocabulary, device, generation_length: int) -> dict[str, float | int]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        predicted = model.generate(
            images,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
            max_length=generation_length,
        )
        references.extend(batch["texts"])
        hypotheses.extend(vocabulary.decode(row.tolist()) for row in predicted.cpu())
    return sequence_metrics(references, hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "data" / "vocabulary.json")
    parser.add_argument("--run-name", default="main_seed_2026")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--max-text-length", type=int, default=384)
    parser.add_argument("--generation-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()

    if not args.manifest.exists() or not args.vocabulary.exists():
        raise SystemExit("Dataset is not prepared. Run scripts/prepare_dataset.py first.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = Vocabulary.from_json(args.vocabulary)
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config["max_text_length"] = args.max_text_length
    model = make_model(vocabulary, model_config).to(device)
    shape = ImageShape(args.image_height, args.image_width)
    train_loader = make_loader(
        args.manifest, "train", vocabulary, shape, args.max_text_length,
        args.batch_size, args.workers, True, args.max_train_samples,
    )
    validation_loader = make_loader(
        args.manifest, "validation", vocabulary, shape, args.max_text_length,
        args.batch_size, args.workers, False, args.max_eval_samples,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = PROJECT_ROOT / "outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_similarity = -1.0

    metadata = {
        "project": "Modi manuscript to Devanagari transliteration",
        "architecture": "from-scratch compact CNN-Transformer encoder-decoder",
        "dataset_id": "historyHulk/MoDeTrans",
        "pretrained": False,
        "initialization": "random",
        "seed": args.seed,
        "image_shape": [shape.height, shape.width],
        "created_by": "scripts/train.py",
    }
    save_json(output_dir / "run_config.json", {"metadata": metadata, "arguments": vars(args), "model_config": model_config})

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        started = time.time()
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images, inputs)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())

        metrics = validate(model, validation_loader, vocabulary, device, args.generation_length)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)
        print(json.dumps(record, ensure_ascii=False))
        similarity = float(metrics["mean_normalized_edit_similarity"])
        if similarity > best_similarity:
            best_similarity = similarity
            torch.save(
                {
                    "metadata": metadata,
                    "model_config": model_config,
                    "model_state": model.state_dict(),
                    "vocabulary": list(vocabulary.tokens),
                    "best_validation_metrics": metrics,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

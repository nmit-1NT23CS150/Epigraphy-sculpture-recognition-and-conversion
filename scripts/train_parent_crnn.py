from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.crnn import CTCVocabulary, ModiLineCRNN, ModiParentCTCDataset, collate_lines
from modi_transliterator.metrics import sequence_metrics
from modi_transliterator.runtime import save_json, seed_everything


def make_loader(args, split, vocabulary, shuffle, max_samples):
    dataset = ModiParentCTCDataset(
        args.manifest, split, vocabulary, args.image_height, args.max_width, max_samples=max_samples
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_lines,
    )


@torch.no_grad()
def evaluate(model, loader, vocabulary, device):
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in loader:
        logits = model(batch["images"].to(device, non_blocking=True))
        predictions = logits.argmax(dim=-1).transpose(0, 1).cpu()
        references.extend(batch["texts"])
        hypotheses.extend(vocabulary.decode_greedy(row.tolist()) for row in predictions)
    return sequence_metrics(references, hypotheses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "data" / "ctc_vocabulary.json")
    parser.add_argument("--run-name", default="parent_crnn_seed_2026")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = CTCVocabulary.from_json(args.vocabulary)
    model_config = {
        "recurrent_hidden": args.hidden_size,
        "recurrent_layers": args.recurrent_layers,
        "dropout": 0.2,
    }
    model = ModiLineCRNN(vocabulary.size_with_blank, **model_config).to(device)
    train_loader = make_loader(args, "train", vocabulary, True, args.max_train_samples)
    validation_loader = make_loader(args, "validation", vocabulary, False, args.max_eval_samples)
    criterion = nn.CTCLoss(blank=vocabulary.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = PROJECT_ROOT / "outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "project": "Matched whole-parent CRNN-CTC ablation for Modi transliteration",
        "architecture": "same from-scratch CRNN with CTC as the line condition",
        "dataset_id": "historyHulk/MoDeTrans",
        "input_unit": "whole parent segment",
        "labels": "released parent-level Devanagari targets",
        "matched_to": "crnn_main_seed_2026 settings except max_width=2048 and input unit",
        "pretrained": False,
        "initialization": "random",
        "seed": args.seed,
        "image_height": args.image_height,
        "max_width": args.max_width,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(output_dir / "run_config.json", {"metadata": metadata, "arguments": vars(args), "model_config": model_config})
    history: list[dict[str, object]] = []
    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        started = time.time()
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            widths = batch["widths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                log_probs = model(images)
                input_lengths = model.output_lengths(widths).clamp(max=log_probs.shape[0])
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        metrics = evaluate(model, validation_loader, vocabulary, device)
        scheduler.step(float(metrics["character_error_rate"]))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)
        print(json.dumps(record), flush=True)
        cer = float(metrics["character_error_rate"])
        if cer < best_cer:
            best_cer = cer
            torch.save(
                {
                    "metadata": metadata,
                    "model_config": model_config,
                    "model_state": model.state_dict(),
                    "vocabulary": list(vocabulary.characters),
                    "best_validation_metrics": metrics,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

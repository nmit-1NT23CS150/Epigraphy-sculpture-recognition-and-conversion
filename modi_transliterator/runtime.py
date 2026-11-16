from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageShape, ModiManifestDataset, collate_batch
from .model import ModiManuscriptTransformer
from .text import Vocabulary


DEFAULT_MODEL_CONFIG = {
    "dimension": 192,
    "heads": 6,
    "encoder_layers": 2,
    "decoder_layers": 3,
    "feedforward_dimension": 512,
    "dropout": 0.1,
    "image_grid": [6, 24],
    "max_text_length": 384,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(vocabulary: Vocabulary, config: dict[str, object]) -> ModiManuscriptTransformer:
    return ModiManuscriptTransformer(
        vocabulary_size=len(vocabulary.tokens),
        pad_id=vocabulary.pad_id,
        dimension=int(config["dimension"]),
        heads=int(config["heads"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_layers=int(config["decoder_layers"]),
        feedforward_dimension=int(config["feedforward_dimension"]),
        dropout=float(config["dropout"]),
        image_grid=tuple(config["image_grid"]),
        max_text_length=int(config["max_text_length"]),
    )


def make_loader(
    manifest_path: Path,
    split: str,
    vocabulary: Vocabulary,
    image_shape: ImageShape,
    max_text_length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    max_samples: int | None = None,
) -> DataLoader:
    dataset = ModiManifestDataset(
        manifest_path,
        split,
        vocabulary,
        image_shape,
        max_text_length,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_batch, pad_id=vocabulary.pad_id),
    )


def save_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, object], Vocabulary, ModiManuscriptTransformer]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("pretrained") is not False or metadata.get("initialization") != "random":
        raise ValueError("Checkpoint does not certify random, non-pretrained initialization")
    vocabulary = Vocabulary(tuple(checkpoint["vocabulary"]))
    model = make_model(vocabulary, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return checkpoint, vocabulary, model

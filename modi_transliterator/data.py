from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from .text import Vocabulary, normalize_target


@dataclass(frozen=True)
class ImageShape:
    height: int = 192
    width: int = 768


def prepare_image(image: Image.Image, shape: ImageShape) -> torch.Tensor:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.1)
    width, height = gray.size
    scale = min(shape.width / max(width, 1), shape.height / max(height, 1))
    resized = gray.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (shape.width, shape.height), 255)
    offset = ((shape.width - resized.width) // 2, (shape.height - resized.height) // 2)
    canvas.paste(resized, offset)
    array = torch.from_numpy(__import__("numpy").asarray(canvas, dtype="float32").copy())
    return (array.unsqueeze(0) / 127.5) - 1.0


class ModiManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        vocabulary: Vocabulary,
        shape: ImageShape,
        max_text_length: int,
        max_samples: int | None = None,
    ) -> None:
        table = pd.read_csv(manifest_path, encoding="utf-8")
        table = table.loc[table["split"] == split].sort_values("filename").reset_index(drop=True)
        if max_samples is not None:
            table = table.iloc[:max_samples].copy()
        self.table = table
        self.vocabulary = vocabulary
        self.shape = shape
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.table.iloc[index]
        path = Path(str(row["image_path"]))
        with Image.open(path) as image:
            tensor = prepare_image(image, self.shape)
        text = normalize_target(str(row["text"]))
        token_ids = self.vocabulary.encode(text, max_length=self.max_text_length)
        return {
            "image": tensor,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "filename": str(row["filename"]),
        }


def collate_batch(rows: list[dict[str, object]], pad_id: int) -> dict[str, object]:
    images = torch.stack([row["image"] for row in rows])
    tokens = torch.nn.utils.rnn.pad_sequence(
        [row["tokens"] for row in rows], batch_first=True, padding_value=pad_id
    )
    return {
        "images": images,
        "tokens": tokens,
        "texts": [str(row["text"]) for row in rows],
        "filenames": [str(row["filename"]) for row in rows],
    }

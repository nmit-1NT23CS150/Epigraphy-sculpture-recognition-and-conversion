from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import Dataset

from .text import normalize_target


@dataclass(frozen=True)
class CTCVocabulary:
    characters: tuple[str, ...]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CTCVocabulary":
        return cls(tuple(sorted({character for text in texts for character in normalize_target(text)})))

    @classmethod
    def from_json(cls, path: Path) -> "CTCVocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(payload["characters"]))

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def size_with_blank(self) -> int:
        return len(self.characters) + 1

    @property
    def stoi(self) -> dict[str, int]:
        return {character: index + 1 for index, character in enumerate(self.characters)}

    def encode(self, text: str) -> list[int]:
        mapping = self.stoi
        return [mapping[character] for character in normalize_target(text)]

    def decode_greedy(self, ids: Iterable[int]) -> str:
        output: list[str] = []
        previous = self.blank_id
        for raw_index in ids:
            index = int(raw_index)
            if index != self.blank_id and index != previous:
                output.append(self.characters[index - 1])
            previous = index
        return normalize_target("".join(output))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"characters": list(self.characters)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prepare_line_image(
    image: Image.Image,
    height: int,
    max_width: int,
    minimum_width: int = 32,
) -> tuple[torch.Tensor, int]:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    gray = ImageEnhance.Contrast(gray).enhance(1.1)
    scale = height / max(gray.height, 1)
    resized_width = max(minimum_width, min(max_width, int(round(gray.width * scale))))
    resized = gray.resize((resized_width, height), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32).copy()
    tensor = (torch.from_numpy(array).unsqueeze(0) / 127.5) - 1.0
    return tensor, resized_width


class ModiLineDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        vocabulary: CTCVocabulary,
        image_height: int,
        max_width: int,
        max_samples: int | None = None,
    ) -> None:
        table = pd.read_csv(manifest_path, encoding="utf-8")
        table = table.loc[table["split"] == split].sort_values(["parent_filename", "line_index"])
        if max_samples is not None:
            table = table.iloc[:max_samples]
        self.table = table.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.table.iloc[index]
        text = normalize_target(str(row["text"]))
        encoded = self.vocabulary.encode(text)
        repeated = sum(left == right for left, right in zip(encoded, encoded[1:]))
        minimum_steps = len(encoded) + repeated + 2
        minimum_width = min(self.max_width, max(32, minimum_steps * 4))
        with Image.open(Path(str(row["image_path"]))) as image:
            tensor, width = prepare_line_image(
                image, self.image_height, self.max_width, minimum_width=minimum_width
            )
        return {
            "image": tensor,
            "width": width,
            "targets": torch.tensor(encoded, dtype=torch.long),
            "text": text,
            "parent_filename": str(row["parent_filename"]),
            "line_index": int(row["line_index"]),
            "line_filename": str(row["line_filename"]),
        }


class ModiParentCTCDataset(Dataset):
    """Whole-parent counterpart to ``ModiLineDataset`` for a matched CTC ablation.

    It deliberately reuses the same vocabulary, image preparation, CTC feasibility
    padding, model, and collation path as the line condition.  Only the manifest
    unit changes from detected line crops to original parent segments.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        vocabulary: CTCVocabulary,
        image_height: int,
        max_width: int,
        max_samples: int | None = None,
    ) -> None:
        table = pd.read_csv(manifest_path, encoding="utf-8")
        table = table.loc[table["split"] == split].sort_values("filename")
        if max_samples is not None:
            table = table.iloc[:max_samples]
        self.table = table.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.table.iloc[index]
        text = normalize_target(str(row["text"]))
        encoded = self.vocabulary.encode(text)
        repeated = sum(left == right for left, right in zip(encoded, encoded[1:]))
        minimum_steps = len(encoded) + repeated + 2
        required_width = minimum_steps * 4
        if required_width > self.max_width:
            raise ValueError(
                f"{row['filename']} needs {minimum_steps} CTC steps "
                f"({required_width}px at stride 4), exceeding max_width={self.max_width}"
            )
        minimum_width = max(32, required_width)
        with Image.open(Path(str(row["image_path"]))) as image:
            tensor, width = prepare_line_image(
                image, self.image_height, self.max_width, minimum_width=minimum_width
            )
        return {
            "image": tensor,
            "width": width,
            "targets": torch.tensor(encoded, dtype=torch.long),
            "text": text,
            "parent_filename": str(row["filename"]),
            "line_index": 0,
            "line_filename": str(row["filename"]),
        }


def collate_lines(rows: list[dict[str, object]]) -> dict[str, object]:
    maximum_width = max(int(row["width"]) for row in rows)
    images = torch.ones(len(rows), 1, rows[0]["image"].shape[1], maximum_width)
    for index, row in enumerate(rows):
        tensor = row["image"]
        images[index, :, :, : tensor.shape[-1]] = tensor
    targets = torch.cat([row["targets"] for row in rows])
    return {
        "images": images,
        "widths": torch.tensor([int(row["width"]) for row in rows], dtype=torch.long),
        "targets": targets,
        "target_lengths": torch.tensor([len(row["targets"]) for row in rows], dtype=torch.long),
        "texts": [str(row["text"]) for row in rows],
        "parent_filenames": [str(row["parent_filename"]) for row in rows],
        "line_indices": [int(row["line_index"]) for row in rows],
        "line_filenames": [str(row["line_filename"]) for row in rows],
    }


class ModiLineCRNN(nn.Module):
    def __init__(
        self,
        vocabulary_size_with_blank: int,
        recurrent_hidden: int = 192,
        recurrent_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.sequence = nn.GRU(
            input_size=256,
            hidden_size=recurrent_hidden,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(recurrent_hidden * 2, vocabulary_size_with_blank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images).mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence).log_softmax(dim=-1).transpose(0, 1)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths + 3, 4, rounding_mode="floor")

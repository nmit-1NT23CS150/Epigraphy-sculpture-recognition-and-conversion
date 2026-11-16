from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def normalize_target(text: str) -> str:
    """Normalize a Devanagari transcription without changing its words."""
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    return text.strip()


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "Vocabulary":
        characters = sorted({char for text in texts for char in normalize_target(text)})
        return cls(tuple(SPECIAL_TOKENS) + tuple(characters))

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(payload["tokens"]))

    @property
    def stoi(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        mapping = self.stoi
        ids = [self.bos_id]
        ids.extend(mapping.get(char, self.unk_id) for char in normalize_target(text))
        ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
            ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        output: list[str] = []
        for index in ids:
            token = self.tokens[int(index)]
            if token == "<eos>":
                break
            if token in {"<pad>", "<bos>"}:
                continue
            output.append("�" if token == "<unk>" else token)
        return normalize_target("".join(output))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"tokens": list(self.tokens)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosition(nn.Module):
    def __init__(self, dimension: int, max_length: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.shape[1]]


class ModiManuscriptTransformer(nn.Module):
    """Compact image-to-Devanagari model with randomly initialized parameters."""

    def __init__(
        self,
        vocabulary_size: int,
        pad_id: int,
        dimension: int = 192,
        heads: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        feedforward_dimension: int = 512,
        dropout: float = 0.1,
        image_grid: tuple[int, int] = (6, 24),
        max_text_length: int = 512,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dimension = dimension
        self.max_text_length = max_text_length
        self.image_grid = image_grid
        self.visual = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, dimension, 3, stride=2, padding=1),
            nn.BatchNorm2d(dimension),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(image_grid),
        )
        self.image_position = SinusoidalPosition(dimension, max_length=image_grid[0] * image_grid[1])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.token_embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=pad_id)
        self.text_position = SinusoidalPosition(dimension, max_length=max_text_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(dimension, vocabulary_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        features = features.flatten(2).transpose(1, 2)
        return self.encoder(self.image_position(features))

    def decode_tokens(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        embedded = self.text_position(self.token_embedding(tokens) * math.sqrt(self.dimension))
        length = tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = tokens.eq(self.pad_id)
        decoded = self.decoder(
            embedded,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.output(decoded)

    def forward(self, images: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.decode_tokens(input_tokens, self.encode_image(images))

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int,
    ) -> torch.Tensor:
        memory = self.encode_image(images)
        generated = torch.full(
            (images.shape[0], 1), bos_id, dtype=torch.long, device=images.device
        )
        finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        for _ in range(max_length - 1):
            logits = self.decode_tokens(generated, memory)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return generated

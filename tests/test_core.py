from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.metrics import edit_distance, sequence_metrics
from modi_transliterator.model import ModiManuscriptTransformer
from modi_transliterator.crnn import CTCVocabulary, ModiLineCRNN
from modi_transliterator.line_segmentation import detect_line_regions, split_target_across_lines
from modi_transliterator.text import Vocabulary, normalize_target


def test_text_round_trip() -> None:
    text = "मोदी  लिपी\nमराठी"
    normalized = normalize_target(text)
    vocabulary = Vocabulary.from_texts([normalized])
    assert vocabulary.decode(vocabulary.encode(text)) == "मोदी लिपी मराठी"


def test_metrics() -> None:
    assert edit_distance("मराठी", "मराटी") == 1
    metrics = sequence_metrics(["अ", "क"], ["अ", "ख"])
    assert metrics["samples"] == 2
    assert metrics["exact_sequence_match"] == 0.5


def test_model_forward_and_generate() -> None:
    vocabulary = Vocabulary.from_texts(["मोदी"])
    model = ModiManuscriptTransformer(
        vocabulary_size=len(vocabulary.tokens),
        pad_id=vocabulary.pad_id,
        dimension=48,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dimension=96,
        image_grid=(2, 4),
        max_text_length=16,
    )
    images = torch.randn(2, 1, 64, 128)
    tokens = torch.tensor([vocabulary.encode("मो")[:-1], vocabulary.encode("दी")[:-1]])
    logits = model(images, tokens)
    assert logits.shape == (2, tokens.shape[1], len(vocabulary.tokens))
    generated = model.generate(images[:1], vocabulary.bos_id, vocabulary.eos_id, max_length=4)
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 4


def test_line_detection_and_target_split() -> None:
    image = Image.new("L", (400, 180), 255)
    draw = ImageDraw.Draw(image)
    for y in (35, 90, 145):
        draw.line((20, y, 380, y), fill=0, width=4)
        draw.rectangle((40, y, 70, y + 18), fill=0)
    regions = detect_line_regions(image)
    assert len(regions) == 3
    labels = split_target_across_lines("एक दोन तीन चार पाच सहा", [1, 1, 1])
    assert len(labels) == 3
    assert " ".join(labels) == "एक दोन तीन चार पाच सहा"


def test_crnn_forward_and_ctc_vocabulary() -> None:
    vocabulary = CTCVocabulary.from_texts(["मोदी लिपी"])
    model = ModiLineCRNN(vocabulary.size_with_blank, recurrent_hidden=32, recurrent_layers=1)
    output = model(torch.randn(2, 1, 64, 256))
    assert output.shape[1] == 2
    assert output.shape[2] == vocabulary.size_with_blank
    assert vocabulary.encode("मोदी")

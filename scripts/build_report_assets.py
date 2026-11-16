from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "report_assets"
NAVY = "#17324D"
BLUE = "#0B6E99"
GREEN = "#2D8A56"
RED = "#C13B2A"
GRID = "#D9E1E7"
MUTED = "#607080"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 32), title, fill=NAVY, font=font(28, True), anchor="ma")
    return image, draw


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_max: float, ticks: int = 5) -> None:
    left, top, right, bottom = box
    for index in range(ticks + 1):
        y = bottom - (bottom - top) * index / ticks
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 12, y), f"{y_max * index / ticks:.1f}", fill=MUTED, font=font(16), anchor="rm")
    draw.line((left, top, left, bottom), fill=NAVY, width=2)
    draw.line((left, bottom, right, bottom), fill=NAVY, width=2)


def line_plot(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], values: list[float], y_max: float, colour: str) -> None:
    left, top, right, bottom = box
    points = []
    for index, value in enumerate(values):
        x = left + (right - left) * index / max(1, len(values) - 1)
        y = bottom - (bottom - top) * value / y_max
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=colour, width=4, joint="curve")


def training_curves() -> None:
    history = json.loads((PROJECT_ROOT / "outputs" / "crnn_main_seed_2026" / "history.json").read_text())
    loss = [row["train_loss"] for row in history]
    cer = [row["character_error_rate"] for row in history]
    similarity = [row["mean_normalized_edit_similarity"] for row in history]
    image, draw = canvas(1800, 700, "CRNN-CTC Training and Validation Curves")
    left_box = (120, 130, 830, 570)
    right_box = (1010, 130, 1720, 570)
    axes(draw, left_box, max(loss) * 1.05)
    axes(draw, right_box, 1.0)
    line_plot(draw, left_box, loss, max(loss) * 1.05, BLUE)
    line_plot(draw, right_box, cer, 1.0, RED)
    line_plot(draw, right_box, similarity, 1.0, GREEN)
    draw.text((475, 92), "Training CTC loss", fill=NAVY, font=font(22, True), anchor="ma")
    draw.text((1365, 92), "Validation sequence metrics", fill=NAVY, font=font(22, True), anchor="ma")
    draw.text((475, 626), "Epoch (1-25)", fill=MUTED, font=font(18), anchor="ma")
    draw.text((1365, 626), "Epoch (1-25)", fill=MUTED, font=font(18), anchor="ma")
    draw.line((1120, 666, 1170, 666), fill=RED, width=4)
    draw.text((1182, 666), "Validation CER", fill=NAVY, font=font(17), anchor="lm")
    draw.line((1370, 666, 1420, 666), fill=GREEN, width=4)
    draw.text((1432, 666), "Normalized edit similarity", fill=NAVY, font=font(17), anchor="lm")
    image.save(ASSET_DIR / "training_curves.png")


def model_comparison() -> None:
    image, draw = canvas(1500, 690, "Held-out Segment-Level Model Comparison")
    panels = [((110, 140, 680, 555), "Character error rate (lower is better)", [0.8793825437, 0.4771601390]),
              ((820, 140, 1390, 555), "Normalized edit similarity (higher is better)", [0.2340591004, 0.4704467281])]
    labels = ["Whole-segment\nTransformer", "Line CRNN +\nreconstruction"]
    colours = ["#A8B8C8", BLUE]
    for box, subtitle, values in panels:
        axes(draw, box, 1.0)
        left, top, right, bottom = box
        centres = [left + (right - left) * 0.32, left + (right - left) * 0.72]
        for index, (centre, value) in enumerate(zip(centres, values)):
            y = bottom - (bottom - top) * value
            draw.rounded_rectangle((centre - 60, y, centre + 60, bottom), radius=8, fill=colours[index])
            draw.text((centre, y - 14), f"{value:.3f}", fill=NAVY, font=font(20, True), anchor="ms")
            draw.multiline_text((centre, bottom + 18), labels[index], fill=NAVY, font=font(17), anchor="ma", align="center", spacing=3)
        draw.text(((left + right) // 2, 102), subtitle, fill=NAVY, font=font(20, True), anchor="ma")
    image.save(ASSET_DIR / "model_comparison.png")


def pipeline_diagram() -> None:
    image, draw = canvas(1900, 520, "Line-Segmentation and CRNN-CTC Transliteration Pipeline")
    labels = ["Modi manuscript\nsegment", "Line detection\nand crop", "CNN visual\nfeatures", "Bidirectional\nGRU", "CTC decoding", "Devanagari\nsequence"]
    colours = ["#E8F1F5", "#DCECF4", "#CBE1EC", "#B7D5E4", "#A4C9DB", "#D9EBDD"]
    start, gap, width, top, height = 48, 42, 260, 150, 155
    for index, (label, colour) in enumerate(zip(labels, colours)):
        x = start + index * (width + gap)
        draw.rounded_rectangle((x, top, x + width, top + height), radius=18, fill=colour, outline="#31576A", width=3)
        draw.multiline_text((x + width // 2, top + height // 2), label, fill=NAVY, font=font(22, True), anchor="mm", align="center", spacing=5)
        if index < len(labels) - 1:
            y = top + height // 2
            draw.line((x + width + 7, y, x + width + gap - 9, y), fill="#31576A", width=5)
            draw.polygon([(x + width + gap - 9, y), (x + width + gap - 24, y - 10), (x + width + gap - 24, y + 10)], fill="#31576A")
    draw.text((950, 400), "Random initialization only: no pretrained checkpoint is loaded.", fill=MUTED, font=font(21), anchor="ma")
    image.save(ASSET_DIR / "pipeline_architecture.png")


def dataset_chart() -> None:
    image, draw = canvas(1350, 720, "Block-Disjoint Dataset Allocation")
    box = (130, 130, 1240, 570)
    axes(draw, box, 5200, ticks=4)
    labels = ["Train", "Validation", "Test"]
    parents = [1680, 123, 240]
    lines = [4873, 359, 787]
    centres = [360, 685, 1010]
    for centre, label, parent, line in zip(centres, labels, parents, lines):
        for offset, value, colour in [(-70, parent, "#7899AA"), (70, line, BLUE)]:
            x = centre + offset
            y = box[3] - (box[3] - box[1]) * value / 5200
            draw.rounded_rectangle((x - 55, y, x + 55, box[3]), radius=7, fill=colour)
            draw.text((x, y - 12), str(value), fill=NAVY, font=font(19, True), anchor="ms")
        draw.text((centre, box[3] + 22), label, fill=NAVY, font=font(20, True), anchor="ma")
    draw.rectangle((400, 650, 430, 680), fill="#7899AA")
    draw.text((445, 665), "Parent segments", fill=NAVY, font=font(18), anchor="lm")
    draw.rectangle((720, 650, 750, 680), fill=BLUE)
    draw.text((765, 665), "Detected line crops", fill=NAVY, font=font(18), anchor="lm")
    image.save(ASSET_DIR / "dataset_splits.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    training_curves()
    model_comparison()
    pipeline_diagram()
    dataset_chart()
    print(f"Created report assets in {ASSET_DIR}")


if __name__ == "__main__":
    main()

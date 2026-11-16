from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.crnn import CTCVocabulary
from modi_transliterator.line_segmentation import detect_line_regions, split_target_across_lines
from modi_transliterator.text import normalize_target


def build_preview(items: list[tuple[Image.Image, list[tuple[int, int, int, int]]]], output: Path) -> None:
    tiles: list[Image.Image] = []
    for image, boxes in items:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for index, box in enumerate(boxes, start=1):
            draw.rectangle(box, outline=(0, 112, 192), width=max(2, annotated.width // 500))
            draw.text((box[0] + 4, box[1] + 3), str(index), fill=(190, 20, 40))
        scale = min(1.0, 900 / annotated.width)
        tiles.append(annotated.resize((round(annotated.width * scale), round(annotated.height * scale))))
    width = max(tile.width for tile in tiles)
    height = sum(tile.height for tile in tiles) + 16 * (len(tiles) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for tile in tiles:
        canvas.paste(tile, (0, y))
        y += tile.height + 16
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    table = pd.read_csv(args.manifest, encoding="utf-8")
    line_dir = args.output / "lines"
    line_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    preview_items: list[tuple[Image.Image, list[tuple[int, int, int, int]]]] = []
    for _, parent in table.iterrows():
        parent_path = Path(str(parent["image_path"]))
        with Image.open(parent_path) as source_image:
            source = source_image.convert("L")
            regions = detect_line_regions(source)
            labels = split_target_across_lines(
                normalize_target(str(parent["text"])), [region.ink_width for region in regions]
            )
            regions = regions[: len(labels)]
            if len(preview_items) < args.preview_count:
                preview_items.append((source.copy(), [region.box for region in regions]))
            for region, label in zip(regions, labels, strict=True):
                line_filename = f"{Path(str(parent['filename'])).stem}_line_{region.index + 1:02d}.png"
                line_path = line_dir / line_filename
                source.crop(region.box).save(line_path, optimize=True)
                rows.append(
                    {
                        "line_filename": line_filename,
                        "parent_filename": str(parent["filename"]),
                        "line_index": region.index,
                        "source_group": int(parent["source_group"]),
                        "split": str(parent["split"]),
                        "image_path": str(line_path.resolve()),
                        "text": label,
                        "x1": region.x1,
                        "y1": region.y1,
                        "x2": region.x2,
                        "y2": region.y2,
                    }
                )

    lines = pd.DataFrame(rows).sort_values(["parent_filename", "line_index"]).reset_index(drop=True)
    manifest_path = args.output / "line_manifest.csv"
    lines.to_csv(manifest_path, index=False, encoding="utf-8")
    vocabulary = CTCVocabulary.from_texts(lines["text"].tolist())
    vocabulary.save(args.output / "ctc_vocabulary.json")
    parent_counts = lines.groupby("parent_filename").size()
    summary = {
        "parent_segments": int(lines["parent_filename"].nunique()),
        "line_samples": len(lines),
        "split_counts": lines["split"].value_counts().sort_index().to_dict(),
        "lines_per_segment_min": int(parent_counts.min()),
        "lines_per_segment_max": int(parent_counts.max()),
        "lines_per_segment_mean": float(parent_counts.mean()),
        "ctc_characters_excluding_blank": len(vocabulary.characters),
        "alignment_method": "automatic line detection plus width-proportional dynamic-programming word-boundary allocation",
        "limitation": "Line targets are pseudo-aligned because MoDeTrans supplies segment-level text without line breaks.",
    }
    (args.output / "line_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_preview(preview_items, PROJECT_ROOT / "outputs" / "line_segmentation_preview.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

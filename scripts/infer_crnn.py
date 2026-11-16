from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.crnn import prepare_line_image
from modi_transliterator.line_segmentation import detect_line_regions
from scripts.evaluate_crnn import load_model


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, vocabulary, model = load_model(args.checkpoint, device)
    height = int(checkpoint["metadata"]["image_height"])
    max_width = int(checkpoint["metadata"]["max_width"])
    with Image.open(args.image) as source_image:
        source = source_image.convert("L")
        regions = detect_line_regions(source)
        tensors = [prepare_line_image(source.crop(region.box), height, max_width)[0] for region in regions]
        widest = max(tensor.shape[-1] for tensor in tensors)
        batch = torch.ones(len(tensors), 1, height, widest)
        for index, tensor in enumerate(tensors):
            batch[index, :, :, : tensor.shape[-1]] = tensor
        with torch.no_grad():
            logits = model(batch.to(device))
        predicted = logits.argmax(dim=-1).transpose(0, 1).cpu()
        lines = [vocabulary.decode_greedy(row.tolist()) for row in predicted]
        result = {"image": str(args.image), "line_count": len(lines), "lines": lines, "combined_text": " ".join(lines)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            annotated = source.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for index, region in enumerate(regions, start=1):
                draw.rectangle(region.box, outline=(0, 112, 192), width=max(2, annotated.width // 500))
                draw.text((region.x1 + 4, region.y1 + 3), str(index), fill=(190, 20, 40))
            annotated.save(args.output_dir / "line_overlay.png")
            (args.output_dir / "transliteration.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()

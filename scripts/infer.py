from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.data import ImageShape, prepare_image
from modi_transliterator.runtime import load_checkpoint


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, vocabulary, model = load_checkpoint(args.checkpoint, device)
    height, width = checkpoint["metadata"]["image_shape"]
    with Image.open(args.image) as image:
        tensor = prepare_image(image, ImageShape(height, width)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model.generate(tensor, vocabulary.bos_id, vocabulary.eos_id, args.max_length)
    print(vocabulary.decode(prediction[0].cpu().tolist()))


if __name__ == "__main__":
    main()

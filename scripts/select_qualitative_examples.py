from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modi_transliterator.metrics import edit_distance


def main() -> None:
    source = ROOT / "outputs" / "crnn_main_seed_2026" / "evaluation_test" / "predictions.csv"
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["parent_filename"]].append(row)

    examples = []
    for filename, rows in groups.items():
        rows.sort(key=lambda item: int(item["line_index"]))
        reference = " ".join(row["reference"] for row in rows)
        prediction = " ".join(row["prediction"] for row in rows)
        similarity = 1.0 - edit_distance(reference, prediction) / max(1, len(reference))
        examples.append({
            "filename": filename,
            "line_count": len(rows),
            "normalized_edit_similarity": similarity,
            "reference": reference,
            "prediction": prediction,
        })
    examples.sort(key=lambda item: item["normalized_edit_similarity"])
    selected = {
        "challenging": examples[max(0, len(examples) // 20)],
        "median": examples[len(examples) // 2],
        "strong": examples[min(len(examples) - 1, len(examples) * 19 // 20)],
        "selection_note": "Representative 5th-percentile, median, and 95th-percentile test segments; selected after evaluation for qualitative reporting only.",
    }
    output = ROOT / "report_assets" / "qualitative_examples.json"
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

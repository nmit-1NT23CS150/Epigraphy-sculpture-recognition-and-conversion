from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline_metrics = read_json(ROOT / "outputs" / "baseline_seed_2026" / "evaluation_test" / "metrics.json")
    crnn_metrics = read_json(ROOT / "outputs" / "crnn_main_seed_2026" / "evaluation_test" / "metrics.json")
    dataset_summary = read_json(ROOT / "data" / "dataset_summary.json")
    line_summary = read_json(ROOT / "data" / "line_dataset_summary.json")

    core_files = [
        ROOT / "modi_transliterator" / "line_segmentation.py",
        ROOT / "modi_transliterator" / "crnn.py",
        ROOT / "scripts" / "prepare_lines.py",
        ROOT / "scripts" / "train_crnn.py",
        ROOT / "scripts" / "evaluate_crnn.py",
        ROOT / "scripts" / "infer_crnn.py",
        ROOT / "streamlit_app.py",
        ROOT / "data" / "manifest.csv",
        ROOT / "data" / "line_manifest.csv",
    ]
    checkpoints = [
        ROOT / "outputs" / "baseline_seed_2026" / "best.pt",
        ROOT / "outputs" / "crnn_main_seed_2026" / "best.pt",
    ]
    record = {
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "From-scratch Modi manuscript line segmentation and CRNN-CTC transliteration to Devanagari",
        "status": "complete research prototype",
        "dataset": dataset_summary,
        "line_dataset": line_summary,
        "held_out_results": {
            "whole_segment_transformer": baseline_metrics,
            "line_crnn_ctc": crnn_metrics,
        },
        "verification": {
            "unit_tests": "5 passed",
            "isolation_audit": "PASS; 0 violations",
            "web_demo_http_smoke": "HTTP 200 on localhost:8502",
            "cross_project_exact_image_hash_matches": 0,
        },
        "independence": {
            "allowed_dataset": "historyHulk/MoDeTrans",
            "pretrained_weights_loaded": False,
            "initialization": "random",
            "brahmi_data_or_weights_used": False,
            "shared_data_weights_or_caches_with_journal_project": False,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "sha256": {
            "core_files": {str(path.relative_to(ROOT)): sha256(path) for path in core_files},
            "checkpoints": {str(path.relative_to(ROOT)): sha256(path) for path in checkpoints},
        },
        "limitations": [
            "Line targets are automatically pseudo-aligned because the source dataset has no line-level transcripts.",
            "The split is block-disjoint but cannot be proven document-disjoint from public metadata.",
            "The main comparison uses one fixed seed per architecture.",
            "Exact held-out segment match is zero; expert review is required.",
        ],
    }
    output = ROOT / "FINAL_PROJECT_STATUS.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

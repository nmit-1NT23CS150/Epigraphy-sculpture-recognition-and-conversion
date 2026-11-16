from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATASET_ID = "historyHulk/MoDeTrans"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    violations: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in {".git", ".venv", ".venv_crnn", "__pycache__", ".pytest_cache"} for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.is_symlink():
            violations.append(f"symbolic link is not allowed: {path.relative_to(PROJECT_ROOT)}")
        lowered = path.name.lower()
        if "rahas" in lowered or "brahmi" in lowered:
            violations.append(f"foreign-project name in path: {path.relative_to(PROJECT_ROOT)}")
    dataset_summary = PROJECT_ROOT / "data" / "dataset_summary.json"
    if dataset_summary.exists():
        payload = json.loads(dataset_summary.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != ALLOWED_DATASET_ID:
            violations.append(f"unexpected dataset: {payload.get('dataset_id')}")
    parent_manifest = PROJECT_ROOT / "data" / "manifest.csv"
    line_manifest = PROJECT_ROOT / "data" / "line_manifest.csv"
    if parent_manifest.exists() and line_manifest.exists():
        parents = pd.read_csv(parent_manifest).set_index("filename")
        lines = pd.read_csv(line_manifest)
        for row in lines.itertuples(index=False):
            if row.parent_filename not in parents.index:
                violations.append(f"line has unknown parent: {row.parent_filename}")
                continue
            expected_split = str(parents.loc[row.parent_filename, "split"])
            if str(row.split) != expected_split:
                violations.append(f"line split differs from parent: {row.line_filename}")
        if lines.groupby("source_group")["split"].nunique().max() > 1:
            violations.append("a source group crosses line-level splits")
    checkpoints = []
    for pattern in ("*.pt", "*.pth"):
        for path in (PROJECT_ROOT / "outputs").rglob(pattern):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
            if metadata.get("pretrained") is not False or metadata.get("initialization") != "random":
                violations.append(f"checkpoint lacks random-initialization certificate: {path.relative_to(PROJECT_ROOT)}")
            if metadata.get("dataset_id") != ALLOWED_DATASET_ID:
                violations.append(f"checkpoint has unexpected dataset: {path.relative_to(PROJECT_ROOT)}")
            checkpoints.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)})
    result = {
        "status": "FAIL" if violations else "PASS",
        "allowed_dataset": ALLOWED_DATASET_ID,
        "violations": violations,
        "locally_trained_checkpoints": checkpoints,
    }
    print(json.dumps(result, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

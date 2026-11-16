# Epigraphy Sculpture Recognition and Conversion

## Implemented project

**Modi manuscript transliteration to Devanagari using line detection and CRNN-CTC recognition.** This repository contains the implementation described in Group 20's Phase-2 synopsis. The demonstrated task is historical manuscript reading; sculpture recognition and translation are not implemented features.

The trained model and two sample images are included. No training or dataset download is needed for the demonstration. The original research code is imported from the team's local `conference_modi_manuscript` project; this repository starts with that import and does not claim to preserve a semester-long Git history.

## Run for the review

Use Python 3.12. On Windows, double-click **Run_Demo.bat**, or run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://127.0.0.1:8502, select **Use example**, then **Recognize Modi text**. Inspect the line overlay and draft Devanagari text; download TXT, JSON and the marked image. Upload another Modi segment to demonstrate that the previous result is cleared. The first setup requires internet access; inference uses the included checkpoint locally.

## Deployment on a separate machine

The repository includes a CPU Docker recipe:

```sh
docker build -t epigraphy-modi:phase2 .
docker run --rm -p 8502:8502 epigraphy-modi:phase2
```

Run this on the presentation machine or an institution-managed server, then open `http://HOST:8502`. Docker is not available on the authoring machine, so the image build and separate-machine deployment remain unvalidated. A GitHub repository hosts source files; it is not a running Streamlit application. Record the target machine, build log, health response and upload/recognize/download checks before claiming external deployment. For public hosting, add appropriate access control and upload safeguards.

## Results

| Pipeline | Held-out unit | CER | Edit similarity | Exact match |
|---|---|---:|---:|---:|
| CNN-Transformer baseline | 240 segments | 0.8794 | 0.2341 | 0.0000 |
| CRNN-CTC | 787 pseudo-labelled lines | 0.5593 | 0.4025 | 0.0000 |
| Reconstructed CRNN-CTC | 240 segments | 0.4772 | 0.4704 | 0.0000 |

These saved results concern the seed-2026 checkpoints. The pipelines use different architectures and training budgets (5 versus 25 epochs); the result does not isolate the effect of segmentation. Line targets are width-based pseudo-labels. Filename-block splitting prevents parent/crop leakage but is not proven writer- or document-disjoint. Exact match is zero: expert correction is essential. Displayed confidence is an uncalibrated CTC score, not measured accuracy.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\audit_isolation.py
```

See [the revised synopsis](docs/Group_20_Epigraphy_Phase2_Sections_7_8_Updated.pdf), [demonstration guide](docs/DEMO_GUIDE.md), and [dated validation results](evidence/review_validation.json). The audit can check the full parent/crop split only after dataset preparation. After preparing the full dataset, run `python scripts/validate_review.py` to repeat the broader checks; `--source-project PATH` may point to an existing prepared copy. The script writes fresh results into `evidence/`.

## Reproduce training and held-out evaluation

```sh
python scripts/prepare_dataset.py
python scripts/prepare_lines.py
python scripts/train.py --epochs 5 --run-name baseline_seed_2026
python scripts/train_crnn.py --epochs 25 --run-name crnn_main_seed_2026
python scripts/evaluate.py --checkpoint outputs/baseline_seed_2026/best.pt --split test
python scripts/evaluate_crnn.py --checkpoint outputs/crnn_main_seed_2026/best.pt --split test
```

The full manuscript collection and derived crops are downloaded/prepared separately. Historical evaluation CSVs accompany the saved checkpoints; the preparation scripts regenerate machine-specific manifests.

## Repository map

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Upload, validation, batched inference and downloads |
| `modi_transliterator/` | Normalisation, datasets, segmentation, models and metrics |
| `scripts/` | Data preparation, training, evaluation, inference and audit |
| `tests/` | Core automated tests |
| `outputs/` | Included trained checkpoints, configs, histories and saved predictions |
| `data/images/` | Two attributed demo manuscript segments |
| `report_assets/` | Recorded experiment figures |
| `docs/`, `evidence/` | CIE report, demonstration instructions and validation evidence |

## Dataset attribution

MoDeTrans by Kausadikar, Kale, Susladkar and Mittal: *Historic Scripts to Modern Vision: A Novel Dataset and a VLM Framework for Transliteration of Modi Script to Devanagari*, ICDAR 2025 / arXiv:2503.13060. Dataset: https://huggingface.co/datasets/historyHulk/MoDeTrans (MIT licence declared by the dataset card). The two demo images come from this dataset. See `DATASET_AND_ETHICS.md`; retain attribution when sharing. Both included models were trained locally from random initialisation on this dataset, without pretrained OCR weights.

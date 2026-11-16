# CIE review demonstration

Run `Run_Demo.bat` on Windows, or use the setup commands in the README. Open http://127.0.0.1:8502. The first installation requires internet access; the included checkpoint then runs without an external OCR service.

1. Explain the scope: a scanned Modi manuscript segment becomes draft Devanagari text. Sculpture recognition and language translation are not demonstrated.
2. Select **Use example**, then **Recognize Modi text**. Show the three detected lines and their reading order beside the output.
3. Download text, JSON and the marked image. Explain the JSON source field, line count, ordered line text and uncalibrated confidence scores.
4. Upload `data/images/mode_00099.png`. Show that the previous result disappears before a new recognition request.
5. Demonstrate a corrupt image or blank white image being rejected. Explain the 25 MiB upload boundary.
6. Open `evidence/review_validation.json`, `evidence/http_smoke.json` and the saved model metrics. State the measured parent CER of 0.4772 and zero exact matches.
7. Open the revised synopsis in this folder for implementation, traceability, test results and limitations.

## Questions to prepare

- **Why line recognition?** It reduces each input to one horizontal sequence and simplifies reading order. It also adds dependence on reliable line detection.
- **Why CTC?** It learns sequence alignment without character boxes. Greedy decoding removes blanks and adjacent repeated symbols.
- **Are line labels expert annotations?** No. They are allocated from parent transcripts using widths and word boundaries, which introduces label noise.
- **How is leakage checked?** Parent/crop split inheritance, filename-block group separation and cross-split exact-image SHA-256 checks. Writer and document independence are not proven.
- **Is 45.7% lower CER an isolated segmentation benefit?** No. It compares whole pipelines with different architectures and training budgets.
- **Does 78% confidence mean 78% accuracy?** No. It is an uncalibrated model probability statistic.
- **Is the GitHub URL a deployed app?** No. It hosts the code, models and report. A running host is required for a remote demonstration.

## Complete the external deployment requirement

On a separate presentation computer, use Python 3.12 and the README setup, or use the provided Docker commands. Then record the host OS/hardware, dependency or container build log, model SHA-256, server URL and date. Check homepage and health HTTP 200, upload, recognition, TXT/JSON/PNG exports, a changed source, invalid input and the size boundary on that host. Save the results alongside the local evidence. Docker build and external-host validation have not been performed in this package.

## Current validation

The dated local runner reports 15 passed checks and zero failures, including the five-test core suite and complete held-out CRNN evaluation reproduction. The separate HTTP check records homepage and health HTTP 200. The baseline evaluation is verified from its saved artifacts; it was not rerun. Local AppTest execution does not certify all browser/device combinations or separate-host deployment.

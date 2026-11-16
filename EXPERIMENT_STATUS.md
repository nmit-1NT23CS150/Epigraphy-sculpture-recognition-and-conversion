# Experiment Status — 2026-08-20

## Completed

- Downloaded and prepared all 2,043 MoDeTrans image/text pairs.
- Created deterministic block-disjoint splits: 1,680 train, 123 validation, and 240 test samples.
- Built a 95-token character vocabulary including four control tokens.
- Implemented a compact CNN-Transformer image-to-text baseline with random initialization.
- Segmented all parent images into 6,019 line crops while retaining parent-split provenance.
- Implemented and trained a four-stage CNN, two-layer bidirectional GRU, and CTC recognizer from random initialization.
- Verified training, checkpoint loading, line evaluation, reconstructed-segment evaluation, and single-image inference.
- Passed the project isolation audit and all unit tests.

## Whole-segment baseline

The first full-data baseline used seed 2026, five epochs, 128×512 inputs, a maximum training target length of 384, and greedy decoding capped at 192 tokens.

| Measurement | Result |
|---|---:|
| Training loss, epoch 1 | 3.1412 |
| Training loss, epoch 5 | 2.5382 |
| Validation normalized edit similarity, best checkpoint | 0.2238 |
| Test samples | 240 |
| Test character error rate | 0.8794 |
| Test normalized edit similarity | 0.2341 |
| Test exact sequence match | 0.0000 |

Checkpoint SHA-256:

`1ae8460810816e0b555783aab1eb4dd32b2cdd94b3828fc9b041b5798256c91e`

The model frequently collapses toward common Marathi character sequences. This checkpoint is retained as a controlled baseline.

## Line CRNN-CTC result

The main CRNN experiment used seed 2026, 25 epochs, 64-pixel line height, maximum width 1,024 pixels, batch size 24, Adam learning rate 0.001, and greedy CTC decoding. The best checkpoint was selected using validation CER at epoch 24.

| Measurement | Result |
|---|---:|
| Model parameters | 2,180,828 |
| Training time | 1,144.5 s (19.1 min) |
| Validation line samples | 359 |
| Validation CER | 0.5971 |
| Validation normalized edit similarity | 0.3900 |
| Test line samples | 787 |
| Test line CER | 0.5593 |
| Test line normalized edit similarity | 0.4025 |
| Reconstructed test segments | 240 |
| Reconstructed segment CER | 0.4772 |
| Reconstructed segment normalized edit similarity | 0.4704 |
| Exact sequence match | 0.0000 |

Checkpoint SHA-256:

`48f77f1e5d2db95942f82dcbf7caae4bc2f60471494e3bdd48c564e2bc0672b7`

Relative to the whole-segment baseline, reconstructed segment CER falls by 0.4022 absolute (45.7% relative) and normalized edit similarity increases by 0.2364 absolute. This supports the engineering value of explicit line segmentation for this dataset, but it is not a state-of-the-art claim.

## Limitations and next experiments

1. MoDeTrans supplies segment-level transcripts but not line-level transcripts. Line targets are pseudo-aligned at word boundaries by a width-proportional dynamic program and therefore contain label noise.
2. The public metadata does not expose reliable manuscript IDs. Splits are block-disjoint, not proven document-disjoint.
3. The reported main comparison uses one fixed seed per architecture. Multi-seed reporting is required before making statistical claims.
4. Exact test sequence match is zero and human review remains mandatory.
5. Decorative strokes, touching lines, skew, and long descenders can cause line segmentation errors.
6. Future work should add expert-corrected line annotations, beam decoding with a Marathi language model, segmentation confidence, and manuscript-level grouping metadata.

## Separation from the journal work

| Dimension | This conference project | Journal project |
|---|---|---|
| Source writing | Historical Modi manuscripts | Brahmi estampage characters |
| Dataset | MoDeTrans image-text sequences | Separate isolated-character inventory |
| Prediction unit | Multi-line sequence | Isolated class label |
| Architecture | Line segmentation + from-scratch CRNN-CTC (with CNN-Transformer baseline) | Restoration-aware low-shot recognition system |
| Training objective | CTC sequence alignment | Direct/prototype character classification |
| Main metrics | CER, edit similarity, exact sequence match | Accuracy, macro-F1, top-k, low-shot accuracy |
| Shared weights/data/caches | None | None |

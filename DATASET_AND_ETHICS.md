# Dataset, Rights, and Ethics

## Permitted source

Only the following external dataset is permitted in this project:

- `historyHulk/MoDeTrans`
- Dataset card licence: MIT
- Contents: historical handwritten Modi manuscript image segments with Devanagari text targets
- Citation: Kausadikar, Kale, Susladkar, and Mittal, "Historic Scripts to Modern Vision: A Novel Dataset and a VLM Framework for Transliteration of Modi Script to Devanagari," ICDAR 2025 / arXiv:2503.13060.

The original dataset card and paper remain the authoritative sources for provenance. Do not redistribute the exported images without retaining the dataset citation and licence information.

## Isolation rule

No dataset, checkpoint, cache, crop, label inventory, restoration output, or experiment result from another project may be copied into this directory. Training begins from random parameters. A checkpoint produced by this project is acceptable only when its metadata identifies the dataset, seed, source-code architecture, and `pretrained: false`.

## Research boundary

Automated transliteration can contain omissions, substitutions, and historically implausible text. Results must not be treated as an authoritative reading without review by someone qualified in Modi paleography and Marathi historical documents.

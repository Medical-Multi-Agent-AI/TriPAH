# Third-party notices

The target repository's original [Apache 2.0 license](LICENSE) is retained.
Third-party components retain their respective terms described below.

TriPAH extends existing open-source components. Project naming, commands, and
documentation describe TriPAH; upstream credits below identify the provenance
of reused code and are intentionally retained.

- **PromptHash** — the hashing and prompt-learning code is adapted from
  [CCRG-XJU/PromptHash](https://github.com/CCRG-XJU/PromptHash), the implementation
  accompanying Qiang Zou, Shuli Cheng, and Jiayi Chen, “PromptHash:
  Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing
  Retrieval,” CVPR 2025. Its original MIT notice, Copyright (c) 2025 ShiShu Mo,
  is preserved in [licenses/PromptHash-MIT.txt](licenses/PromptHash-MIT.txt).
- **OpenCLIP** — `model/open_clip/` bundles and modifies OpenCLIP (the bundled
  version identifies itself as 2.32.0). See
  [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) and the
  retained [MIT license](licenses/OpenCLIP-MIT.txt).
- **OpenAI CLIP** — OpenCLIP includes code adapted from
  [openai/CLIP](https://github.com/openai/CLIP). Original file headers and the
  [MIT license](licenses/OpenAI-CLIP-MIT.txt) are retained.
- **MAE positional embeddings** — `model/open_clip/pos_embed.py` includes
  positional-embedding utilities originating in
  [facebookresearch/mae](https://github.com/facebookresearch/mae).
  The Meta copyright header is retained; see [MAE license](licenses/MAE.txt).
- **BERT optimizer** — `optimization.py` retains the 2018 Google AI Language
  Team / HuggingFace copyright and Apache 2.0 header. See
  [Apache License 2.0](licenses/Apache-2.0.txt).
- **CMCL** — the upstream retrieval implementation also credits
  [DarrenZZhang/CMCL](https://github.com/DarrenZZhang/CMCL).

PyTorch, timm, Mamba, MetaCLIP weights, and other installed dependencies remain
subject to their own licenses. Dataset and model-weight redistribution rights
are separate from this repository's source-code license.

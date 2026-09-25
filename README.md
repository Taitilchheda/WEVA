# WEVA: Unified Spectral Architecture for Multimodal Learning

> Wave-Energy Vibrational Architecture — one shared spectral encoder and one shared Spectral Wave Mixer backbone process text, image, audio, and video as a common wave-primitive representation.

**Status:** Preprint v0.2 (2026-06-12). Target venue: IEEE TNNLS.

---

## What is this?

WEVA is a multimodal representation learning architecture that **removes dot-product attention** in favor of a learned frequency-domain mixer, and **shares the entire backbone** across four modalities. The only modality-specific components are the input frontends and the output heads.

**Headline empirical claim:** adding audio and video to a text+image contrastive training mix does not degrade text–image R@1 on COCO Captions 5K by more than 0.5 points (pass criterion: 2.0). The shared SWM backbone matches a per-modality-tower SWM baseline of equal parameter count within 0.4 R@1.

## Three structural differentiators (no prior model has all three)

1. **Modality-agnostic backbone.** One shared spectral encoder and one shared SWM process all 4 modalities. ImageBind has 6 separate encoders; CLIP has 2.
2. **Amplitude + phase learned separately.** Standard complex filters use a single complex `W = W_amp + i·W_phase`. We split into two real matrices; the phase matrix is the load-bearing novel parameter.
3. **Modality token is the only modality-specific signal in the backbone.** 4 learned vectors (1024 params for d=256) replace per-modality routing or cross-attention gates.

## Repository layout

```
nicola inspired transformer/
├── README.md                  ← you are here
├── docs/
│   ├── WEVA_DESIGN.md         ← design doc (architecture, theory, risks)
│   ├── WEVA_Four_Proposals.md ← 4 candidate contributions, why A won
│   └── file.js                ← v1 research document (Tesla/OER origin)
├── paper/
│   ├── WEVA_paper.md          ← Q1 paper draft (Markdown, primary)
│   ├── WEVA_paper.tex         ← Q1 paper draft (IEEE TNNLS LaTeX)
│   └── WEVA.pdf               ← compiled PDF
├── diagrams/                  ← 10 editable SVG diagrams
│   ├── 01_overall_architecture.svg
│   ├── 02_universal_spectral_encoder.svg
│   ├── 03_swm_block.svg
│   ├── 04_modality_token_shared_backbone.svg
│   ├── 05_contrastive_training.svg
│   ├── 06_comparison_table.svg
│   ├── 07_swm_kernel_interpretation.svg
│   ├── 08_gradient_flow.svg
│   ├── 09_scaling_law.svg
│   └── 10_oer_energy_shells.svg
├── code/                      ← end-to-end PyTorch implementation
│   ├── weva/
│   │   ├── config.py          ← WEVAConfig + Tiny/Small/Medium presets
│   │   ├── encoder.py         ← Universal Spectral Encoder + 4 frontends
│   │   ├── mixer.py           ← Spectral Wave Mixer + multi-head + block
│   │   ├── frn.py             ← SIREN-style Feedforward Resonance Network
│   │   ├── model.py           ← WEVAModel
│   │   ├── losses.py          ← Multimodal InfoNCE + OER (appendix)
│   │   ├── data.py            ← SyntheticMultimodalDataset + CocoCaptions
│   │   ├── train.py           ← training loop
│   │   ├── eval.py            ← retrieval evaluation
│   │   └── demo.py            ← cross-modal similarity demo
│   ├── scripts/
│   │   ├── smoke_test.py      ← 60s CPU end-to-end correctness check
│   │   └── profile_60m.py     ← peak-VRAM and step-time profiler
│   └── requirements.txt
├── chat_interface/            ← retrieval-style chat web app (see below)
├── models/                    ← trained checkpoints + histories
│   ├── weva_tiny_train1.pt           (16M, 5 epochs on 1024 samples)
│   ├── weva_small_train1.pt          (33M, 1 epoch on 8192 samples)
│   ├── weva_tiny_train1_eval.json
│   ├── weva_small_train1_eval.json
│   └── RTX_3060_PROFILE.md   ← measured peak VRAM and step time
└── findings/                  ← raw experiment outputs
```

## Quick start

```bash
cd "code"
pip install -r requirements.txt

# 1) Verify the implementation (60s on CPU)
python scripts/smoke_test.py

# 2) Profile memory + speed on your GPU
python scripts/profile_60m.py
#    → produces models/RTX_3060_PROFILE.md with measured numbers

# 3) Train WEVA-Tiny on synthetic data (a few minutes on CPU)
python -m weva.train --config tiny --n-samples 1024 --batch-size 8 --n-epochs 3

# 4) Evaluate retrieval (synthetic)
python -m weva.eval --checkpoint ../models/weva_tiny_train1.pt --n-samples 256

# 5) Run a cross-modal similarity demo
python -m weva.demo --checkpoint ../models/weva_tiny_train1.pt
```

For real COCO/AudioSet/MSR-VTT training, swap the dataset in `code/weva/data.py` and use `--config small` (33M params, fits 12GB at batch 32). See `models/RTX_3060_PROFILE.md` for measured VRAM and step times.

## Model sizes (verified on RTX 3060 12 GB)

| Name | d_model | L | H | Params | Best batch on 12 GB | Step time |
|---|---|---|---|---:|---:|---:|
| WEVA-Tiny   | 256 | 4  | 4 | 16 M  | 64  | 176 ms  |
| WEVA-Small  | 384 | 8  | 6 | 33 M  | 32  | 564 ms  |
| WEVA-Medium | 512 | 12 | 8 | 58 M  | 16  | 836 ms  |

(60M is the **largest** model that fits at useful batch size; see `models/RTX_3060_PROFILE.md`.)

## Headline result (modality ablation, COCO 5K, single seed)

| Training modalities | i→t R@1 | t→i R@1 |
|---|---|---|
| Text + Image (2M) | 27.1 | 21.0 |
| Text + Image + Audio (3M) | 26.9 | 20.8 |
| Text + Image + Audio + Video (4M) | 26.8 | 20.6 |

All three configurations are within 0.5 R@1. Adding modalities is approximately neutral. Pass criterion (≤ 2 R@1) is met.

## Chat interface

A separate folder `chat_interface/` contains a Flask web app that loads a trained WEVA model and serves a retrieval-style chat: users can type text, upload an image / audio / video, and WEVA returns the most similar items from a pre-indexed database with a template-based natural-language summary. **It is NOT a generative chatbot** — WEVA is a representation model, not a language model. See `chat_interface/README.md` for the full scope statement.

```bash
cd chat_interface
pip install -r requirements.txt
python index_dataset.py --checkpoint ../models/weva_tiny_train1.pt
python app.py --checkpoint ../models/weva_tiny_train1.pt
# → open http://localhost:5000
```

## How to read the paper

- **Markdown** (`paper/WEVA_paper.md`) — for browsing, comments, PR reviews.
- **LaTeX** (`paper/WEVA_paper.tex`) — for IEEE TNNLS submission. Compile with `pdflatex` (twice for refs) or `latexmk`.
- **PDF** (`paper/WEVA.pdf`) — compiled output.

## Citation

```bibtex
@misc{chheda2026weva,
  author = {Chheda, Taitil},
  title  = {WEVA: A Unified Spectral Architecture for Multimodal Representation Learning},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT.

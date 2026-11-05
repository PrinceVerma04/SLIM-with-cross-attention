# SLIM-XA: Cross-Attention Extended SLIM for Audio Deepfake Detection

> An extension of [SLIM (NeurIPS 2024, arXiv 2407.18517)](https://arxiv.org/abs/2407.18517) that replaces the fixed L2 mismatch measure with a learned **bidirectional cross-attention module**, producing directional residual features intended to make the style-linguistics mismatch signal more discriminative (not yet ablated against the SLIM baseline).

**Results on ASVspoof 2019 LA Evaluation Set (50% stratified subset, attack labels A07–A19, 1 GPU, no augmentation):**

| Metric | SLIM (paper) | SLIM-XA (ours) |
|--------|:-----------:|:--------------:|
| EER | 0.60% | 22.37% |
| Accuracy | — | 90% |
| Fake-speech recall | — | 99.98% |

> - **EER is the primary metric.** It does not depend on a decision threshold. Accuracy and fake-speech recall are measured at a single operating threshold (see [Section 8](#8-results)).
> - The test subset is ~90% fake (31,941 of 35,618 utterances), so a classifier that labels everything as fake would score ≈89.7% accuracy. Accuracy and fake-speech recall alone are therefore not informative.
> - SLIM-XA does **not** reach the paper's EER. The setups also differ (50% of the data, one GPU, no augmentation, different Stage 1 data; see the comparison table in Section 8). No ablation has been run to isolate the cause of the gap, and these results do not show that cross-attention improves over SLIM.

---

## Table of Contents

1. [What is SLIM?](#1-what-is-slim)
2. [What Does SLIM-XA Add?](#2-what-does-slim-xa-add)
3. [Architecture](#3-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup](#5-environment-setup)
6. [Datasets](#6-datasets)
7. [Run Sequence](#7-run-sequence)
8. [Results](#8-results)
9. [Citation](#9-citation)

---

## 1. What is SLIM?

**SLIM (Style-Linguistics Mismatch Model)** builds on the observation that real speech has a natural, learned dependency between *style* (emotion, speaker identity, prosody) and *linguistics* (word content). TTS and voice-conversion systems synthesize these independently, which, according to the SLIM paper, introduces an artificial style-linguistics dependency that differs from real speech and helps generalization to unseen spoofing methods.

SLIM uses:
- A frozen **style encoder** (Wav2Vec2-XLSR-SER, layers 0–10) to capture paralinguistic style
- A frozen **linguistic encoder** (Wav2Vec2-XLSR-ASR, layers 14–21) to capture phonetic content
- **Compression modules** trained self-supervised on real speech to align the two subspaces
- A binary **classifier** that fuses raw SSL features with learned dependency features

---

## 2. What Does SLIM-XA Add?

SLIM-XA replaces the fixed Euclidean distance (L2/cosine) mismatch measure with **learned bidirectional cross-attention**:

```
style_pred  = Attn(Q=style_dep, KV=ling_dep)   # what style does linguistics predict?
ling_pred   = Attn(Q=ling_dep,  KV=style_dep)   # what content does style predict?

style_res   = LayerNorm(style_dep - style_pred)  # unexplained style (expected to be larger for TTS)
ling_res    = LayerNorm(ling_dep  - ling_pred)   # unexplained content (expected to be larger for VC)
```

These **two directional residual streams** are trained in Stage 1 and frozen in Stage 2, raising the total feature dimension from **1024 (SLIM)** to **1536 (SLIM-XA)**.

**Motivation (hypotheses, not yet validated by ablation):**
- **Directional:** may capture distinct failure modes of TTS (large `style_res`) vs. VC (large `ling_res`)
- **Learns the dependency manifold:** attends over the temporal sequence rather than assuming Euclidean proximity
- **Richer Stage 2 features:** two additional 256-dim streams
- **Potentially interpretable:** attention weights could show which time steps are correlated in real vs. fake speech (not yet analyzed)

---

## 3. Architecture

### Stage 1 — Self-Supervised Dependency Learning (real speech only)

```
Real Speech Waveform  [B, 1, 80000]  (5s @ 16 kHz)
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
┌──────────────────┐  FROZEN                  ┌──────────────────────┐  FROZEN
│   Style Encoder  │                          │  Linguistic Encoder  │
│  Wav2Vec2-XLSR   │                          │  Wav2Vec2-XLSR-ASR   │
│  (layers 0–10)   │                          │  (layers 14–21)      │
└──────────────────┘                          └──────────────────────┘
     [B, 11, 1024, T']                              [B, 8, 1024, T']
        │ avg over layers                           │ avg over layers
        ▼                                           ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│   Compression Module     │        │   Compression Module     │  TRAINABLE
│  1024→256→1024→256 (BN)  │        │  1024→256→1024→256 (BN)  │
└──────────────────────────┘        └──────────────────────────┘
     style_dep [B, 256, T']              ling_dep [B, 256, T']
        │                                    │
        └─────────────┬──────────────────────┘
                      ▼
          ┌─────────────────────────┐
          │  CrossSubspaceAttention │  TRAINABLE  ★ SLIM-XA novel
          │  (4 heads, D=256)       │
          │  style_res, ling_res    │
          └─────────────────────────┘

Stage 1 Loss = L_cross + 0.007 × L_intra + 0.1 × L_attn
```

### Stage 2 — Supervised Binary Classification (real + fake)

```
6 feature streams concatenated → 1536-dim → classifier

  Stream 1:  style_raw_vec  [B, 256]   raw style SSL (signal artifacts)
  Stream 2:  ling_raw_vec   [B, 256]   raw linguistic SSL (phonetic artifacts)
  Stream 3:  style_dep_vec  [B, 256]   compressed style dependency
  Stream 4:  ling_dep_vec   [B, 256]   compressed linguistic dependency
  Stream 5:  style_res_vec  [B, 256]   ★ style unexplained by linguistics
  Stream 6:  ling_res_vec   [B, 256]   ★ linguistics unexplained by style
                                       ─────────────────────────────────
                             concat  →  final_feat [B, 1536]
                                           │
                             Linear(1536→512) → LeakyReLU → Dropout(0.25)
                                           │
                             Linear(512→1) → sigmoid → P(fake)
```

All SSL encoders, compression modules, and the cross-attention module are **frozen** in Stage 2. Only the ASP heads and the classifier are trained.

---

## 4. Repository Structure

```
SLIM/
├── src/
│   ├── models/
│   │   ├── paper_ssl_encoders.py        # Wav2Vec2 style + linguistic encoders (frozen)
│   │   ├── compression.py               # Bottleneck autoencoder 1024→256→1024→256
│   │   ├── cross_subspace_attention.py  # CrossSubspaceAttention (SLIM-XA novelty)
│   │   ├── stage1_slim.py               # Stage 1 model
│   │   └── stage2_slim.py               # Stage 2 model (6 streams, 1536-dim)
│   ├── losses/
│   │   └── stage1_loss.py               # Exact Algorithm 1 from paper
│   ├── data/
│   │   ├── audio_dataset.py             # PyTorch Dataset for CSV-based audio loading
│   │   └── prepare_metadata.py          # Generates data/splits/*.csv
│   ├── train/
│   │   ├── train_stage1.py              # Stage 1 training script
│   │   └── train_stage2.py              # Stage 2 training script
│   ├── eval/
│   │   ├── metrics.py                   # EER, F1, ROC, DET
│   │   ├── evaluate_stage2.py           # Full test-set evaluation
│   │   ├── visualize_features.py        # Feature space visualization (PCA, t-SNE)
│   │   └── visualize_metrics.py         # ROC / DET / confusion matrix plots
│   ├── infer/
│   │   └── predict_audio.py             # Single-file inference (REAL / FAKE)
│   └── utils/
│       └── audio_utils.py               # Audio load, pad/trim, normalize
├── data/
│   └── splits/                          # Generated CSVs (created by prepare_metadata.py)
│       ├── stage1_real.csv
│       ├── stage2_train.csv
│       ├── stage2_val.csv
│       └── stage2_test.csv
├── outputs/                             # Created at runtime
│   ├── stage1_best.pt                   # Stage 1 checkpoint
│   ├── stage2_best.pt                   # Stage 2 checkpoint
│   ├── eval/                            # Evaluation artifacts (.npz, .json, .csv)
│   └── plots/                           # Visualization outputs
├── ARCHITECTURE_FLOWCHART.md
├── PROJECT_DESCRIPTION.md
├── RUN_SEQUENCE.md
├── requirements.txt
└── README.md
```

---

## 5. Environment Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (11 GB+ VRAM recommended; tested on NVIDIA Titan Xp)
- Conda (recommended)

### Install Dependencies

```bash
# Create and activate a conda environment
conda create -n slim_xa python=3.9
conda activate slim_xa

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
`torch`, `transformers`, `librosa`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`

### Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

---

## 6. Datasets

### Stage 1 (real speech only — self-supervised)

| Source | Samples (50%) | Role |
|--------|:------------:|------|
| ASVspoof 2019 LA train (bonafide) | ~1,290 | Clean studio speech; establishes the style-linguistic dependency baseline |
| MILABS (m_ailabs) | ~5,600 | Multi-lingual emotional speech; diversifies style variation |
| **Total** | **6,890** | |

### Stage 2 (real + fake — supervised)

| Split | Source | Samples (50%) | Real | Fake |
|-------|--------|:-------------:|:----:|:----:|
| Train | ASVspoof 2019 LA train | 12,690 | 1,290 | 11,400 |
| Val | ASVspoof 2019 LA dev | 12,422 | 1,274 | 11,148 |
| Test | ASVspoof 2019 LA eval | 35,618 | 3,677 | 31,941 |

The eval set uses attack labels A07–A19, none of which appear among the training attack labels (A01–A06), so it tests generalization to attack systems not seen in training.

### Expected Data Layout

```
data/
└── raw/
    ├── ASVspoof2019/
    │   └── LA/
    │       ├── ASVspoof2019_LA_train/
    │       ├── ASVspoof2019_LA_dev/
    │       ├── ASVspoof2019_LA_eval/
    │       └── ASVspoof2019_LA_cm_protocols/
    ├── MILABS/
    │   └── train/real/*.wav
    └── FoR/   (optional)
```

Download ASVspoof 2019 LA from the [official ASVspoof website](https://datashare.ed.ac.uk/handle/10283/3336).

---

## 7. Run Sequence

Work from the repo root with the conda environment active.

### Step 1 — Generate Data Splits

```bash
python src/data/prepare_metadata.py
```

Generates `data/splits/stage1_real.csv`, `stage2_train.csv`, `stage2_val.csv`, `stage2_test.csv` (50% stratified sampling).

```
Expected output:
  stage1_real.csv    total=  6890  real=  6890  fake=     0
  stage2_train.csv   total= 12690  real=  1290  fake= 11400
  stage2_val.csv     total= 12422  real=  1274  fake= 11148
  stage2_test.csv    total= 35618  real=  3677  fake= 31941
```

### Step 2 — Train Stage 1 (~1 hour on Titan Xp)

Trains: `style_compression`, `ling_compression`, `cross_attn`

```bash
python -m src.train.train_stage1 \
  --batch-size 16 \
  --epochs     50 \
  --lr         5e-3 \
  --end-lr     1e-4 \
  --lambda-intra 0.007 \
  --lambda-attn  0.1 \
  --patience   3 \
  --num-workers 4 \
  --duration   5.0
```

Saves `outputs/stage1_best.pt`. Monitor for `cross` and `attn` losses decreasing.

### Step 3 — Train Stage 2 (~3–4 hours on Titan Xp)

Trains: `style_asp`, `ling_asp`, `style_projection`, `ling_projection`, `classifier`
Frozen: SSL encoders, compression modules, `cross_attn`

```bash
python -m src.train.train_stage2 \
  --batch-size 2 \
  --epochs     10 \
  --lr         1e-4 \
  --end-lr     1e-5 \
  --patience   3 \
  --num-workers 4 \
  --duration   5.0 \
  --stage1-ckpt outputs/stage1_best.pt
```

Saves `outputs/stage2_best.pt`.

### Step 4 — Evaluate on Test Set

```bash
python -m src.eval.evaluate_stage2 --threshold-mode eer
```

Outputs to `outputs/eval/threshold_eer/`:
- `metrics_summary.json` — EER, F1, accuracy, precision, recall
- `predictions.csv` — per-sample scores

### Step 5 — Visualize Metrics

```bash
python -m src.eval.visualize_metrics \
  --artifacts outputs/eval/threshold_eer/stage2_eval_artifacts.npz \
  --run-name  eer_threshold
```

Generates ROC curve, DET curve, confusion matrix, score histogram in `outputs/plots/`.

### Step 6 — Visualize Features

```bash
python -m src.eval.visualize_features \
  --artifacts outputs/eval/threshold_eer/stage2_eval_artifacts.npz \
  --run-name  eer_threshold
```

Generates mismatch histograms, PCA/t-SNE of 1536-dim feature space, CSAI residual norm histograms.

### Step 7 — Single-File Inference

```bash
python -m src.infer.predict_audio \
  --audio     path/to/audio.wav \
  --threshold 0.5
```

Output: `Prediction: REAL / FAKE  |  Confidence: 0.xxxx`

### Full Pipeline (one-liner)

```bash
python src/data/prepare_metadata.py && \
python -m src.train.train_stage1 --batch-size 16 --epochs 50 --lr 5e-3 --end-lr 1e-4 \
  --lambda-intra 0.007 --lambda-attn 0.1 --patience 3 --num-workers 4 --duration 5.0 && \
python -m src.train.train_stage2 --batch-size 2 --epochs 10 --lr 1e-4 --end-lr 1e-5 \
  --patience 3 --num-workers 4 --duration 5.0 --stage1-ckpt outputs/stage1_best.pt && \
python -m src.eval.evaluate_stage2 --threshold-mode eer && \
python -m src.eval.visualize_features && \
python -m src.eval.visualize_metrics
```

---

## 8. Results

### ASVspoof 2019 LA Evaluation Set

50% stratified subset: 35,618 utterances (3,677 real / 31,941 fake).

| Metric | Value |
|--------|:-----:|
| EER | **22.37%** |
| Accuracy | 90% |
| Fake-speech recall | 99.98% |
| Decision threshold (accuracy / recall) | [fill in from your final evaluation run] |

**How to read these numbers**
- **EER** is threshold-independent and is the primary metric for this task.
- **Accuracy and fake-speech recall** depend on the decision threshold and on class balance. A classifier that labels every utterance as fake would reach ≈89.7% accuracy and 100% fake-speech recall on this subset, so these two numbers should not be read as evidence of detection quality on their own.
- The results come from a single configuration. No ablation (for example SLIM without cross-attention under identical training conditions) has been run.

### Comparison: SLIM vs SLIM-XA

| | SLIM (paper) | SLIM-XA (ours) |
|--|--|--|
| Mismatch signal | L2/cosine distance | Cross-attention residuals ★ |
| Feature streams | 4 × 256 = 1024-dim | 6 × 256 = 1536-dim ★ |
| EER (ASVspoof 2019 LA eval) | 0.60% | 22.37% |
| Evaluation set | Full eval set (71,237 utterances) | 50% stratified subset (35,618 utterances) |
| Stage 1 data | Common Voice + RAVDESS | ASVspoof 2019 LA train (bonafide) + MILABS |
| Stage 2 training data | Full ASVspoof 2019 LA train | 50% of ASVspoof 2019 LA train |
| Compute | 4 GPUs | 1 GPU (Titan Xp) |
| Stage 2 augmentation | Noise, Reverb, SpecAug | None |

SLIM (paper) figures are for the frozen-frontend "Full" variant, averaged over three seeds, as reported in the SLIM paper.

### Configuration changes from the initial implementation

The current code follows the SLIM paper's hyperparameters (Table 5). The table lists what changed from the first version of this codebase.

| Parameter | Initial implementation | Current |
|-----------|:----------------------:|:-------:|
| Optimizer | Adam | AdamW (wd=1e-4) |
| LR schedule | None | Linear decay + 5% warmup |
| λ (intra-subspace) | 0.1 | **0.007** |
| Compression | Conv1d 1024→512→256 | Bottleneck 1024→256→1024→256 |
| Stage 1 batch size | 1 | 16 |
| Audio length | 2 s | **5 s** |
| Gradient clipping | None | max_norm=1.0 |

The Stage 2 data augmentation used in the paper (Noise, Reverb, SpecAug) is not implemented here.

---

## 9. Citation

If you use this work, please cite the original SLIM paper:

```bibtex
@inproceedings{zhu2024neurips-slim,
  title     = {{SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection}},
  author    = {Zhu, Yi and Koppisetti, Surya and Tran, Trang and Bharaj, Gaurav},
  booktitle = {Neural Information Processing Systems},
  year      = {2024}
}
```

For this SLIM-XA extension (cross-attention variant):

```
Prince Verma, SLIM-XA: Cross-Attention Extended SLIM for Audio Deepfake Detection, 2026.
GitHub: https://github.com/PrinceVerma04/SLIM-with-cross-attention
```

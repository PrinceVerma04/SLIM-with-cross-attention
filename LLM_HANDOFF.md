# LLM Handoff Document — SLIM-CSAI Project
## Complete context for continuing this project with another LLM

---

## 1. What This Project Is

**Task:** Audio deepfake detection — classify speech as REAL (human) or FAKE (TTS/voice-conversion generated).

**Base paper:** "SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection" (arXiv 2407.18517, Reality Defender, Jul 2024).

**Our contribution:** We extended SLIM with a novel module called **CSAI (Cross-Subspace Attention Inconsistency)** and corrected several implementation bugs from the original codebase.

---

## 2. Core Idea — SLIM vs SLIM-CSAI

### Original SLIM (paper)
Real speech has a natural dependency between:
- **Style** (emotion, speaker identity, prosody) — encoded by Wav2Vec2-XLSR-SER (emotion recognition), layers 0–10
- **Linguistics** (verbal content) — encoded by Wav2Vec2-XLSR-ASR (speech recognition), layers 14–21

TTS/VC systems model style and content **independently**, creating an artificial mismatch.

SLIM learns compressed "dependency features" via a two-stage pipeline:
- **Stage 1:** Self-supervised on real speech only — trains compression modules to align style and linguistic subspaces
- **Stage 2:** Supervised on real+fake — concatenates dependency features with raw SSL features → binary classifier

### Our Extension: CSAI
Instead of measuring mismatch with a fixed L2/cosine distance (SLIM's approach), we use **bidirectional cross-attention**:

```
style_pred  = CrossAttn(Q=style_dep, KV=ling_dep)   # what style does ling predict?
ling_pred   = CrossAttn(Q=ling_dep,  KV=style_dep)  # what content does style predict?
style_res   = style_dep - style_pred                 # unexplained style   → large for fake
ling_res    = ling_dep  - ling_pred                  # unexplained content → large for fake
```

The residuals (`style_res`, `ling_res`) are additional features for Stage 2 — increasing feature dimension from 1024 to **1536**.

**Why better than L2:**
1. Directional (style→ling ≠ ling→style, captures different failure modes of TTS vs VC)
2. Learns the actual dependency manifold from real speech rather than assuming Euclidean proximity
3. Attention weights are interpretable (show which time steps are correlated)

---

## 3. Environment

- **Conda env:** `wildspoof_sasv`
- **Python:** `/home/princeverma/miniconda3/envs/wildspoof_sasv/bin/python`
- **PyTorch:** 2.2.2 (CUDA compatible with Titan Xp, sm_61)
- **GPU:** NVIDIA Titan Xp, 11 GB VRAM, CUDA available ✓
- **Key packages:** torch 2.2.2, transformers 4.57.6, librosa 0.11.0, scikit-learn, pandas, tqdm, soundfile, matplotlib

**Activate with:**
```bash
conda activate wildspoof_sasv
cd "/media/princeverma/044e2d0c-caca-4d10-867b-0cc38ebc5ac4/wildspoof_dataset/new dataset/slim/SLIM"
```

---

## 4. Project Directory Structure

```
SLIM/
├── src/
│   ├── models/
│   │   ├── paper_ssl_encoders.py        # Wav2Vec2 style + linguistic encoders (UNCHANGED)
│   │   ├── compression.py               # MODIFIED: bottleneck autoencoder 1024→256→1024→256
│   │   ├── cross_subspace_attention.py  # NEW: CSAI bidirectional cross-attention module
│   │   ├── stage1_slim.py               # MODIFIED: includes CrossSubspaceAttention
│   │   └── stage2_slim.py               # MODIFIED: 6 feature streams (1536-dim)
│   ├── losses/
│   │   └── stage1_loss.py               # MODIFIED: exact Algorithm 1 from paper
│   ├── data/
│   │   ├── audio_dataset.py             # UNCHANGED: loads CSV, calls preprocess_audio
│   │   └── prepare_metadata.py          # MODIFIED: ASVspoof2019+MILABS, 50% sampling
│   ├── train/
│   │   ├── train_stage1.py              # MODIFIED: AdamW, linear scheduler, 5s, patience
│   │   └── train_stage2.py              # MODIFIED: AdamW, linear scheduler, 5s, patience
│   ├── eval/
│   │   ├── metrics.py                   # UNCHANGED: EER, F1, confusion matrix
│   │   ├── evaluate_stage2.py           # MODIFIED: duration=5s, saves CSAI residuals to NPZ
│   │   ├── visualize_features.py        # MODIFIED: fixed paths, added CSAI residual plots
│   │   └── visualize_metrics.py         # MODIFIED: fixed default artifacts path
│   ├── infer/
│   │   └── predict_audio.py             # MODIFIED: duration=5s
│   └── utils/
│       └── audio_utils.py               # UNCHANGED: librosa load, pad/trim, normalize
├── data/splits/                         # Generated CSVs (50% of each dataset)
│   ├── stage1_real.csv                  # 6,890 real samples
│   ├── stage2_train.csv                 # 12,690 samples (real+fake)
│   ├── stage2_val.csv                   # 12,422 samples
│   └── stage2_test.csv                  # 35,618 samples
├── outputs/
│   ├── stage1_best.pt                   # Stage 1 checkpoint (if trained)
│   ├── stage2_best.pt                   # Stage 2 checkpoint (if trained)
│   ├── eval/                            # Evaluation artifacts (.npz, .json, .csv)
│   └── plots/                           # Visualization outputs
├── AVSSPOOF_2019/ASVSpoof_2019/LA/LA/   # ASVspoof 2019 LA dataset (.flac files)
├── MILABS/m_ailabs_mlaad/               # MILABS dataset (.wav files, multilingual)
├── MLAAD/Mlaad_v5/mlaad_v5/fake/       # MLAAD v5 (38 languages, fake only — for future OOD eval)
├── ARCHITECTURE_FLOWCHART.md            # ASCII flowchart of full architecture
├── PROJECT_DESCRIPTION.md              # Detailed rationale for all design choices
├── RUN_SEQUENCE.md                      # Step-by-step commands to run everything
└── LLM_HANDOFF.md                       # This file
```

---

## 5. Datasets Used and Why

### Stage 1 (real speech only — self-supervised)
| Dataset | Samples used | Why |
|---------|-------------|-----|
| ASVspoof 2019 LA train bonafide | ~1,290 (50%) | Standard benchmark real speech; no label leakage into Stage 2 eval |
| MILABS m_ailabs_mlaad train/real | ~5,600 (50%) | Multi-lingual emotional speech; adds stylistic diversity to Stage 1 |
| **Total** | **6,890** | |

### Stage 2 (real + fake — supervised)
| Split | Source | Samples (50%) | Real | Fake |
|-------|--------|--------------|------|------|
| Train | ASVspoof 2019 LA train | 12,690 | 1,290 | 11,400 |
| Val | ASVspoof 2019 LA dev | 12,422 | 1,274 | 11,148 |
| Test | ASVspoof 2019 LA eval | 35,618 | 3,677 | 31,941 |

**Why 50%:** Reduces training time ~50% while preserving class balance (stratified sampling per class).

**MLAAD:** Not used in training — 38 language fake-only dataset, reserved for future out-of-domain evaluation.

### Data format
- ASVspoof 2019: `.flac` files at 16kHz (loaded by librosa+soundfile)
- MILABS: `.wav` files at 16kHz
- All audio padded/trimmed to **5 seconds** (80,000 samples at 16kHz)

---

## 6. Architecture — Full Detail

### SSL Encoders (FROZEN throughout both stages)
```python
# Style encoder: paralinguistic info (emotion, speaker)
ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition  → layers 0-10  → [B, 11, 1024, T']

# Linguistic encoder: content/phonetic info
jonatasgrosman/wav2vec2-large-xlsr-53-english              → layers 14-21 → [B, 8, 1024, T']
```

### Compression Module (FIXED from original — paper Fig. 7)
```
Input [B, K, 1024, T'] → avg(K) → [B, 1024, T']
Per time step [B*T, 1024]:
  Linear(1024→256) → BN → LeakyReLU(0.1) → Dropout(0.1)   # compress
  Linear(256→1024) → BN → LeakyReLU(0.1) → Dropout(0.1)   # bottleneck recovery
  Linear(1024→256)                                           # projection head
Output [B, 256, T']
```

### CrossSubspaceAttention (NEW — CSAI)
```python
# style_dep, ling_dep: [B, 256, T']
style_pred = MultiheadAttention(Q=style_dep, K=ling_dep, V=ling_dep)  # 4 heads
ling_pred  = MultiheadAttention(Q=ling_dep,  K=style_dep, V=style_dep)
style_residual = LayerNorm(style_dep - style_pred)  # [B, 256, T']
ling_residual  = LayerNorm(ling_dep  - ling_pred)   # [B, 256, T']
attn_loss = ||style_residual||² + ||ling_residual||²  # minimized for real in Stage 1
```

### Stage 1 Training
- Trains: `style_compression`, `ling_compression`, `cross_attn`
- Frozen: `style_encoder`, `ling_encoder`
- Loss (exact Algorithm 1 from paper):
  ```
  x_norm = BatchNorm1d(x, affine=False) / batch_size
  L_cross = ||style_norm - ling_norm||_F²           (Frobenius norm)
  L_intra = ||S^T S - I||_F² + ||L^T L - I||_F²    (intra-subspace redundancy)
  L_attn  = ||style_residual||² + ||ling_residual||² (CSAI addition)
  L_total = L_cross + 0.007 * L_intra + 0.1 * L_attn
  ```
- Hyperparams: batch=16, epochs=50, AdamW lr=5e-3→1e-4 (linear decay), patience=3, grad_clip=1.0
- Saves: `outputs/stage1_best.pt` (keys: style_compression, ling_compression, cross_attn, epoch, loss)

### Stage 2 Training
- Trains: `style_asp`, `ling_asp`, `style_projection`, `ling_projection`, `classifier`
- Frozen: all of Stage 1 (encoders + compression + cross_attn)
- Feature concatenation (1536-dim):
  ```
  [style_raw_vec(256), ling_raw_vec(256),     ← ASP+MLP over raw SSL embeddings
   style_dep_vec(256), ling_dep_vec(256),     ← mean-pooled compressed dependency
   style_res_vec(256), ling_res_vec(256)]     ← mean-pooled CSAI residuals (NOVEL)
  ```
- Classifier: `Linear(1536→512) → LeakyReLU → Dropout(0.25) → Linear(512→1)`
- Loss: `BCEWithLogitsLoss`
- Hyperparams: batch=2, epochs=10, AdamW lr=1e-4→1e-5 (linear decay), patience=3, grad_clip=1.0
- Saves: `outputs/stage2_best.pt` (full model state_dict)

---

## 7. Complete Run Sequence

```bash
# 0. Activate environment
conda activate wildspoof_sasv
cd "/media/princeverma/044e2d0c-caca-4d10-867b-0cc38ebc5ac4/wildspoof_dataset/new dataset/slim/SLIM"

# 1. Check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 2. Generate data splits (50% of each dataset)
python src/data/prepare_metadata.py

# 3. Train Stage 1 (~1 hour)
python -m src.train.train_stage1 --batch-size 16 --epochs 50 --lr 5e-3 --end-lr 1e-4 --lambda-intra 0.007 --lambda-attn 0.1 --patience 3 --num-workers 4 --duration 5.0

# 4. Train Stage 2 (~3-4 hours)
python -m src.train.train_stage2 --batch-size 2 --epochs 10 --lr 1e-4 --end-lr 1e-5 --patience 3 --num-workers 4 --duration 5.0 --stage1-ckpt outputs/stage1_best.pt

# 5. Evaluate
python -m src.eval.evaluate_stage2 --threshold-mode eer

# 6. Visualize metrics (ROC, DET, confusion matrix, score histogram)
python -m src.eval.visualize_metrics --artifacts outputs/eval/threshold_eer/stage2_eval_artifacts.npz --run-name eer_threshold

# 7. Visualize features (mismatch, cosine sim, CSAI residuals, PCA, t-SNE)
python -m src.eval.visualize_features --artifacts outputs/eval/threshold_eer/stage2_eval_artifacts.npz --run-name eer_threshold

# 8. Single-file inference (after training)
python -m src.infer.predict_audio --audio path/to/audio.wav --threshold 0.5
```

---

## 8. All Bugs Fixed (do NOT re-introduce these)

| File | Bug | Fix Applied |
|------|-----|-------------|
| `src/losses/stage1_loss.py` | Used MSE instead of Frobenius norm; no BatchNorm normalization; λ=0.1 instead of 0.007 | Rewrote to exact Algorithm 1 |
| `src/models/compression.py` | Simple Conv1d projection (1024→512→256); no bottleneck | Rewrote as bottleneck autoencoder (1024→256→1024→256) per paper Fig. 7 |
| `src/eval/evaluate_stage2.py` | `duration_sec=2.0` hardcoded | Changed to 5.0 |
| `src/eval/evaluate_stage2.py` | `style_res_vec`/`ling_res_vec` not collected or saved | Added both to collection and NPZ |
| `src/eval/visualize_features.py` | Wrong default `--artifacts` path (missing `threshold_eer/` subdir) | Fixed to `outputs/eval/threshold_eer/stage2_eval_artifacts.npz` |
| `src/eval/visualize_features.py` | No CSAI residual visualization | Added 3 CSAI plots |
| `src/eval/visualize_metrics.py` | Same wrong default `--artifacts` path | Fixed |
| `src/infer/predict_audio.py` | `duration_sec=2.0` hardcoded | Changed to 5.0 |
| `src/train/train_stage1.py` | Adam optimizer, no scheduler, batch=1, epochs=3, 2s audio | AdamW, linear scheduler, batch=16, epochs=50, 5s |
| `src/train/train_stage2.py` | Adam optimizer, no scheduler, batch=1, epochs=5, 2s audio, no cross_attn load | AdamW, linear scheduler, batch=2, epochs=10, 5s, loads cross_attn |
| `src/data/prepare_metadata.py` | Used FoR dataset (wrong); no 50% sampling | ASVspoof2019+MILABS, 50% stratified sampling |
| `src/models/stage1_slim.py` | No cross_attn, no residual outputs | Added CrossSubspaceAttention |
| `src/models/stage2_slim.py` | 4 feature streams (1024-dim), no residuals | 6 feature streams (1536-dim), includes CSAI residuals |
| Deleted stubs | `baseline_model.py`, `test_paper_ssl.py`, `train_baseline.py` were empty files | Deleted |

---

## 9. What Is NOT Yet Done (Future Work)

1. **Out-of-domain evaluation on MLAAD** — the `MLAAD/` directory has 38 language fake samples but no real pairs. Would need to pair with real speech (e.g., CommonVoice) to create OOD test set.
2. **Multi-GPU training** — paper used 4 GPUs. Current setup is single Titan Xp (11GB).
3. **Data augmentation in Stage 2** — paper used Noise/Reverb/SpecAug. Not implemented.
4. **Baseline comparison** — no W2V-ASP or other baseline implemented for comparison table.
5. **Cross-domain generalization numbers** — In-the-wild dataset not available locally.

---

## 10. Key Files to Read for Context

In order of importance:
1. `PROJECT_DESCRIPTION.md` — logical rationale for every design decision
2. `ARCHITECTURE_FLOWCHART.md` — full ASCII flowchart of Stage 1 and Stage 2
3. `src/models/cross_subspace_attention.py` — the novel CSAI module (~80 lines)
4. `src/models/stage2_slim.py` — full Stage 2 model with 6 feature streams
5. `src/losses/stage1_loss.py` — exact Algorithm 1 implementation
6. `RUN_SEQUENCE.md` — step-by-step run commands

---

## 11. Expected Results After Training

Based on the paper's Table 2 for the "Full" SLIM variant (frozen frontend):
- **ASVspoof 2019 eval:** EER ~0.6%, F1 ~0.969
- **CSAI should match or improve** on out-of-domain data because residuals provide
  a more discriminative mismatch signal than fixed L2 distance

Current checkpoints in `outputs/` are from the **old (wrong) codebase** — retrain from scratch using the run sequence above.

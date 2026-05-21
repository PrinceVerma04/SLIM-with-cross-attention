# SLIM-XA: Architecture Flowchart
## Cross-Attention Extended SLIM for Generalized Audio Deepfake Detection

---

## STAGE 1  —  One-class Self-Supervised Training  (real speech only)

```
  Real Speech Waveform  [B, 1, T]   (5 seconds @ 16 kHz = 80,000 samples)
          │
          ├──────────────────────────────────────────────────────────────┐
          ▼                                                              ▼
  ┌─────────────────────────┐                        ┌─────────────────────────────┐
  │     Style Encoder       │  ← FROZEN              │    Linguistic Encoder       │ ← FROZEN
  │  Wav2Vec2-XLSR-SER      │                        │  Wav2Vec2-XLSR-ASR          │
  │  (emotion recognition)  │                        │  (speech recognition)       │
  │  layers  0 – 10  (K=11) │                        │  layers  14 – 21  (K=8)    │
  └─────────────────────────┘                        └─────────────────────────────┘
          │  [B, 11, 1024, T']                                          │  [B, 8, 1024, T']
          │                                                             │
          ▼  avg(layers) → [B, 1024, T']                               ▼  avg(layers) → [B, 1024, T']
  ┌─────────────────────────┐                        ┌─────────────────────────────┐
  │   Compression Module    │  ← TRAINABLE           │   Compression Module        │ ← TRAINABLE
  │   Bottleneck Autoencoder│                        │   Bottleneck Autoencoder    │
  │   1024 → 256 → 1024 → 256 (BN+LeakyReLU+Drop)  │   1024 → 256 → 1024 → 256  │
  └─────────────────────────┘                        └─────────────────────────────┘
          │  style_dep [B, 256, T']                                     │  ling_dep [B, 256, T']
          │                                                             │
          └────────────────────────┬────────────────────────────────────┘
                                   ▼
                 ┌─────────────────────────────────────┐
                 │    CrossSubspaceAttention           │  ← TRAINABLE  ★ NOVEL (SLIM-XA) ★
                 │                                     │
                 │  style → ling  (MultiheadAttn, 4h): │
                 │    Q = style_dep  [B, T', 256]       │
                 │    K = ling_dep   [B, T', 256]       │
                 │    V = ling_dep   [B, T', 256]       │
                 │    → style_pred   [B, T', 256]       │
                 │    → style_res = LayerNorm(style_dep - style_pred)  │
                 │                                     │
                 │  ling → style  (MultiheadAttn, 4h): │
                 │    Q = ling_dep   [B, T', 256]       │
                 │    K = style_dep  [B, T', 256]       │
                 │    V = style_dep  [B, T', 256]       │
                 │    → ling_pred    [B, T', 256]       │
                 │    → ling_res  = LayerNorm(ling_dep  - ling_pred )  │
                 └─────────────────────────────────────┘
                          │                       │
                          ▼                       ▼
                   style_residual           ling_residual
                   [B, 256, T']             [B, 256, T']
                   (≈ 0 for real,           (≈ 0 for real,
                    large for fake)          large for fake)

  ─────────────────────────────────────────────────────────────────────────
  STAGE 1 LOSS  (Algorithm 1 from paper + SLIM-XA attention term):

    1.  x_norm = BatchNorm1d(x, affine=False) / batch_size    ← paper normalization
    2.  L_cross = ||style_norm - ling_norm||_F²               ← Frobenius norm
    3.  L_intra = ||S^T·S - I||_F² + ||L^T·L - I||_F²        ← intra-subspace redundancy  λ=0.007
    4.  L_attn  = ||style_residual||² + ||ling_residual||²    ← SLIM-XA novelty  λ=0.1
    5.  L_total = L_cross + 0.007 × L_intra + 0.1 × L_attn

    Optimizer: AdamW  |  LR: 5e-3 → 1e-4 (linear, 5% warmup)  |  Batch: 16
    Patience: 3  |  Max epochs: 50  |  Grad clip: 1.0

    Saves: outputs/stage1_best.pt
           keys: {style_compression, ling_compression, cross_attn, epoch, loss}
  ─────────────────────────────────────────────────────────────────────────
```

---

## STAGE 2  —  Supervised Binary Classification  (real + fake)

```
  Waveform [B, 1, T]  (5 seconds @ 16 kHz)
          │
          ├──────────────────────────────────────────────────────────────┐
          ▼                                                              ▼
  ┌───────────────────┐  FROZEN                         ┌──────────────────────────┐  FROZEN
  │   Style Encoder   │                                 │   Linguistic Encoder     │
  │   (XLSR-SER)      │                                 │   (XLSR-ASR)             │
  └───────────────────┘                                 └──────────────────────────┘
      │ [B, 11, 1024, T']                                   │ [B, 8, 1024, T']
      │                                                     │
      ├── avg(K) ──→ [B, 1024, T'] ───────────────────────┤── avg(K) ──→ [B, 1024, T']
      │                    │                               │                    │
      │         ┌──────────┴──────────┐           ┌───────┘         ┌──────────┴──────────┐
      │         │   ASP + MLP         │ TRAINABLE  │                │   ASP + MLP         │ TRAINABLE
      │         │  [B,1024,T']→[B,2048]            │                │  [B,1024,T']→[B,2048]
      │         │   Linear(2048→512)  │             │               │   Linear(2048→512)  │
      │         │   LeakyReLU + Drop  │             │               │   LeakyReLU + Drop  │
      │         │   Linear(512→256)   │             │               │   Linear(512→256)   │
      │         └──────────┬──────────┘             │               └──────────┬──────────┘
      │           style_raw_vec [B,256]              │                ling_raw_vec [B,256]
      │                                             │
      ▼                                             ▼
  ┌──────────────────────┐  FROZEN           ┌──────────────────────────┐  FROZEN
  │  Compression Module  │                   │  Compression Module      │
  │  (from Stage 1 ckpt) │                   │  (from Stage 1 ckpt)     │
  └──────────────────────┘                   └──────────────────────────┘
      │ style_dep [B, 256, T']                    │ ling_dep [B, 256, T']
      │                                           │
      └────────────────────┬──────────────────────┘
                           ▼
              ┌─────────────────────────────────┐  FROZEN
              │   CrossSubspaceAttention        │  (loaded from Stage 1)  ★ NOVEL ★
              │   (same weights from Stage 1)   │
              └─────────────────────────────────┘
                     │                     │
               mean_pool             mean_pool
                     │                     │
            style_res_vec [B,256]   ling_res_vec [B,256]
                    ★ CSAI residuals — large for fake, small for real ★

       style_dep → mean_pool → style_dep_vec [B, 256]
       ling_dep  → mean_pool → ling_dep_vec  [B, 256]

  ──────────────────────── CONCATENATION ─────────────────────────────

   Stream 1:  style_raw_vec   [B, 256]   ← raw style SSL (signal artifacts)
   Stream 2:  ling_raw_vec    [B, 256]   ← raw linguistic SSL (phonetic artifacts)
   Stream 3:  style_dep_vec   [B, 256]   ← compressed style dependency
   Stream 4:  ling_dep_vec    [B, 256]   ← compressed linguistic dependency
   Stream 5:  style_res_vec   [B, 256]   ← SLIM-XA: style unexplained by linguistics  ★
   Stream 6:  ling_res_vec    [B, 256]   ← SLIM-XA: linguistics unexplained by style  ★
                                          ─────────────────────────────────────────────
                              concat  →   final_feat  [B, 1536]

  ───────────────────────── CLASSIFIER  (TRAINABLE) ──────────────────

   Linear(1536 → 512) → LeakyReLU(0.1) → Dropout(0.25) → Linear(512 → 1)
                                   │
                                   ▼
                              logit  →  sigmoid  →  P(fake) ∈ [0, 1]
                                   │
                              EER threshold = 2.01×10⁻¹⁰

  STAGE 2 LOSS:   BCEWithLogitsLoss
  Optimizer:      AdamW  |  LR: 1e-4 → 1e-5 (linear)  |  Batch: 2  |  Patience: 3
  Saves:          outputs/stage2_best.pt   (full model state_dict, ~2.4 GB)
  ────────────────────────────────────────────────────────────────────
```

---

## Evaluation & Inference Pipeline

```
  outputs/stage2_best.pt
          │
          ▼
  ┌───────────────────────────────────┐
  │   evaluate_stage2.py             │
  │   35,618 test samples (50%)      │
  │   ASVspoof 2019 LA eval          │
  └───────────────────────────────────┘
          │
          ├── EER, F1, Precision, Recall, Accuracy
          ├── ROC curve  (roc_fpr, roc_tpr)
          ├── DET curve  (det_fpr, det_fnr)
          └── NPZ artifact  (labels, scores, logits, all 6 feature vectors)
                  │
          ┌───────┴──────────┐
          ▼                  ▼
  visualize_metrics.py  visualize_features.py
  ┌──────────────────┐  ┌────────────────────────────────────┐
  │ ROC curve        │  │ Mismatch L2 histogram              │
  │ DET curve        │  │ Dependency cosine similarity hist. │
  │ Confusion matrix │  │ Pearson correlation histogram      │
  │ Score histogram  │  │ PCA of 1536-dim feature space      │
  └──────────────────┘  │ t-SNE of 1536-dim feature space    │
                        │ Sample explanation plot + report   │
                        │ CSAI residual norm histograms  ★   │
                        └────────────────────────────────────┘

  Single-file inference:
  python -m src.infer.predict_audio --audio path/to/file.wav
    → Prediction: REAL / FAKE  |  Confidence: 0.xxxx
```

---

## SLIM (Paper) vs SLIM-XA (Ours) — Side-by-Side

```
┌───────────────────────────────────┬──────────────────────────────────────────────────┐
│      Original SLIM (paper)        │              SLIM-XA (ours)                      │
├───────────────────────────────────┼──────────────────────────────────────────────────┤
│ SSL encoders: same (frozen)       │ SSL encoders: same (frozen)                      │
│ Compression: MLP 1024→256 only    │ Compression: bottleneck 1024→256→1024→256        │
│ Stage 1 loss: Frobenius + intra   │ Stage 1 loss: Frobenius + intra + L_attn  ★      │
│ Mismatch signal: L2/cosine dist   │ Mismatch signal: cross-attention RESIDUALS  ★    │
│ Feature streams: 4 × 256 = 1024   │ Feature streams: 6 × 256 = 1536  ★              │
│ Classifier: Linear(1024→512→1)    │ Classifier: Linear(1536→512→1)                  │
│ No attention module               │ CrossSubspaceAttention (4 heads, trained S1) ★  │
│ Stage 1 data: CommonVoice+RAVDESS │ Stage 1 data: ASVspoof 2019 bon. + MILABS        │
│ Stage 2 data: ASVspoof 2019 LA    │ Stage 2 data: ASVspoof 2019 LA (50%)             │
│ Training: 4 GPUs, full dataset    │ Training: 1× Titan Xp, 50% data                 │
│ EER (ASVspoof 2019 eval): 0.60%   │ EER (ASVspoof 2019 eval): 2.62%                 │
│ F1: 0.969                         │ F1: 0.985                                        │
└───────────────────────────────────┴──────────────────────────────────────────────────┘
★ = novel contribution of SLIM-XA
```

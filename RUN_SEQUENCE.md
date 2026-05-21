# SLIM-CSAI: Complete Run Sequence

Environment: `wildspoof_sasv` conda env
Python: `/home/princeverma/miniconda3/envs/wildspoof_sasv/bin/python`
GPU: NVIDIA Titan Xp (11 GB VRAM) — CUDA available via PyTorch 2.2.2

---

## Step 0: Activate Environment

```bash
conda activate wildspoof_sasv
cd "/media/princeverma/044e2d0c-caca-4d10-867b-0cc38ebc5ac4/wildspoof_dataset/new dataset/slim/SLIM"
```

---

## Step 1: Check GPU

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name}  |  VRAM: {p.total_memory // 1024**3} GB')
    print(f'CUDA capability: sm_{p.major}{p.minor}')
"
```

Expected output:
```
PyTorch: 2.2.2
CUDA available: True
GPU: NVIDIA TITAN Xp  |  VRAM: 11 GB
CUDA capability: sm_61
```

---

## Step 2: Prepare Data Splits (50% of each dataset)

```bash
python src/data/prepare_metadata.py
```

Expected output:
```
Building Stage 1 split (real only, 50% sample) …
  stage1_real.csv           total=  6890  real=  6890  fake=     0
Building Stage 2 splits (50% sample, class-balanced) …
  stage2_train.csv          total= 12690  real=  1290  fake= 11400
  stage2_val.csv            total= 12442  real=  1274  fake= 11168
  stage2_test.csv           total= 35618  real=  3677  fake= 31941
All splits written to data/splits/
```

---

## Step 3: Train Stage 1 (~1 hour on Titan Xp)

```bash
python -m src.train.train_stage1 \
    --batch-size 16 \
    --epochs 50 \
    --lr 5e-3 \
    --end-lr 1e-4 \
    --lambda-intra 0.007 \
    --lambda-attn 0.1 \
    --patience 3 \
    --num-workers 4 \
    --duration 5.0
```

What it trains: `style_compression` + `ling_compression` + `cross_attn`
Saves to: `outputs/stage1_best.pt`

Monitor: watch for `cross` loss decreasing (style-ling alignment) and
`attn` loss decreasing (residuals shrinking for real speech).

---

## Step 4: Train Stage 2 (~3-4 hours on Titan Xp with 50% data)

```bash
python -m src.train.train_stage2 \
    --batch-size 2 \
    --epochs 10 \
    --lr 1e-4 \
    --end-lr 1e-5 \
    --patience 3 \
    --num-workers 4 \
    --duration 5.0 \
    --stage1-ckpt outputs/stage1_best.pt
```

What it trains: `style_asp` + `ling_asp` + `style_projection` +
               `ling_projection` + `classifier`
Frozen: SSL encoders, compression modules, cross_attn
Saves to: `outputs/stage2_best.pt`

---

## Step 5: Evaluate on Test Set

```bash
python -m src.eval.evaluate_stage2 \
    --threshold-mode eer
```

Outputs:
- `outputs/eval/eer_threshold/metrics_summary.json`  (EER, F1, accuracy)
- `outputs/eval/eer_threshold/predictions.csv`

---

## Step 6: Visualize Features and Metrics

```bash
# Feature distribution plots (mismatch histograms, PCA, t-SNE)
python -m src.eval.visualize_features

# Confusion matrix, ROC curve, DET curve, score histogram
python -m src.eval.visualize_metrics
```

Outputs saved to `outputs/plots/`

---

## Full One-Liner (if running overnight)

```bash
conda activate wildspoof_sasv && \
cd "/media/princeverma/044e2d0c-caca-4d10-867b-0cc38ebc5ac4/wildspoof_dataset/new dataset/slim/SLIM" && \
python src/data/prepare_metadata.py && \
python -m src.train.train_stage1 --batch-size 16 --epochs 50 --lr 5e-3 --end-lr 1e-4 --lambda-intra 0.007 --lambda-attn 0.1 --patience 3 --num-workers 4 --duration 5.0 && \
python -m src.train.train_stage2 --batch-size 2 --epochs 10 --lr 1e-4 --end-lr 1e-5 --patience 3 --num-workers 4 --duration 5.0 && \
python -m src.eval.evaluate_stage2 --threshold-mode eer && \
python -m src.eval.visualize_features && \
python -m src.eval.visualize_metrics
```

---

## Notes

- The `wildspoof_sasv` env has: torch 2.2.2, CUDA 11.x (sm_61 compatible),
  transformers 4.57.6, librosa 0.11.0
- GPU is auto-detected in all training scripts via `torch.cuda.is_available()`
- `drop_last=True` is set in DataLoader so BatchNorm never sees batch_size=1
- If Stage 1 stops early (patience=3), the best checkpoint is still saved
- Stage 2 loads `cross_attn` weights from the Stage 1 checkpoint

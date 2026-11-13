"""
Stage 2 training for SLIM-CSAI.

Trains:  style_asp, ling_asp, style_projection, ling_projection, classifier
Frozen:  style_encoder, ling_encoder, style_compression, ling_compression, cross_attn

Loss: BCEWithLogitsLoss

Hyperparams follow Table 5 of the SLIM paper.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.audio_dataset import AudioDataset
from src.models.stage2_slim import Stage2SLIM
from src.eval.metrics import compute_classification_metrics


def collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    labels    = torch.stack([item["label"]    for item in batch])
    return waveforms, labels


def get_linear_schedule(optimizer, num_warmup_steps, num_training_steps,
                        start_lr, end_lr):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(end_lr / start_lr, 1.0 - progress * (1.0 - end_lr / start_lr))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Eval", leave=False):
            waveforms = waveforms.to(device)
            labels    = labels.to(device).float()

            outputs = model(waveforms)
            loss    = criterion(outputs["logits"], labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs["logits"]) >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics  = compute_classification_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--end-lr",      type=float, default=1e-5)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--duration",    type=float, default=5.0)
    parser.add_argument("--stage1-ckpt", type=str,
                        default="outputs/stage1_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_dataset = AudioDataset("data/splits/stage2_train.csv", duration_sec=args.duration)
    val_dataset   = AudioDataset("data/splits/stage2_val.csv",   duration_sec=args.duration)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = Stage2SLIM().to(device)

    # Load Stage 1 checkpoint
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    model.style_compression.load_state_dict(ckpt["style_compression"])
    model.ling_compression.load_state_dict(ckpt["ling_compression"])
    model.cross_attn.load_state_dict(ckpt["cross_attn"])
    print(f"Loaded Stage 1 checkpoint from {args.stage1_ckpt} (epoch {ckpt.get('epoch','?')})")

    # Freeze everything except Stage 2 trainable modules
    for module in [model.style_encoder, model.ling_encoder,
                   model.style_compression, model.ling_compression,
                   model.cross_attn]:
        for p in module.parameters():
            p.requires_grad = False

    trainable_params = (
        list(model.style_asp.parameters()) +
        list(model.ling_asp.parameters()) +
        list(model.style_projection.parameters()) +
        list(model.ling_projection.parameters()) +
        list(model.classifier.parameters())
    )
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_linear_schedule(optimizer, warmup_steps, total_steps,
                                       args.lr, args.end_lr)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    best_val_f1        = 0.0
    epochs_no_improve  = 0

    for epoch in range(args.epochs):
        # Frozen modules stay in eval mode to disable dropout/BN updates
        model.train()
        for module in [model.style_encoder, model.ling_encoder,
                       model.style_compression, model.ling_compression,
                       model.cross_attn]:
            module.eval()

        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{args.epochs}")

        for waveforms, labels in pbar:
            waveforms = waveforms.to(device)
            labels    = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(waveforms)
            loss    = criterion(outputs["logits"], labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train loss: {avg_train_loss:.4f}   Val loss: {val_loss:.4f}")
        print(f"  Val acc:    {val_metrics['accuracy']:.4f}   Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val prec:   {val_metrics['precision']:.4f}   Val recall: {val_metrics['recall']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1       = val_metrics["f1"]
            epochs_no_improve = 0
            save_path = output_dir / "stage2_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}  (F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()

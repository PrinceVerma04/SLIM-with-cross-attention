"""
Stage 1 training for SLIM-CSAI.

Trains: style_compression, ling_compression, cross_attn
Frozen: style_encoder, ling_encoder

Loss = stage1_loss (Frobenius cross-subspace + intra-subspace redundancy)
     + lambda_attn * cross-attention reconstruction loss

Hyperparams follow Table 5 of the SLIM paper.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader

from src.data.audio_dataset import AudioDataset
from src.models.stage1_slim import Stage1SLIM
from src.losses.stage1_loss import stage1_loss


def collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    return waveforms


def get_linear_schedule(optimizer, num_warmup_steps, num_training_steps,
                        start_lr, end_lr):
    """Linear decay from start_lr to end_lr over num_training_steps."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(end_lr / start_lr, 1.0 - progress * (1.0 - end_lr / start_lr))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=5e-3)
    parser.add_argument("--end-lr",       type=float, default=1e-4)
    parser.add_argument("--lambda-intra", type=float, default=0.007)
    parser.add_argument("--lambda-attn",  type=float, default=0.1,
                        help="Weight for cross-attention reconstruction loss (CSAI novelty)")
    parser.add_argument("--patience",     type=int,   default=3)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--duration",     type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    dataset = AudioDataset("data/splits/stage1_real.csv", duration_sec=args.duration)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,   # BatchNorm requires batch_size > 1
    )

    model = Stage1SLIM().to(device)

    trainable_params = (
        list(model.style_compression.parameters()) +
        list(model.ling_compression.parameters()) +
        list(model.cross_attn.parameters())
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * len(loader)
    warmup_steps = int(0.05 * total_steps)   # 5% warmup
    scheduler = get_linear_schedule(optimizer, warmup_steps, total_steps,
                                    args.lr, args.end_lr)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Stage1 Epoch {epoch+1}/{args.epochs}")

        for waveforms in pbar:
            waveforms = waveforms.to(device)

            optimizer.zero_grad()
            outputs = model(waveforms)

            loss_dict = stage1_loss(
                outputs["style_dep"],
                outputs["ling_dep"],
                lambda_intra=args.lambda_intra,
            )

            # Add CSAI attention reconstruction loss
            total = loss_dict["total_loss"] + args.lambda_attn * outputs["attn_loss"]
            total.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total.item()
            pbar.set_postfix(
                total=f"{total.item():.4f}",
                cross=f"{loss_dict['cross_loss'].item():.4f}",
                intra=f"{loss_dict['intra_loss'].item():.4f}",
                attn=f"{outputs['attn_loss'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} — avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            save_path = output_dir / "stage1_best.pt"
            torch.save({
                "style_compression": model.style_compression.state_dict(),
                "ling_compression": model.ling_compression.state_dict(),
                "cross_attn": model.cross_attn.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
            }, save_path)
            print(f"  Saved best checkpoint → {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()

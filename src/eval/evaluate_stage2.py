import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.audio_dataset import AudioDataset
from src.eval.metrics import compute_classification_metrics, compute_curve_data
from src.models.stage2_slim import Stage2SLIM


def collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])  # [B, 1, T]
    labels = torch.stack([item["label"] for item in batch])        # [B]
    filepaths = [item["filepath"] for item in batch]
    return waveforms, labels, filepaths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--threshold-mode", type=str, choices=["fixed", "eer"], default="fixed")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    if args.run_name is None:
        if args.threshold_mode == "eer":
            run_name = "threshold_eer"
        else:
            run_name = f"threshold_{args.threshold:.2f}".replace(".", "p")
    else:
        run_name = args.run_name

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = AudioDataset("data/splits/stage2_test.csv", duration_sec=5.0)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    model = Stage2SLIM().to(device)
    model.load_state_dict(torch.load("outputs/stage2_best.pt", map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_scores = []
    all_logits = []
    all_filepaths = []
    all_style_raw = []
    all_ling_raw = []
    all_style_dep = []
    all_ling_dep = []
    all_style_res = []
    all_ling_res = []
    all_final_feat = []
    total_loss = 0.0

    with torch.no_grad():
        for waveforms, labels, filepaths in test_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).float()

            outputs = model(waveforms)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            all_filepaths.extend(filepaths)
            all_style_raw.append(outputs["style_raw_vec"].cpu().numpy())
            all_ling_raw.append(outputs["ling_raw_vec"].cpu().numpy())
            all_style_dep.append(outputs["style_dep_vec"].cpu().numpy())
            all_ling_dep.append(outputs["ling_dep_vec"].cpu().numpy())
            all_style_res.append(outputs["style_res_vec"].cpu().numpy())
            all_ling_res.append(outputs["ling_res_vec"].cpu().numpy())
            all_final_feat.append(outputs["final_feat"].cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    base_metrics = compute_classification_metrics(all_labels, (np.asarray(all_scores) >= 0.5).astype(np.int64), all_scores)
    curves = compute_curve_data(all_labels, all_scores)

    if args.threshold_mode == "eer" and "eer_threshold" in base_metrics:
        decision_threshold = float(base_metrics["eer_threshold"])
    else:
        decision_threshold = float(args.threshold)

    style_raw = np.concatenate(all_style_raw, axis=0)
    ling_raw = np.concatenate(all_ling_raw, axis=0)
    style_dep = np.concatenate(all_style_dep, axis=0)
    ling_dep = np.concatenate(all_ling_dep, axis=0)
    style_res = np.concatenate(all_style_res, axis=0)
    ling_res = np.concatenate(all_ling_res, axis=0)
    final_feat = np.concatenate(all_final_feat, axis=0)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    scores_np = np.asarray(all_scores, dtype=np.float32)
    logits_np = np.asarray(all_logits, dtype=np.float32)
    filepaths_np = np.asarray(all_filepaths)
    preds_np = (scores_np >= decision_threshold).astype(np.int64)
    metrics = compute_classification_metrics(labels_np, preds_np, scores_np)

    np.savez_compressed(
        output_dir / "stage2_eval_artifacts.npz",
        labels=labels_np,
        preds=preds_np,
        scores=scores_np,
        logits=logits_np,
        filepaths=filepaths_np,
        style_raw_vec=style_raw,
        ling_raw_vec=ling_raw,
        style_dep_vec=style_dep,
        ling_dep_vec=ling_dep,
        style_res_vec=style_res,
        ling_res_vec=ling_res,
        final_feat=final_feat,
        roc_fpr=curves["roc"]["fpr"],
        roc_tpr=curves["roc"]["tpr"],
        roc_thresholds=curves["roc"]["thresholds"],
        det_fpr=curves["det"]["fpr"],
        det_fnr=curves["det"]["fnr"],
        det_thresholds=curves["det"]["thresholds"],
        threshold_used=np.asarray([decision_threshold], dtype=np.float32),
    )

    summary = {
        "run_name": run_name,
        "threshold_mode": args.threshold_mode,
        "threshold_used": decision_threshold,
        "test_loss": float(avg_loss),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "eer": float(base_metrics.get("eer", float("nan"))),
        "eer_threshold": float(base_metrics.get("eer_threshold", float("nan"))),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "filepath",
            "label",
            "pred",
            "score_fake",
            "logit",
            "threshold_used",
        ])
        for filepath, label, pred, score, logit in zip(filepaths_np, labels_np, preds_np, scores_np, logits_np):
            writer.writerow([filepath, int(label), int(pred), float(score), float(logit), decision_threshold])

    print("\n=== Stage-2 Test Results ===")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if "eer" in base_metrics:
        print(f"EER:       {base_metrics['eer']:.4f}")
        print(f"EER Thr:   {base_metrics['eer_threshold']:.4f}")
    print(f"Decision Thr Used: {decision_threshold:.4f}")
    print(f"Saved eval artifacts to: {output_dir / 'stage2_eval_artifacts.npz'}")
    print(f"Saved metrics summary to: {output_dir / 'metrics_summary.json'}")
    print(f"Saved predictions CSV to: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()

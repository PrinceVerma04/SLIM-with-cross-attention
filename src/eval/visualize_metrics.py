import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import compute_classification_metrics


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(fpr, tpr, eer, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (EER={eer:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_det(fpr, fnr, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, fnr, linewidth=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("DET Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_hist(scores, labels, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    ax.hist(real_scores, bins=40, alpha=0.6, label="REAL", density=True)
    ax.hist(fake_scores, bins=40, alpha=0.6, label="FAKE", density=True)
    ax.set_xlabel("Predicted Probability of FAKE")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default="outputs/eval/threshold_eer/stage2_eval_artifacts.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/metrics")
    parser.add_argument("--run-name", type=str, default="default")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = np.load(args.artifacts, allow_pickle=True)
    labels = artifacts["labels"]
    preds = artifacts["preds"]
    scores = artifacts["scores"]
    roc_fpr = artifacts["roc_fpr"]
    roc_tpr = artifacts["roc_tpr"]
    det_fpr = artifacts["det_fpr"]
    det_fnr = artifacts["det_fnr"]

    metrics = compute_classification_metrics(labels, preds, scores)

    plot_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    plot_roc(roc_fpr, roc_tpr, metrics.get("eer", float("nan")), output_dir / "roc_curve.png")
    plot_det(det_fpr, det_fnr, output_dir / "det_curve.png")
    plot_score_hist(scores, labels, output_dir / "score_histogram.png")

    print(f"Saved metric plots to: {output_dir}")


if __name__ == "__main__":
    main()

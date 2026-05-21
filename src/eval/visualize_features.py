import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1) + 1e-8
    b_norm = np.linalg.norm(b, axis=1) + 1e-8
    return np.sum(a * b, axis=1) / (a_norm * b_norm)


def rowwise_pearson(a, b):
    sims = []
    for row_a, row_b in zip(a, b):
        if np.std(row_a) < 1e-8 or np.std(row_b) < 1e-8:
            sims.append(0.0)
        else:
            sims.append(float(np.corrcoef(row_a, row_b)[0, 1]))
    return np.asarray(sims, dtype=np.float32)


def plot_histogram(real_values, fake_values, title, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(real_values, bins=40, alpha=0.6, density=True, label="REAL")
    ax.hist(fake_values, bins=40, alpha=0.6, density=True, label="FAKE")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(embedding, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for class_name, class_color, class_value in [("REAL", "tab:blue", 0), ("FAKE", "tab:red", 1)]:
        mask = labels == class_value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.7,
            c=class_color,
            label=class_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_explanations(filepaths, labels, scores, mismatch, cosine_sim, pearson_sim, output_path):
    real_mask = labels == 0
    fake_mask = labels == 1

    real_mismatch_mean = float(mismatch[real_mask].mean()) if real_mask.any() else 0.0
    fake_mismatch_mean = float(mismatch[fake_mask].mean()) if fake_mask.any() else 0.0
    real_cos_mean = float(cosine_sim[real_mask].mean()) if real_mask.any() else 0.0
    fake_cos_mean = float(cosine_sim[fake_mask].mean()) if fake_mask.any() else 0.0
    real_pearson_mean = float(pearson_sim[real_mask].mean()) if real_mask.any() else 0.0
    fake_pearson_mean = float(pearson_sim[fake_mask].mean()) if fake_mask.any() else 0.0

    top_indices = np.argsort(-mismatch)[:20]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("SLIM Feature Explanation Summary\n")
        handle.write("===============================\n\n")
        handle.write(f"REAL mismatch mean:   {real_mismatch_mean:.6f}\n")
        handle.write(f"FAKE mismatch mean:   {fake_mismatch_mean:.6f}\n")
        handle.write(f"REAL cosine mean:     {real_cos_mean:.6f}\n")
        handle.write(f"FAKE cosine mean:     {fake_cos_mean:.6f}\n")
        handle.write(f"REAL pearson mean:    {real_pearson_mean:.6f}\n")
        handle.write(f"FAKE pearson mean:    {fake_pearson_mean:.6f}\n\n")
        handle.write("Top 20 highest mismatch samples\n")
        handle.write("--------------------------------\n")

        for idx in top_indices:
            label_name = "REAL" if labels[idx] == 0 else "FAKE"
            handle.write(f"filepath: {filepaths[idx]}\n")
            handle.write(f"label: {label_name}\n")
            handle.write(f"prob_fake: {scores[idx]:.6f}\n")
            handle.write(f"mismatch_l2: {mismatch[idx]:.6f}\n")
            handle.write(f"cosine_similarity: {cosine_sim[idx]:.6f}\n")
            handle.write(f"pearson_similarity: {pearson_sim[idx]:.6f}\n\n")


def plot_sample_explanation(filepaths, labels, scores, mismatch, cosine_sim, pearson_sim, save_path):
    idx = int(np.argmax(mismatch))
    label_name = "REAL" if labels[idx] == 0 else "FAKE"

    real_mask = labels == 0
    fake_mask = labels == 1
    values = [
        mismatch[idx],
        mismatch[real_mask].mean() if real_mask.any() else 0.0,
        mismatch[fake_mask].mean() if fake_mask.any() else 0.0,
        cosine_sim[idx],
        cosine_sim[real_mask].mean() if real_mask.any() else 0.0,
        cosine_sim[fake_mask].mean() if fake_mask.any() else 0.0,
    ]
    labels_x = [
        "Sample mismatch",
        "REAL mean mismatch",
        "FAKE mean mismatch",
        "Sample cosine",
        "REAL mean cosine",
        "FAKE mean cosine",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(values)), values, color=["tab:red", "tab:blue", "tab:red", "tab:red", "tab:blue", "tab:red"])
    ax.set_xticks(np.arange(len(values)), labels_x, rotation=30, ha="right")
    ax.set_title(
        f"Sample Explanation\nLabel={label_name}, ProbFake={scores[idx]:.4f}\n{filepaths[idx]}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default="outputs/eval/threshold_eer/stage2_eval_artifacts.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/features")
    parser.add_argument("--tsne-samples", type=int, default=2000)
    parser.add_argument("--run-name", type=str, default="default")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = np.load(args.artifacts, allow_pickle=True)
    labels = artifacts["labels"]
    filepaths = artifacts["filepaths"]
    scores = artifacts["scores"]
    style_dep = artifacts["style_dep_vec"]
    ling_dep = artifacts["ling_dep_vec"]
    final_feat = artifacts["final_feat"]

    has_csai = "style_res_vec" in artifacts and "ling_res_vec" in artifacts
    if not has_csai:
        print("WARNING: style_res_vec/ling_res_vec not in NPZ — CSAI plots will be skipped.")
        print("         Re-run evaluate_stage2.py to generate a NPZ with CSAI residuals.")
    else:
        style_res = artifacts["style_res_vec"]
        ling_res = artifacts["ling_res_vec"]

    mismatch = np.linalg.norm(style_dep - ling_dep, axis=1)
    cosine_sim = cosine_similarity(style_dep, ling_dep)
    pearson_sim = rowwise_pearson(style_dep, ling_dep)

    real_mask = labels == 0
    fake_mask = labels == 1

    plot_histogram(
        mismatch[real_mask],
        mismatch[fake_mask],
        "Style-Linguistics Mismatch Distribution",
        "L2(style_dep_vec - ling_dep_vec)",
        output_dir / "mismatch_histogram.png",
    )
    plot_histogram(
        cosine_sim[real_mask],
        cosine_sim[fake_mask],
        "Dependency Cosine Similarity",
        "Cosine Similarity",
        output_dir / "cosine_similarity_histogram.png",
    )
    plot_histogram(
        pearson_sim[real_mask],
        pearson_sim[fake_mask],
        "Dependency Pearson Similarity",
        "Pearson Correlation",
        output_dir / "pearson_similarity_histogram.png",
    )

    if has_csai:
        style_res_norm = np.linalg.norm(style_res, axis=1)
        ling_res_norm = np.linalg.norm(ling_res, axis=1)
        csai_residual_cosine = cosine_similarity(style_res, ling_res)
        plot_histogram(
            style_res_norm[real_mask],
            style_res_norm[fake_mask],
            "CSAI: Style Residual Norm (||style - style_pred||)",
            "L2 norm of style attention residual",
            output_dir / "csai_style_residual_norm.png",
        )
        plot_histogram(
            ling_res_norm[real_mask],
            ling_res_norm[fake_mask],
            "CSAI: Linguistic Residual Norm (||ling - ling_pred||)",
            "L2 norm of linguistic attention residual",
            output_dir / "csai_ling_residual_norm.png",
        )
        plot_histogram(
            csai_residual_cosine[real_mask],
            csai_residual_cosine[fake_mask],
            "CSAI: Cosine Similarity of Residuals",
            "Cosine(style_res, ling_res)",
            output_dir / "csai_residual_cosine.png",
        )

    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(final_feat)
    plot_embedding(pca_embedding, labels, "PCA of Final Feature Space", output_dir / "pca_final_feat.png")

    sample_count = min(args.tsne_samples, len(final_feat))
    if sample_count >= 3:
        subset_indices = np.linspace(0, len(final_feat) - 1, sample_count, dtype=int)
        perplexity = min(30, max(2, sample_count // 20), sample_count - 1)
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=42,
        )
        tsne_embedding = tsne.fit_transform(final_feat[subset_indices])
        plot_embedding(tsne_embedding, labels[subset_indices], "t-SNE of Final Feature Space", output_dir / "tsne_final_feat.png")

    write_explanations(
        filepaths,
        labels,
        scores,
        mismatch,
        cosine_sim,
        pearson_sim,
        output_dir / "sample_explanations.txt",
    )
    plot_sample_explanation(
        filepaths,
        labels,
        scores,
        mismatch,
        cosine_sim,
        pearson_sim,
        output_dir / "sample_explanation_plot.png",
    )

    print(f"Saved feature plots to: {output_dir}")


if __name__ == "__main__":
    main()

"""Run fitting, evaluation, or single-image inspection."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from inspection.data import digest, mask_for, split_normal, test_images
from inspection.model import (
    Encoder,
    calibrate,
    extract,
    fit_statistics,
    image_scores,
    load_statistics,
    score_maps,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fit(root, model, report):
    report, model = Path(report), Path(model)
    report.mkdir(parents=True, exist_ok=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    fitting, calibration = split_normal(root)
    encoder = Encoder()
    features = extract(encoder, fitting)
    mean, variance = fit_statistics(features)
    scores = image_scores(score_maps(extract(encoder, calibration), mean, variance))
    threshold = calibrate(scores)
    np.savez_compressed(model, mean=mean, variance=variance, threshold=threshold)
    rows = [
        {"path": str(p.relative_to(root)), "split": split, "sha256": digest(p)}
        for split, paths in [("fit", fitting), ("calibration", calibration)]
        for p in paths
    ]
    write_json(report / "training_manifest.json", rows)
    write_json(
        report / "calibration.json",
        {
            "normal_fit_images": len(fitting),
            "normal_calibration_images": len(calibration),
            "threshold": threshold,
            "alpha": 0.05,
            "scores": scores.tolist(),
            "seed": 42,
            "model_sha256": digest(model),
        },
    )
    print(
        json.dumps({"fit": len(fitting), "calibration": len(calibration), "threshold": threshold})
    )


def evaluate(root, model, report):
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    paths = test_images(root)
    mean, variance, threshold = load_statistics(model)
    maps = score_maps(extract(Encoder(), paths), mean, variance)
    scores = image_scores(maps)
    labels = np.array([p.parent.name != "good" for p in paths])
    masks = np.stack([mask_for(root, p) for p in paths])
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    metrics = {
        "test_images": len(paths),
        "normal_images": int((~labels).sum()),
        "anomalous_images": int(labels.sum()),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "pixel_auroc_at_256": float(roc_auc_score(masks.ravel(), maps.ravel())),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "anomaly_recall": float(tp / (tp + fn)),
        "normal_false_positive_rate": float(fp / (fp + tn)),
        "model_sha256": digest(model),
    }
    metrics["by_defect"] = {
        category: {
            "count": sum(p.parent.name == category for p in paths),
            "flagged": int(predicted[[p.parent.name == category for p in paths]].sum()),
        }
        for category in sorted({p.parent.name for p in paths})
    }
    write_json(report / "metrics.json", metrics)
    with gzip.open(report / "predictions.csv.gz", "wt", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["path", "sha256", "anomalous", "score", "predicted_anomalous"])
        writer.writerows(
            [str(p.relative_to(root)), digest(p), int(y), float(s), int(pred)]
            for p, y, s, pred in zip(paths, labels, scores, predicted, strict=True)
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0].plot(fpr, tpr, label=f"AUROC = {metrics['image_auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Anomaly recall", title="Official bottle test")
    axes[0].legend()
    axes[1].hist(scores[~labels], bins=12, alpha=0.65, label="Normal")
    axes[1].hist(scores[labels], bins=12, alpha=0.65, label="Defective")
    axes[1].axvline(threshold, color="black", linestyle="--", label="Normal calibration threshold")
    axes[1].set(xlabel="Anomaly score", ylabel="Images", title="Score distribution")
    axes[1].legend(fontsize=8)
    fig.savefig(report / "evaluation.png", dpi=140)
    plt.close(fig)

    # Show false alarms, one example per defect, and nearby decisions
    margins = np.abs(scores - threshold)
    selected = [i for i in range(len(paths)) if predicted[i] != labels[i]][:2]
    for category in sorted({p.parent.name for p in paths} - {"good"}):
        candidates = [i for i, p in enumerate(paths) if p.parent.name == category]
        selected.append(min(candidates, key=lambda i: margins[i]))
    for index in np.argsort(margins):
        if len(selected) >= 6:
            break
        if int(index) not in selected:
            selected.append(int(index))
    selected = selected[:6]
    fig, axes = plt.subplots(6, 3, figsize=(9, 15), layout="constrained")
    records = []
    for row, index in enumerate(selected):
        with Image.open(paths[index]) as image:
            rgb = image.convert("RGB").resize((256, 256))
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(masks[index], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(
            maps[index], cmap="magma", alpha=0.6, vmin=0, vmax=float(np.quantile(maps, 0.995))
        )
        axes[row, 0].set_title(f"{paths[index].parent.name}/{paths[index].name}", fontsize=9)
        axes[row, 1].set_title("Reference defect mask", fontsize=9)
        axes[row, 2].set_title(
            f"Score {scores[index]:.2f} | {'flagged' if predicted[index] else 'normal'}", fontsize=9
        )
        for ax in axes[row]:
            ax.axis("off")
        records.append(
            {
                "path": str(paths[index].relative_to(root)),
                "correct": bool(predicted[index] == labels[index]),
            }
        )
    fig.savefig(report / "inspection.png", dpi=100)
    plt.close(fig)
    write_json(report / "inspection_examples.json", records)
    print(json.dumps(metrics, indent=2))


def predict(image, model, output):
    mean, variance, threshold = load_statistics(model)
    heatmap = score_maps(extract(Encoder(), [Path(image)]), mean, variance)[0]
    score = float(image_scores(heatmap[None])[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    with Image.open(image) as opened:
        rgb = opened.convert("RGB").resize((256, 256))
    axes[0].imshow(rgb)
    axes[1].imshow(rgb)
    overlay = axes[1].imshow(heatmap, cmap="magma", alpha=0.6)
    fig.colorbar(overlay, ax=axes[1], label="Anomaly distance")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Score {score:.3f} | threshold {threshold:.3f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(json.dumps({"score": score, "threshold": threshold, "anomalous": score > threshold}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "evaluate", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/bottle"))
    parser.add_argument("--model", default="models/bottle_gaussian.npz")
    parser.add_argument("--report", default="reports/bottle")
    parser.add_argument("--image")
    parser.add_argument("--output", default="reports/prediction.png")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "fit":
        fit(args.data, args.model, args.report)
    elif args.command == "evaluate":
        evaluate(args.data, args.model, args.report)
    elif args.image:
        predict(args.image, args.model, args.output)
    else:
        parser.error("predict requires --image")


if __name__ == "__main__":
    main()

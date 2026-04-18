#!/usr/bin/env python3
"""
Visualise classification predictions on ShapeNet test objects.

Produces a grid of point clouds, each titled with the true and predicted category.
Samples many candidates and picks a mix that includes at least one wrong
prediction (honest visual evidence, not cherry-picked).
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import ShapeNet

from data_shapenet import subsample_points
from models.point_transformer import PointTransformerClassifier


CATEGORIES = ["Airplane", "Chair", "Car", "Table", "Lamp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/ShapeNet")
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--pool_size", type=int, default=40,
                   help="How many candidates to draw from before selecting the grid")
    p.add_argument("--save", type=str, default="vis_classification.png")
    return p.parse_args()


def equal_axes(ax, pos: np.ndarray) -> None:
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    centers = (maxs + mins) / 2
    max_range = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[2] - max_range, centers[2] + max_range)
    ax.set_zlim(centers[1] - max_range, centers[1] + max_range)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()

    dataset = ShapeNet(args.data_root, categories=CATEGORIES, split="test", include_normals=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PointTransformerClassifier(
        in_channels=6, dim=args.dim, num_blocks=args.num_blocks,
        k=args.k, num_classes=len(CATEGORIES),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    pool_ids = random.sample(range(len(dataset)), min(args.pool_size, len(dataset)))
    results = []
    with torch.no_grad():
        for data_idx in pool_ids:
            raw = dataset[data_idx]
            data = subsample_points(raw, args.num_points, use_normals=True)
            cat_idx = raw.category.item() if torch.is_tensor(raw.category) else int(raw.category)
            batch = torch.zeros(data.pos.size(0), dtype=torch.long)
            pred_idx = model(data.x, data.pos, batch).argmax(dim=-1).item()
            results.append((data_idx, cat_idx, pred_idx, data.pos.clone()))

    wrong = [r for r in results if r[2] != r[1]]
    correct = [r for r in results if r[2] == r[1]]
    random.shuffle(correct)

    n_wrong = min(len(wrong), max(1, args.num_samples // 4))
    picked = wrong[:n_wrong] + correct[: args.num_samples - n_wrong]
    random.shuffle(picked)
    picked = picked[: args.num_samples]

    cols = 4
    rows = (len(picked) + cols - 1) // cols
    fig = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    fig.suptitle("Classification: Point Transformer on ShapeNet",
                 fontsize=16, fontweight="bold", y=0.98)

    for plot_idx, (_, cat_idx, pred_idx, pos) in enumerate(picked):
        true_name = CATEGORIES[cat_idx]
        pred_name = CATEGORIES[pred_idx]
        correct_pred = pred_idx == cat_idx
        color = "#2ca02c" if correct_pred else "#d62728"

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection="3d")
        pts = pos.numpy()
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
                   c=pts[:, 1], cmap="viridis", s=4, alpha=0.8, edgecolors="none")
        equal_axes(ax, pts)
        ax.set_title(f"GT: {true_name}   |   Pred: {pred_name}",
                     fontsize=11, color=color, fontweight="bold")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.2)

    total_correct = sum(1 for _, c, p, _ in picked if c == p)
    fig.text(0.5, 0.01,
             f"Grid accuracy: {total_correct}/{len(picked)} "
             f"(pool accuracy: {sum(1 for r in results if r[1]==r[2])}/{len(results)})",
             ha="center", fontsize=11, color="#444444")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Image saved: {args.save}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualise classification predictions on random ShapeNet objects.

Produces a grid of point clouds, each titled with the true and predicted category.
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
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
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--save", type=str, default="vis_classification.png")
    return p.parse_args()


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

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig = plt.figure(figsize=(4.5 * cols, 4 * rows))
    fig.suptitle("Classification: Point Transformer on ShapeNet", fontsize=16, fontweight="bold", y=0.98)

    for plot_idx, data_idx in enumerate(indices):
        raw = dataset[data_idx]
        data = subsample_points(raw, args.num_points, use_normals=True)

        cat_idx = raw.category.item() if torch.is_tensor(raw.category) else int(raw.category)
        true_name = CATEGORIES[cat_idx]

        with torch.no_grad():
            batch = torch.zeros(data.pos.size(0), dtype=torch.long)
            logits = model(data.x, data.pos, batch)
            pred_idx = logits.argmax(dim=-1).item()
        pred_name = CATEGORIES[pred_idx]

        correct = pred_idx == cat_idx
        color = "green" if correct else "red"

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection="3d")
        pos = data.pos.numpy()
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=pos[:, 1], cmap="viridis", s=2, alpha=0.7)
        ax.set_title(f"GT: {true_name}\nPred: {pred_name}", fontsize=10, color=color, fontweight="bold")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Image saved: {args.save}")


if __name__ == "__main__":
    main()

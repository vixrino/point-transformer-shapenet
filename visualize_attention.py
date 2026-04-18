#!/usr/bin/env python3
"""
Visualise attention weights from the Point Transformer on a single object.

For a chosen query point (red), colors every neighbor by the attention weight
it received.  Bright yellow = high attention, dark purple = low.
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import ShapeNet
from torch_geometric.utils import softmax

from data_shapenet import subsample_points
from models.point_transformer import knn_graph_pure, PointTransformerClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/ShapeNet")
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    p.add_argument("--category", type=str, default="Airplane")
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--num_query_points", type=int, default=4,
                   help="Number of query points to visualize")
    p.add_argument("--save", type=str, default="vis_attention.png")
    return p.parse_args()


def extract_attention_weights(model: nn.Module, x: torch.Tensor, pos: torch.Tensor,
                               batch: torch.Tensor, block_idx: int = 0) -> tuple:
    """Run forward up to block_idx and extract attention from that block's PT layer."""
    h = model.embed(x)
    for i, blk in enumerate(model.blocks):
        if i == block_idx:
            h_norm = blk.norm1(h)
            pt = blk.pt
            edge_index = knn_graph_pure(pos, k=pt.k, batch=batch)
            row, col = edge_index

            q_i = pt.lin_q(h_norm)[col]
            k_j = pt.lin_k(h_norm)[row]

            delta_p = pos[col] - pos[row]
            rho = pt.mlp_rho(delta_p)

            attn = pt.mlp_gamma(q_i - k_j + rho)
            attn = softmax(attn, index=col, dim=0)

            attn_scalar = attn.mean(dim=-1)
            return edge_index, attn_scalar, h_norm
        h = blk(h, pos, batch)
    raise ValueError(f"block_idx {block_idx} out of range")


def main() -> None:
    args = parse_args()
    categories = ["Airplane", "Chair", "Car", "Table", "Lamp"]

    dataset = ShapeNet(args.data_root, categories=categories, split="test", include_normals=True)

    cat_id = categories.index(args.category)
    candidates = [i for i in range(len(dataset))
                  if (dataset[i].category.item() if torch.is_tensor(dataset[i].category)
                      else int(dataset[i].category)) == cat_id]
    data_idx = random.choice(candidates) if candidates else 0
    raw = dataset[data_idx]
    data = subsample_points(raw, args.num_points, use_normals=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PointTransformerClassifier(
        in_channels=6, dim=args.dim, num_blocks=args.num_blocks,
        k=args.k, num_classes=len(categories),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = torch.zeros(data.pos.size(0), dtype=torch.long)
    with torch.no_grad():
        edge_index, attn_weights, _ = extract_attention_weights(model, data.x, data.pos, batch, block_idx=0)

    row, col = edge_index.numpy()
    pos = data.pos.numpy()
    attn_np = attn_weights.numpy()

    n_pts = data.pos.size(0)
    query_points = random.sample(range(n_pts), min(args.num_query_points, n_pts))

    cols = len(query_points)
    fig = plt.figure(figsize=(5 * cols, 5))
    fig.suptitle(f"Attention weights — {args.category} (block 0, k={args.k})",
                 fontsize=14, fontweight="bold", y=1.02)

    for plot_i, qp in enumerate(query_points):
        ax = fig.add_subplot(1, cols, plot_i + 1, projection="3d")

        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c="lightgrey", s=2, alpha=0.3)

        neighbor_mask = col == qp
        neighbor_idx = row[neighbor_mask]
        neighbor_attn = attn_np[neighbor_mask]

        if len(neighbor_attn) > 0:
            a_min, a_max = neighbor_attn.min(), neighbor_attn.max()
            if a_max > a_min:
                norm_attn = (neighbor_attn - a_min) / (a_max - a_min)
            else:
                norm_attn = np.ones_like(neighbor_attn)
            colors = plt.cm.plasma(norm_attn)
            ax.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 2], pos[neighbor_idx, 1],
                       c=colors, s=80, edgecolors="black", linewidths=0.5, zorder=5)
            for ni, na in zip(neighbor_idx, norm_attn):
                ax.plot3D([pos[qp, 0], pos[ni, 0]],
                          [pos[qp, 2], pos[ni, 2]],
                          [pos[qp, 1], pos[ni, 1]],
                          color=plt.cm.plasma(na), alpha=0.6, linewidth=1.5)

        ax.scatter([pos[qp, 0]], [pos[qp, 2]], [pos[qp, 1]],
                   c="red", s=120, marker="*", edgecolors="darkred", linewidths=1, zorder=10)

        ax.set_title(f"Query point #{qp}", fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Image saved: {args.save}")


if __name__ == "__main__":
    main()

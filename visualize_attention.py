#!/usr/bin/env python3
"""
Visualise attention weights from the Point Transformer.

Two-row figure:
- Top row: full object in grey, query point highlighted in red + its k neighbors
  as a halo, so the reader can see WHERE the query is located on the object.
- Bottom row: zoomed-in view of the local neighborhood with neighbors colored
  by attention weight (bright = high attention, dark = low).
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
    p.add_argument("--category", type=str, default="Chair")
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--num_query_points", type=int, default=3)
    p.add_argument("--save", type=str, default="vis_attention.png")
    return p.parse_args()


def extract_attention_weights(model: nn.Module, x: torch.Tensor, pos: torch.Tensor,
                               batch: torch.Tensor, block_idx: int = 0) -> tuple:
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
            return edge_index, attn.mean(dim=-1)
        h = blk(h, pos, batch)
    raise ValueError(f"block_idx {block_idx} out of range")


def equal_box(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


def pick_spread_queries(pos: np.ndarray, n: int, rng: random.Random) -> list[int]:
    """Pick n query points that are far from each other for visual diversity."""
    n_pts = len(pos)
    first = rng.randrange(n_pts)
    picked = [first]
    for _ in range(n - 1):
        picked_pos = pos[picked]
        dists = np.linalg.norm(pos[:, None, :] - picked_pos[None, :, :], axis=-1).min(axis=1)
        picked.append(int(np.argmax(dists)))
    return picked


def main() -> None:
    args = parse_args()
    rng = random.Random()
    categories = ["Airplane", "Chair", "Car", "Table", "Lamp"]

    dataset = ShapeNet(args.data_root, categories=categories, split="test", include_normals=True)

    cat_id = categories.index(args.category)
    candidates = [i for i in range(len(dataset))
                  if (dataset[i].category.item() if torch.is_tensor(dataset[i].category)
                      else int(dataset[i].category)) == cat_id]
    data_idx = rng.choice(candidates) if candidates else 0
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
        edge_index, attn_weights = extract_attention_weights(model, data.x, data.pos, batch, block_idx=0)

    row, col = edge_index.numpy()
    pos = data.pos.numpy()
    attn_np = attn_weights.numpy()

    query_points = pick_spread_queries(pos, args.num_query_points, rng)

    obj_center = (pos.max(axis=0) + pos.min(axis=0)) / 2
    obj_radius = (pos.max(axis=0) - pos.min(axis=0)).max() / 2 * 1.05

    cols = len(query_points)
    fig = plt.figure(figsize=(5.5 * cols, 11))
    fig.suptitle(f"Attention weights — {args.category} (Point Transformer, block 0, k={args.k})",
                 fontsize=16, fontweight="bold", y=0.98)

    for plot_i, qp in enumerate(query_points):
        neighbor_mask = col == qp
        neighbor_idx = row[neighbor_mask]
        neighbor_attn = attn_np[neighbor_mask]
        a_min, a_max = neighbor_attn.min(), neighbor_attn.max()
        norm_attn = (neighbor_attn - a_min) / (a_max - a_min + 1e-9)

        # ── TOP ROW : global context ────────────────────────────────────────
        ax_top = fig.add_subplot(2, cols, plot_i + 1, projection="3d")
        ax_top.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                       c="#b0b0b0", s=8, alpha=0.55, edgecolors="none", zorder=1)
        ax_top.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 2], pos[neighbor_idx, 1],
                       c="#1f77b4", s=80, alpha=0.95, edgecolors="white", linewidths=1.0, zorder=5)
        ax_top.scatter([pos[qp, 0]], [pos[qp, 2]], [pos[qp, 1]],
                       c="red", s=400, marker="*",
                       edgecolors="black", linewidths=1.5, zorder=10)
        equal_box(ax_top, obj_center, obj_radius)
        ax_top.view_init(elev=20, azim=-60)
        ax_top.set_title(f"Query #{qp} — location on the object", fontsize=11, pad=8)
        ax_top.set_xticklabels([]); ax_top.set_yticklabels([]); ax_top.set_zticklabels([])
        ax_top.grid(True, alpha=0.2)

        # ── BOTTOM ROW : local zoom with attention ──────────────────────────
        ax_bot = fig.add_subplot(2, cols, cols + plot_i + 1, projection="3d")

        local_pts = pos[list(neighbor_idx) + [qp]]
        local_center = local_pts.mean(axis=0)
        local_radius = np.linalg.norm(local_pts - local_center, axis=1).max() * 1.6 + 1e-3

        nearby_mask = np.linalg.norm(pos - local_center, axis=1) < local_radius * 1.5
        ax_bot.scatter(pos[nearby_mask, 0], pos[nearby_mask, 2], pos[nearby_mask, 1],
                       c="#d8d8d8", s=6, alpha=0.4, edgecolors="none", zorder=1)

        for ni, na in zip(neighbor_idx, norm_attn):
            ax_bot.plot3D([pos[qp, 0], pos[ni, 0]],
                          [pos[qp, 2], pos[ni, 2]],
                          [pos[qp, 1], pos[ni, 1]],
                          color=plt.cm.plasma(na), alpha=0.95, linewidth=3.5, zorder=3)

        colors = plt.cm.plasma(norm_attn)
        ax_bot.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 2], pos[neighbor_idx, 1],
                       c=colors, s=340, edgecolors="black", linewidths=1.2, zorder=5)

        ax_bot.scatter([pos[qp, 0]], [pos[qp, 2]], [pos[qp, 1]],
                       c="red", s=600, marker="*",
                       edgecolors="black", linewidths=1.8, zorder=10)

        equal_box(ax_bot, local_center, local_radius)
        ax_bot.view_init(elev=20, azim=-60)
        ax_bot.set_title(f"Query #{qp} — zoom on k={args.k} neighbors", fontsize=11, pad=8)
        ax_bot.set_xticklabels([]); ax_bot.set_yticklabels([]); ax_bot.set_zticklabels([])
        ax_bot.grid(True, alpha=0.2)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.02, location="right")
    cbar.set_label("Attention weight (normalized per query)", fontsize=11)

    # legend
    fig.text(0.02, 0.02,
             "red star = query point   |   blue dots / colored dots = k nearest neighbors   "
             "|   edge color = attention weight",
             fontsize=10, color="#333333")

    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Image saved: {args.save}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Attention visualization for the Point Transformer.

Produces a 2 x N_QUERIES figure:
- Top row: full point cloud in light grey, query point as a red star and its
  k=8 nearest neighbors as blue dots. All three top subplots share the same
  axis limits and camera angle to stay visually comparable.
- Bottom row: local zoom around each query. Neighbors and query-neighbor edges
  are colored by attention weight (normalized per query, plasma colormap).

Queries are picked on semantically distinct parts of the object (e.g. chair
back, seat, leg) using the ShapeNet part segmentation labels.
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

from models.point_transformer import knn_graph_pure, PointTransformerClassifier


SHAPENET_PART_NAMES: dict[int, str] = {
    0: "body", 1: "wing", 2: "tail", 3: "engine",
    4: "handle", 5: "body",
    6: "panels", 7: "peak",
    8: "roof", 9: "hood", 10: "wheel", 11: "body",
    12: "backrest", 13: "seat", 14: "leg", 15: "armrest",
    16: "earcup", 17: "headband", 18: "data-wire",
    19: "head", 20: "neck", 21: "body",
    22: "blade", 23: "handle",
    24: "base", 25: "pole", 26: "lampshade", 27: "canopy",
    28: "keyboard", 29: "screen",
    30: "gas-tank", 31: "seat", 32: "wheel", 33: "handle",
    34: "light", 35: "frame",
    36: "handle", 37: "cup",
    38: "barrel", 39: "handle", 40: "trigger",
    41: "nose", 42: "body", 43: "fin",
    44: "wheel", 45: "deck", 46: "truck",
    47: "top", 48: "leg", 49: "support",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/ShapeNet")
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    p.add_argument("--category", type=str, default="Chair")
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n_queries", type=int, default=3)
    p.add_argument("--save", type=str, default="vis_attention.png")
    p.add_argument("--seed", type=int, default=None)
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
            rho = pt.mlp_rho(pos[col] - pos[row])
            attn = pt.mlp_gamma(q_i - k_j + rho)
            attn = softmax(attn, index=col, dim=0)
            return edge_index, attn.mean(dim=-1)
        h = blk(h, pos, batch)
    raise ValueError(f"block_idx {block_idx} out of range")


def subsample(pos: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n: int,
              rng: random.Random) -> tuple:
    total = pos.size(0)
    if total <= n:
        idx = torch.tensor(rng.choices(range(total), k=n), dtype=torch.long)
    else:
        idx = torch.tensor(rng.sample(range(total), n), dtype=torch.long)
    return pos[idx], x[idx], y[idx]


def pick_queries_on_distinct_parts(pos: np.ndarray, labels: np.ndarray,
                                    n_queries: int, rng: random.Random) -> list[int]:
    unique_parts, counts = np.unique(labels, return_counts=True)
    parts_sorted = unique_parts[np.argsort(-counts)]

    queries: list[int] = []
    for part in parts_sorted:
        if len(queries) >= n_queries:
            break
        candidates = np.where(labels == part)[0]
        queries.append(int(rng.choice(candidates.tolist())))

    while len(queries) < n_queries:
        queries.append(rng.randrange(len(pos)))
    return queries


def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def set_equal_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    categories = ["Airplane", "Chair", "Car", "Table", "Lamp"]
    dataset = ShapeNet(args.data_root, categories=categories, split="test", include_normals=True)

    cat_id = categories.index(args.category)
    candidates = [i for i in range(len(dataset))
                  if (dataset[i].category.item() if torch.is_tensor(dataset[i].category)
                      else int(dataset[i].category)) == cat_id]
    data_idx = rng.choice(candidates) if candidates else 0
    raw = dataset[data_idx]

    combined_x = torch.cat([raw.pos, raw.x], dim=-1)
    pos_t, x_t, y_t = subsample(raw.pos, combined_x, raw.y, args.num_points, rng)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PointTransformerClassifier(
        in_channels=6, dim=args.dim, num_blocks=args.num_blocks,
        k=args.k, num_classes=len(categories),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = torch.zeros(pos_t.size(0), dtype=torch.long)
    with torch.no_grad():
        edge_index, attn_weights = extract_attention_weights(model, x_t, pos_t, batch, block_idx=0)

    row, col = edge_index.numpy()
    pos = pos_t.numpy()
    labels = y_t.numpy()
    attn_np = attn_weights.numpy()

    queries = pick_queries_on_distinct_parts(pos, labels, args.n_queries, rng)

    obj_center = (pos.max(axis=0) + pos.min(axis=0)) / 2
    obj_radius = (pos.max(axis=0) - pos.min(axis=0)).max() / 2 * 1.05

    top_elev, top_azim = 22, -55
    bot_elev, bot_azim = 22, -55

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Attention weights — {args.category} (Point Transformer, block 0, k={args.k})",
        fontsize=15, fontweight="bold", y=0.99,
    )

    for col_i, qp in enumerate(queries):
        neighbor_mask = col == qp
        neighbor_idx = row[neighbor_mask]
        neighbor_attn = attn_np[neighbor_mask]
        a_min, a_max = neighbor_attn.min(), neighbor_attn.max()
        norm_attn = (neighbor_attn - a_min) / (a_max - a_min + 1e-9)

        part_id = int(labels[qp])
        part_name = SHAPENET_PART_NAMES.get(part_id, f"part {part_id}")

        ax_top = fig.add_subplot(2, args.n_queries, col_i + 1, projection="3d")
        ax_top.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                       c="#b8b8b8", s=8, alpha=0.30, edgecolors="none", zorder=1)
        ax_top.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 2], pos[neighbor_idx, 1],
                       c="#1f77b4", s=60, alpha=0.95,
                       edgecolors="white", linewidths=0.8, zorder=5)
        ax_top.scatter([pos[qp, 0]], [pos[qp, 2]], [pos[qp, 1]],
                       c="red", s=200, marker="*",
                       edgecolors="black", linewidths=1.0, zorder=10)
        set_equal_limits(ax_top, obj_center, obj_radius)
        ax_top.view_init(elev=top_elev, azim=top_azim)
        ax_top.set_title(f"Query #{qp} — {part_name}", fontsize=11, pad=6)
        hide_axes(ax_top)

        ax_bot = fig.add_subplot(2, args.n_queries, args.n_queries + col_i + 1, projection="3d")

        local_pts = pos[list(neighbor_idx) + [qp]]
        local_center = local_pts.mean(axis=0)
        max_dist = np.linalg.norm(local_pts - pos[qp], axis=1).max()
        local_radius = max_dist * 1.5 + 1e-3

        for ni, na in zip(neighbor_idx, norm_attn):
            ax_bot.plot3D([pos[qp, 0], pos[ni, 0]],
                          [pos[qp, 2], pos[ni, 2]],
                          [pos[qp, 1], pos[ni, 1]],
                          color=plt.cm.plasma(na), alpha=0.95, linewidth=2.0, zorder=3)

        colors = plt.cm.plasma(norm_attn)
        ax_bot.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 2], pos[neighbor_idx, 1],
                       c=colors, s=300, alpha=0.8,
                       edgecolors="black", linewidths=1.0, zorder=5)
        ax_bot.scatter([pos[qp, 0]], [pos[qp, 2]], [pos[qp, 1]],
                       c="red", s=200, marker="*",
                       edgecolors="black", linewidths=1.0, zorder=10)

        set_equal_limits(ax_bot, np.array([pos[qp, 0], pos[qp, 1], pos[qp, 2]]), local_radius)
        ax_bot.view_init(elev=bot_elev, azim=bot_azim)
        ax_bot.set_title(f"Query #{qp} — attention weights", fontsize=11, pad=6)
        hide_axes(ax_bot)

    bottom_axes = [ax for ax in fig.axes[args.n_queries:]]
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=bottom_axes, shrink=0.75, pad=0.02, location="right")
    cbar.set_label("Attention weight (normalized per query)", fontsize=11)

    fig.text(
        0.5, 0.02,
        "Top row: global view, red star = query, blue dots = k=8 neighbors.   "
        "Bottom row: local zoom, dots and edges colored by attention weight.",
        ha="center", fontsize=10, color="#333333",
    )

    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Image saved: {args.save}")


if __name__ == "__main__":
    main()

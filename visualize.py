#!/usr/bin/env python3
"""
Visualise la segmentation prédite vs la vérité terrain sur un objet ShapeNet.

Usage :
    python visualize.py                         # objet aléatoire, toutes catégories
    python visualize.py --category Airplane     # un avion au hasard
    python visualize.py --index 42              # le 42e objet du set de test
    python visualize.py --save pred.png         # sauvegarde au lieu d'afficher
"""

from __future__ import annotations

import argparse
import random

import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import ShapeNet

from models.point_transformer import PointTransformerSegmentor

NUM_SEG_CLASSES = 50
COLORS = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/ShapeNet")
    p.add_argument("--checkpoint", type=str, default="checkpoint_seg_best.pt")
    p.add_argument("--category", type=str, default=None, help="Ex: Airplane, Chair, Car…")
    p.add_argument("--index", type=int, default=None, help="Indice dans le test set")
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--save", type=str, default=None, help="Chemin pour sauvegarder l'image")
    return p.parse_args()


def subsample(data, num_points: int):
    n = data.pos.size(0)
    if n <= num_points:
        idx = torch.randint(0, n, (num_points,))
    else:
        idx = torch.randperm(n)[:num_points]
    data.pos = data.pos[idx]
    if data.x is not None:
        data.x = torch.cat([data.pos, data.x[idx]], dim=-1)
    else:
        data.x = data.pos.clone()
    data.y = data.y[idx]
    return data


def plot_pointcloud(ax, pos, labels, title: str, valid_parts: list[int]) -> None:
    """Affiche un nuage de points coloré par label de pièce."""
    pos = pos.numpy()
    labels = labels.numpy()
    part_to_local = {p: i for i, p in enumerate(sorted(valid_parts))}
    colors = [COLORS[part_to_local.get(l, 0) % len(COLORS)] for l in labels]
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=colors, s=3, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()

    all_cats = list(ShapeNet.category_ids.keys())
    test_set = ShapeNet(
        args.data_root, categories=all_cats, split="test", include_normals=True,
    )

    if args.index is not None:
        idx = args.index
    elif args.category:
        cat_id = all_cats.index(args.category)
        candidates = [i for i in range(len(test_set))
                      if (test_set[i].category.item() if torch.is_tensor(test_set[i].category)
                          else int(test_set[i].category)) == cat_id]
        idx = random.choice(candidates) if candidates else 0
    else:
        idx = random.randint(0, len(test_set) - 1)

    data = test_set[idx]
    data = subsample(data, args.num_points)

    cat_idx = data.category.item() if torch.is_tensor(data.category) else int(data.category)
    cat_name = test_set.categories[cat_idx]
    valid_parts = ShapeNet.seg_classes[cat_name]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = PointTransformerSegmentor(
        in_channels=6, dim=args.dim, num_blocks=args.num_blocks,
        k=args.k, num_seg_classes=NUM_SEG_CLASSES,
        num_categories=len(test_set.categories),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        batch_idx = torch.zeros(data.pos.size(0), dtype=torch.long)
        category = torch.tensor([cat_idx], dtype=torch.long)
        logits = model(data.x, data.pos, batch_idx, category)
        mask = torch.full((NUM_SEG_CLASSES,), -1e9)
        mask[valid_parts] = 0.0
        pred = (logits + mask.unsqueeze(0)).argmax(dim=-1)

    gt = data.y
    accuracy = (pred == gt).float().mean().item() * 100

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"{cat_name} (#{idx}) — accuracy: {accuracy:.1f}%", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(121, projection="3d")
    plot_pointcloud(ax1, data.pos, gt, "Vérité terrain", valid_parts)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_pointcloud(ax2, data.pos, pred, "Prédiction du modèle", valid_parts)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée : {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

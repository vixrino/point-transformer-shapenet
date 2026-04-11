#!/usr/bin/env python3
"""
Entraînement : segmentation de pièces ShapeNet avec Point Transformer.

Chaque point d'un nuage reçoit un label (ex: aile, fuselage, réacteur pour un avion).
Métrique : mIoU (mean Intersection over Union) — le standard du benchmark ShapeNet.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.point_transformer import PointTransformerSegmentor


NUM_SEG_CLASSES = 50  # ShapeNet a 50 labels de pièces au total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point Transformer — ShapeNet part segmentation")
    p.add_argument("--data_root", type=str, default="./data/ShapeNet")
    p.add_argument("--categories", nargs="+", default=None,
                   help="None = toutes les 16 catégories")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def subsample_transform(num_points: int):
    """Sous-échantillonne chaque nuage à N points."""
    def transform(data):
        n = data.pos.size(0)
        if n <= num_points:
            idx = torch.randint(0, n, (num_points,))
        else:
            idx = torch.randperm(n)[:num_points]
        data.pos = data.pos[idx]
        if data.x is not None:
            data.x = torch.cat([data.pos, data.x[idx]], dim=-1)  # [N, 6]
        else:
            data.x = data.pos.clone()
        data.y = data.y[idx]
        return data
    return transform


def compute_miou(predictions: dict, seg_classes: dict) -> tuple[float, dict]:
    """
    Calcule le mIoU par catégorie et le mIoU global (moyenne des catégories).
    predictions : { cat_name: [(pred_labels, true_labels), ...] }
    """
    cat_ious = {}
    for cat_name, parts in seg_classes.items():
        if cat_name not in predictions or len(predictions[cat_name]) == 0:
            continue
        shape_ious = []
        for pred, target in predictions[cat_name]:
            part_ious = []
            for part in parts:
                pred_part = (pred == part)
                target_part = (target == part)
                intersection = (pred_part & target_part).sum().item()
                union = (pred_part | target_part).sum().item()
                if union == 0:
                    part_ious.append(1.0)
                else:
                    part_ious.append(intersection / union)
            shape_ious.append(sum(part_ious) / len(part_ious))
        cat_ious[cat_name] = sum(shape_ious) / len(shape_ious)

    miou = sum(cat_ious.values()) / max(len(cat_ious), 1)
    return miou, cat_ious


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cats = args.categories
    train_set = ShapeNet(
        args.data_root, categories=cats, split="train",
        include_normals=True, transform=subsample_transform(args.num_points),
    )
    val_set = ShapeNet(
        args.data_root, categories=cats, split="val",
        include_normals=True, transform=subsample_transform(args.num_points),
    )

    num_categories = len(train_set.categories)
    cat_to_idx = {cat: i for i, cat in enumerate(train_set.categories)}
    idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = PointTransformerSegmentor(
        in_channels=6,
        dim=args.dim,
        num_blocks=args.num_blocks,
        k=args.k,
        num_seg_classes=NUM_SEG_CLASSES,
        num_categories=num_categories,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start_epoch = 1
    best_miou = 0.0
    checkpoint_path = "checkpoint_seg_best.pt"

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        print(f"Reprise depuis epoch {start_epoch - 1}  (meilleur mIoU={best_miou:.4f})")

    seg_classes = ShapeNet.seg_classes

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.pos, batch.batch, batch.category)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        # ── Validation + mIoU ──
        model.eval()
        predictions = defaultdict(list)
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.pos, batch.batch, batch.category)

                # Masque les classes invalides pour cette catégorie
                sizes = []
                ptr = 0
                for i in range(batch.num_graphs):
                    n_pts = (batch.batch == i).sum().item()
                    sizes.append(n_pts)
                    cat_idx = batch.category[i].item()
                    cat_name = idx_to_cat[cat_idx]
                    valid_parts = seg_classes[cat_name]
                    point_logits = logits[ptr:ptr + n_pts]
                    mask = torch.full((NUM_SEG_CLASSES,), -1e9, device=device)
                    mask[valid_parts] = 0.0
                    point_logits = point_logits + mask.unsqueeze(0)
                    pred = point_logits.argmax(dim=-1).cpu()
                    target = batch.y[ptr:ptr + n_pts].cpu()
                    predictions[cat_name].append((pred, target))
                    ptr += n_pts

        train_loss = total_loss / max(n_batches, 1)
        miou, cat_ious = compute_miou(predictions, seg_classes)
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  mIoU={miou:.4f}")
        for cat, iou in sorted(cat_ious.items()):
            print(f"  {cat:12s}: {iou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best_miou": best_miou,
            }, checkpoint_path)
            print(f"  -> meilleur modèle sauvegardé (mIoU={best_miou:.4f})")

    print(f"\nTerminé. Meilleur mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Entraînement : classification de catégories ShapeNet avec Point Transformer.

Étapes suggérées (apprentissage) :
1) Lire le papier Point Transformer (Zhao et al., ICCV 2021).
2) Installer PyTorch puis torch-geometric (+ torch-scatter, torch-cluster) selon
   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
3) Lancer :  python train.py --data_root ./data/ShapeNet --epochs 30 --categories Airplane Chair Car Table Lamp

Le premier run télécharge ShapeNet (~ Go) depuis le serveur Stanford (peut être lent).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_shapenet import ShapeNetClassification
from models.point_transformer import PointTransformerClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point Transformer — ShapeNet classification")
    p.add_argument("--data_root", type=str, default="./data/ShapeNet", help="Répertoire racine du dataset")
    p.add_argument(
        "--categories",
        nargs="+",
        default=["Airplane", "Chair", "Car", "Table", "Lamp"],
        help="Sous-ensemble de catégories ShapeNet (noms complets, voir ShapeNet.category_ids)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_points", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--k", type=int, default=8, help="Nombre de voisins k-NN")
    p.add_argument("--no_normals", action="store_true", help="Utiliser seulement xyz (3 canaux)")
    p.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier checkpoint")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_normals = not args.no_normals
    in_ch = 3 if not use_normals else 6

    train_set = ShapeNetClassification(
        args.data_root,
        categories=args.categories,
        split="train",
        num_points=args.num_points,
        use_normals=use_normals,
    )
    val_set = ShapeNetClassification(
        args.data_root,
        categories=args.categories,
        split="val",
        num_points=args.num_points,
        use_normals=use_normals,
    )

    num_classes = train_set.num_shape_classes
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = PointTransformerClassifier(
        in_channels=in_ch,
        dim=args.dim,
        num_blocks=args.num_blocks,
        k=args.k,
        num_classes=num_classes,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start_epoch = 1
    best_val_acc = 0.0

    checkpoint_path = "checkpoint_best.pt"
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Reprise depuis epoch {start_epoch - 1}  (meilleure val_acc={best_val_acc:.2f}%)")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.pos, batch.batch)
            loss = F.cross_entropy(logits, batch.y_cls.view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.pos, batch.batch)
                pred = logits.argmax(dim=-1)
                correct += (pred == batch.y_cls.view(-1)).sum().item()
                total += batch.num_graphs
        train_loss = total_loss / max(n_batches, 1)
        val_acc = 100.0 * correct / max(total, 1)
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best_val_acc": best_val_acc,
            }, checkpoint_path)
            print(f"  -> meilleur modèle sauvegardé ({best_val_acc:.2f}%)")

    print(f"\nTerminé. Meilleure val_acc = {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

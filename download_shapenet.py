#!/usr/bin/env python3
"""
Télécharge ShapeNet part-annotation depuis plusieurs miroirs.
Le serveur Stanford est souvent hors-ligne ; on tente d'abord Kaggle (via API)
puis un téléchargement direct avec retries.

Usage :
    python download_shapenet.py --data_root ./data/ShapeNet
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

STANFORD_URL = (
    "https://shapenet.cs.stanford.edu/media/"
    "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
)
KAGGLE_DATASET = "mitkir/shapenet"
ZIP_NAME = "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
EXTRACTED_DIR = "shapenetcore_partanno_segmentation_benchmark_v0_normal"


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)


def try_stanford(dest_zip: Path, retries: int = 3, timeout: int = 30) -> bool:
    """Télécharge depuis Stanford avec retries."""
    import socket

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    for attempt in range(1, retries + 1):
        try:
            print(f"[Stanford] tentative {attempt}/{retries} ...")
            urlretrieve(STANFORD_URL, str(dest_zip), reporthook=progress_hook)
            print()
            return True
        except Exception as e:
            print(f"\n  echec: {e}")
            if attempt < retries:
                time.sleep(3)
    socket.setdefaulttimeout(old_timeout)
    return False


def try_kaggle(dest_dir: Path) -> bool:
    """Télécharge depuis Kaggle (nécessite kaggle CLI + API token)."""
    if shutil.which("kaggle") is None:
        print("[Kaggle] CLI non installée, skip.")
        return False
    print("[Kaggle] téléchargement via kaggle datasets download ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest_dir)],
            check=True,
            timeout=300,
        )
        return True
    except Exception as e:
        print(f"  echec Kaggle: {e}")
        return False


def try_curl(dest_zip: Path, timeout: int = 120) -> bool:
    """Télécharge avec curl (meilleure gestion réseau que urllib)."""
    if shutil.which("curl") is None:
        return False
    print(f"[curl] téléchargement (timeout {timeout}s) ...")
    try:
        subprocess.run(
            ["curl", "-L", "--retry", "3", "--connect-timeout", "20",
             "--max-time", str(timeout), "-o", str(dest_zip), STANFORD_URL],
            check=True,
        )
        return dest_zip.stat().st_size > 1_000_000
    except Exception as e:
        print(f"  echec curl: {e}")
        return False


def extract_zip(zip_path: Path, target_raw: Path) -> None:
    print(f"Extraction vers {target_raw} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_raw.parent)
    extracted = target_raw.parent / EXTRACTED_DIR
    if extracted.exists() and extracted != target_raw:
        if target_raw.exists():
            shutil.rmtree(target_raw)
        extracted.rename(target_raw)
    print("OK.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/ShapeNet")
    args = parser.parse_args()

    root = Path(args.data_root)
    raw_dir = root / "raw"

    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"Données déjà présentes dans {raw_dir}, rien à faire.")
        return

    root.mkdir(parents=True, exist_ok=True)
    dest_zip = root / ZIP_NAME

    if dest_zip.exists() and dest_zip.stat().st_size > 1_000_000:
        print(f"Zip déjà téléchargé : {dest_zip}")
    else:
        ok = try_curl(dest_zip) or try_stanford(dest_zip) or try_kaggle(root)
        if not ok:
            print(
                "\n=== Echec du téléchargement automatique ===\n"
                "Le serveur Stanford est probablement hors-ligne.\n\n"
                "Télécharge manuellement depuis Kaggle :\n"
                "  https://www.kaggle.com/datasets/mitkir/shapenet\n\n"
                f"Puis place le zip dans : {dest_zip}\n"
                "Et relance cette commande."
            )
            sys.exit(1)

    extract_zip(dest_zip, raw_dir)
    dest_zip.unlink(missing_ok=True)
    print("ShapeNet prêt.")


if __name__ == "__main__":
    main()

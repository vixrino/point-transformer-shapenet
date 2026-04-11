# Point Transformer on ShapeNet

PyTorch implementation of **Point Transformer** ([Zhao et al., ICCV 2021](https://arxiv.org/abs/2012.09164)) for 3D point clouds from **ShapeNet** (via [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)).

Two tasks are supported:

- **Shape classification** — predict the object category (e.g. chair, airplane).
- **Part segmentation** — per-point part labels (ShapeNet part benchmark).

The code uses a **pure PyTorch k-NN graph** and PyG’s `scatter` / `softmax` utilities so you do not need `torch-cluster` or `torch-scatter` wheels.

## Requirements

- **Python 3.11 or 3.12** is recommended (PyTorch has stable wheels; 3.13 may not work yet).
- macOS / Linux with enough disk space for ShapeNet (~several GB when unpacked).

## Setup

```bash
git clone <your-repo-url>
cd <repository-folder>
make install
```

This creates `.venv/`, upgrades `pip`, and installs `requirements.txt`.

## Data

Place ShapeNet Part under `./data/ShapeNet` (default), or use the helper:

```bash
make download
# or
.venv/bin/python3 download_shapenet.py --data_root ./data/ShapeNet
```

If the automatic download fails, the script prints mirror / manual options.

## Training

**Classification** (default: a few categories, small model for CPU-friendly runs):

```bash
make train
# resume from checkpoint_best.pt
make train-resume
```

**Part segmentation** (all 16 ShapeNet categories by default):

```bash
make seg
make seg-resume
```

Override hyperparameters via Makefile variables, e.g. `make train EPOCHS=30 K=16 BATCH=32`, or call `train.py` / `train_segmentation.py` with `--help`.

## Sanity check (mesh junction faces)

A small **geometry-only** script (`sanity_check.py`) builds procedural meshes, masks a region, extracts **junction faces** and boundary edges, and saves PNG figures (useful when exploring mesh editing / infill ideas):

```bash
make sanity-check
```

## Repository layout

| Path | Role |
|------|------|
| `models/point_transformer.py` | Point Transformer blocks, classifier, segmentor |
| `data_shapenet.py` | Classification dataset wrapper + subsampling |
| `train.py` | Classification training loop + checkpointing |
| `train_segmentation.py` | Segmentation training + mIoU |
| `sanity_check.py` | Procedural mesh + junction-face visualization |
| `download_shapenet.py` | Dataset download with retries |
| `Makefile` | `install`, `download`, `train`, `seg`, `sanity-check`, `clean` |

## Citation

If you use this for research, cite the original paper:

```bibtex
@inproceedings{zhao2021point,
  title={Point Transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={ICCV},
  year={2021}
}
```

## Acknowledgements

ShapeNet is from Chang et al.; the part segmentation benchmark is the standard PyG `ShapeNet` dataset. This repo is for learning and experimentation, not an official reimplementation of the authors’ code.

"""
ShapeNet (PyTorch Geometric) en *classification de catégorie* (chaise, voiture, …).

Le dataset officiel est une tâche de segmentation de pièces ; chaque échantillon porte
`data.category` = indice parmi les catégories demandées (voir shapenet.py PyG).
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ShapeNet


def _category_to_label(data: Data) -> torch.Tensor:
    c = data.category
    if torch.is_tensor(c):
        return c.long().view(1)
    return torch.tensor([int(c)], dtype=torch.long, device=data.pos.device)


def subsample_points(data: Data, num_points: int, use_normals: bool) -> Data:
    """Sous-échantillonne aléatoirement pour batchs de taille fixe."""
    n = data.pos.size(0)
    assert n > 0, "nuage vide"
    if n <= num_points:
        idx = torch.randint(0, n, (num_points,), device=data.pos.device)
    else:
        idx = torch.randperm(n, device=data.pos.device)[:num_points]

    out = Data(
        pos=data.pos[idx],
        y_cls=_category_to_label(data),
    )
    if use_normals and data.x is not None:
        out.x = torch.cat([data.pos[idx], data.x[idx]], dim=-1)  # [N, 6] = xyz + normales
    else:
        out.x = data.pos[idx]  # [N, 3] = xyz seulement
    return out


class ShapeNetClassification(ShapeNet):
    """
    Même chargement que ShapeNet, avec transform optionnel pour N points + features.
    """

    def __init__(
        self,
        root: str,
        categories: list[str] | None = None,
        split: str = "train",
        num_points: int = 2048,
        use_normals: bool = True,
    ) -> None:
        self._num_points = num_points
        self._use_normals = use_normals

        def t(data: Data) -> Data:
            return subsample_points(data, num_points, use_normals)

        cats = categories if categories is not None else list(ShapeNet.category_ids.keys())
        super().__init__(
            root,
            categories=cats,
            include_normals=use_normals,
            split=split,
            transform=t,
        )

    @property
    def num_shape_classes(self) -> int:
        return len(self.categories)

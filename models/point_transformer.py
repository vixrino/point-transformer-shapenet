"""
Point Transformer (Zhao et al., ICCV 2021) — version pédagogique pour nuages de points.

Idées clés :
- Attention *vectorielle* : un coefficient d'attention par canal (pas un seul scalaire).
- Encodage de position relatif : MLP sur (p_i - p_j), voisins définis par k-NN dans l'espace 3D.
- Agrégation locale (k petit) pour limiter le coût O(N²).

Zéro dépendance externe au-delà de torch + torch_geometric.

Référence : https://arxiv.org/abs/2012.09164
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax


# ── k-NN en pur PyTorch (pas besoin de torch-cluster) ──────────────────────


def knn_graph_pure(pos: torch.Tensor, k: int, batch: torch.Tensor | None) -> torch.Tensor:
    """
    Construit un graphe k-NN dirigé (j -> i) en pur PyTorch.
    Retourne edge_index [2, N*k].
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_rows = []
    edge_cols = []
    for b in batch.unique():
        mask = batch == b
        idx = mask.nonzero(as_tuple=False).view(-1)
        pts = pos[idx]
        dist = torch.cdist(pts, pts)
        # k+1 car la distance à soi-même = 0, on garde tout (self-loop inclus)
        actual_k = min(k, pts.size(0))
        _, topk_idx = dist.topk(actual_k, dim=-1, largest=False)
        n = pts.size(0)
        center = idx.unsqueeze(1).expand(-1, actual_k)
        neighbor = idx[topk_idx]
        edge_cols.append(center.reshape(-1))   # i (centre)
        edge_rows.append(neighbor.reshape(-1)) # j (voisin)

    row = torch.cat(edge_rows, dim=0)
    col = torch.cat(edge_cols, dim=0)
    return torch.stack([row, col], dim=0)


# ── Point Transformer Layer ────────────────────────────────────────────────


class PointTransformerLayer(nn.Module):
    """
    Une couche d'auto-attention locale sur le graphe k-NN.

    edge_index : source = voisin j, target = centre i (flux j -> i).
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.out_channels = out_channels

        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)

        # rho(p_i - p_j) — encodage de position relatif (papier)
        self.mlp_rho = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        # gamma : produit le vecteur d'attention (vector attention)
        self.mlp_gamma = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.lin_out = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """
        x   : [N, in_channels]
        pos : [N, 3]
        batch : [N] ou None (un seul nuage)
        """
        edge_index = knn_graph_pure(pos, k=self.k, batch=batch)
        row, col = edge_index  # row = j (voisin), col = i (centre)

        q_i = self.lin_q(x)[col]
        k_j = self.lin_k(x)[row]
        v_j = self.lin_v(x)[row]

        delta_p = pos[col] - pos[row]
        rho = self.mlp_rho(delta_p)

        # Vector attention : gamma(q_i - k_j + rho)
        attn = self.mlp_gamma(q_i - k_j + rho)
        attn = softmax(attn, index=col, dim=0)

        m = attn * (v_j + rho)
        agg = scatter(m, col, dim=0, dim_size=x.size(0), reduce="sum")
        return self.lin_out(agg)


class PointTransformerBlock(nn.Module):
    def __init__(self, channels: int, k: int = 16) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.pt = PointTransformerLayer(channels, channels, k=k)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.pt(h, pos, batch)
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class PointTransformerClassifier(nn.Module):
    """
    Classification globale : empilement de blocs + max pooling sur les points + MLP tête.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        num_blocks: int = 4,
        k: int = 16,
        num_classes: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, k=k) for _ in range(num_blocks))
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """
        x : [N, C_in] — peut être les coordonnées seules (3) ou xyz + normales (6)
        pos : [N, 3]
        batch : [N] indices du graphe dans le batch PyG
        """
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, pos, batch)

        if batch is None:
            pooled = h.max(dim=0).values.unsqueeze(0)
        else:
            pooled = scatter(h, batch, dim=0, reduce="max")

        return self.head(pooled)


class PointTransformerSegmentor(nn.Module):
    """
    Segmentation par point : chaque point reçoit un label (aile, fuselage, réacteur…).

    Architecture :
    - Embed + N blocs de Point Transformer (features par point)
    - Concatène un vecteur global (contexte de la forme entière) à chaque point
    - MLP tête qui prédit un label par point
    """

    def __init__(
        self,
        in_channels: int = 6,
        dim: int = 64,
        num_blocks: int = 2,
        k: int = 8,
        num_seg_classes: int = 50,
        num_categories: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.cat_embed = nn.Embedding(num_categories, dim)
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, k=k) for _ in range(num_blocks))
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_seg_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        category: torch.Tensor,
    ) -> torch.Tensor:
        """
        x        : [N, C_in]
        pos      : [N, 3]
        batch    : [N]
        category : [B] indice de catégorie par forme
        Returns  : [N, num_seg_classes] logits par point
        """
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, pos, batch)

        # Vecteur global par forme (max pool) dupliqué sur chaque point
        global_feat = scatter(h, batch, dim=0, reduce="max")  # [B, dim]
        global_per_point = global_feat[batch]                  # [N, dim]

        # Embedding de la catégorie dupliqué sur chaque point
        cat_feat = self.cat_embed(category)                    # [B, dim]
        cat_per_point = cat_feat[batch]                        # [N, dim]

        # Concatène : features locales + globales + catégorie
        combined = torch.cat([h, global_per_point, cat_per_point], dim=-1)  # [N, dim*3]
        return self.head(combined)

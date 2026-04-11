#!/usr/bin/env python3
"""
Sanity check géométrique — junction faces sur un maillage 3D.

Aucun modèle, aucun GPU. On valide visuellement que les junction faces
portent assez d'information géométrique pour conditionner une génération
autorégressive dans la zone masquée.

Usage :
    python sanity_check.py                              # table procédurale par défaut
    python sanity_check.py --mesh chair.obj             # n'importe quel OBJ
    python sanity_check.py --mask_mode z_threshold      # masque par seuil Z
    python sanity_check.py --mask_mode face_index --face_start 20 --face_end 50
    python sanity_check.py --save results.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Création de maillages test (pas besoin de ShapeNetCore)
# ═══════════════════════════════════════════════════════════════════════════════


def make_table_mesh() -> trimesh.Trimesh:
    """Table = plateau + 4 pieds, tout en un seul mesh soudé."""
    top = trimesh.creation.box(extents=[1.0, 0.06, 0.6])
    top.apply_translation([0, 0.47, 0])

    legs = []
    for x, z in [(-0.4, -0.22), (0.4, -0.22), (-0.4, 0.22), (0.4, 0.22)]:
        leg = trimesh.creation.box(extents=[0.06, 0.88, 0.06])
        leg.apply_translation([x, 0.0, z])
        legs.append(leg)

    mesh = trimesh.util.concatenate([top] + legs)
    return mesh


def make_chair_mesh() -> trimesh.Trimesh:
    """Chaise simplifiée = assise + 4 pieds + dossier."""
    seat = trimesh.creation.box(extents=[0.5, 0.05, 0.5])
    seat.apply_translation([0, 0.4, 0])

    back = trimesh.creation.box(extents=[0.5, 0.5, 0.05])
    back.apply_translation([0, 0.67, -0.225])

    legs = []
    for x, z in [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]:
        leg = trimesh.creation.box(extents=[0.04, 0.38, 0.04])
        leg.apply_translation([x, 0.19, z])
        legs.append(leg)

    mesh = trimesh.util.concatenate([seat, back] + legs)
    return mesh


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sélection du masque (zone à supprimer)
# ═══════════════════════════════════════════════════════════════════════════════


def mask_by_z_threshold(mesh: trimesh.Trimesh, z_min: float, z_max: float) -> np.ndarray:
    """Masque les faces dont le centroïde est dans [z_min, z_max]."""
    centroids = mesh.triangles_center
    return (centroids[:, 2] >= z_min) & (centroids[:, 2] <= z_max)


def mask_by_y_threshold(mesh: trimesh.Trimesh, y_min: float, y_max: float) -> np.ndarray:
    """Masque les faces dont le centroïde Y est dans [y_min, y_max]."""
    centroids = mesh.triangles_center
    return (centroids[:, 1] >= y_min) & (centroids[:, 1] <= y_max)


def mask_by_face_range(mesh: trimesh.Trimesh, start: int, end: int) -> np.ndarray:
    """Masque les faces par indice [start, end)."""
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[start:end] = True
    return mask


def mask_by_bbox(mesh: trimesh.Trimesh, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Masque les faces dont le centroïde tombe dans la bounding box."""
    centroids = mesh.triangles_center
    inside = np.all(centroids >= bbox_min, axis=1) & np.all(centroids <= bbox_max, axis=1)
    return inside


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extraction des junction faces
# ═══════════════════════════════════════════════════════════════════════════════


def build_edge_to_faces(faces: np.ndarray) -> dict:
    """Construit un dictionnaire arête -> liste d'indices de faces."""
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_to_faces[edge].append(fi)
    return edge_to_faces


def find_junction_faces(
    faces: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifie les junction faces : faces NON masquées qui partagent au moins
    une arête avec une face masquée.

    Retourne :
        junction_mask : booléen [num_faces] — True pour les junction faces
        boundary_edges : array [E, 2] — arêtes de jonction (frontière du trou)
    """
    edge_to_faces = build_edge_to_faces(faces)
    masked_set = set(np.where(mask)[0])

    junction_mask = np.zeros(len(faces), dtype=bool)
    boundary_edges = []

    for edge, face_indices in edge_to_faces.items():
        has_masked = any(fi in masked_set for fi in face_indices)
        has_kept = any(fi not in masked_set for fi in face_indices)
        if has_masked and has_kept:
            boundary_edges.append(edge)
            for fi in face_indices:
                if fi not in masked_set:
                    junction_mask[fi] = True

    return junction_mask, np.array(boundary_edges) if boundary_edges else np.empty((0, 2), dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Métriques géométriques sur la bordure
# ═══════════════════════════════════════════════════════════════════════════════


def boundary_stats(
    mesh: trimesh.Trimesh,
    boundary_edges: np.ndarray,
    junction_mask: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Quelques métriques quantitatives sur la bordure du trou."""
    stats = {}
    stats["num_boundary_edges"] = len(boundary_edges)
    stats["num_junction_faces"] = int(junction_mask.sum())
    stats["num_masked_faces"] = int(mask.sum())
    stats["num_total_faces"] = len(mesh.faces)

    if len(boundary_edges) > 0:
        verts = mesh.vertices
        edge_lengths = np.linalg.norm(
            verts[boundary_edges[:, 0]] - verts[boundary_edges[:, 1]], axis=1
        )
        stats["boundary_length_total"] = float(edge_lengths.sum())
        stats["boundary_edge_mean_len"] = float(edge_lengths.mean())
        stats["boundary_edge_std_len"] = float(edge_lengths.std())

        junction_normals = mesh.face_normals[junction_mask]
        if len(junction_normals) > 1:
            dots = np.einsum("ij,ij->i", junction_normals[:-1], junction_normals[1:])
            angles = np.arccos(np.clip(dots, -1, 1))
            stats["junction_normal_angle_mean_deg"] = float(np.degrees(angles.mean()))
            stats["junction_normal_angle_std_deg"] = float(np.degrees(angles.std()))

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════


def plot_mesh(ax, mesh, face_colors, title, edge_alpha=0.15):
    """Affiche un maillage triangulaire 3D avec couleurs par face."""
    verts = mesh.vertices
    triangles = verts[mesh.faces]

    pc = Poly3DCollection(triangles, alpha=0.85, linewidths=0.3, edgecolors=(0, 0, 0, edge_alpha))
    pc.set_facecolor(face_colors)
    ax.add_collection3d(pc)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-60)


def visualize_sanity_check(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    junction_mask: np.ndarray,
    boundary_edges: np.ndarray,
    stats: dict,
    save_path: str | None = None,
) -> None:
    """3 vues côte à côte : original / trou / trou + junction faces."""
    fig = plt.figure(figsize=(20, 7))

    COLOR_NORMAL = [0.75, 0.82, 0.90, 1.0]
    COLOR_MASKED = [0.95, 0.30, 0.30, 1.0]
    COLOR_JUNCTION = [0.05, 0.85, 0.15, 1.0]
    COLOR_REMAINING = [0.82, 0.87, 0.93, 0.45]

    n = len(mesh.faces)

    # ─── Panel 1 : Maillage original ─────────────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d")
    colors1 = np.array([COLOR_NORMAL] * n)
    colors1[mask] = COLOR_MASKED
    plot_mesh(ax1, mesh, colors1, "Original mesh\n(red = masked region)")

    # ─── Panel 2 : Maillage avec trou ────────────────────────────────────
    ax2 = fig.add_subplot(132, projection="3d")
    kept = ~mask
    mesh_hole = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces[kept], process=False
    )
    colors2 = np.array([COLOR_REMAINING] * int(kept.sum()))
    plot_mesh(ax2, mesh_hole, colors2, "After removal\n(visible hole)")

    # ─── Panel 3 : Trou + junction faces en vert ─────────────────────────
    ax3 = fig.add_subplot(133, projection="3d")
    kept_junction = junction_mask[kept]

    # D'abord dessiner les faces non-junction (transparentes)
    non_junction_idx = np.where(~kept_junction)[0]
    if len(non_junction_idx) > 0:
        mesh_non_junc = trimesh.Trimesh(
            vertices=mesh_hole.vertices, faces=mesh_hole.faces[non_junction_idx], process=False
        )
        colors_nj = np.array([COLOR_REMAINING] * len(non_junction_idx))
        plot_mesh(ax3, mesh_non_junc, colors_nj, "Junction faces (green)\n= conditioning signal")

    # Puis dessiner les junction faces opaques par-dessus avec contour épais
    junction_idx = np.where(kept_junction)[0]
    if len(junction_idx) > 0:
        junc_tris = mesh_hole.vertices[mesh_hole.faces[junction_idx]]
        junc_pc = Poly3DCollection(junc_tris, alpha=1.0, linewidths=3.0, edgecolors="#00CC00", zorder=5)
        junc_pc.set_facecolor([0.05, 0.85, 0.15, 1.0])
        ax3.add_collection3d(junc_pc)
        # Scatter des sommets des junction faces pour les rendre visibles même de côté
        junc_verts = junc_tris.reshape(-1, 3)
        ax3.scatter(junc_verts[:, 0], junc_verts[:, 1], junc_verts[:, 2],
                    c="lime", s=60, zorder=15, edgecolors="darkgreen", linewidths=1.5)

    # Boundary edges en orange pour contraster avec le vert
    if len(boundary_edges) > 0:
        for e in boundary_edges:
            pts = mesh.vertices[e]
            ax3.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], color="#FF6600", linewidth=3.5, alpha=1.0, zorder=10)

    # Ajuster les limites de ax3 si on a dessiné les non-junction
    verts = mesh_hole.vertices
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.1
    ax3.set_xlim(center[0] - span, center[0] + span)
    ax3.set_ylim(center[1] - span, center[1] + span)
    ax3.set_zlim(center[2] - span, center[2] + span)

    # ─── Stats en texte ──────────────────────────────────────────────────
    stat_lines = [
        f"Masked faces: {stats['num_masked_faces']} / {stats['num_total_faces']}",
        f"Junction faces: {stats['num_junction_faces']}",
        f"Boundary edges: {stats['num_boundary_edges']}",
    ]
    if "boundary_length_total" in stats:
        stat_lines.append(f"Boundary length: {stats['boundary_length_total']:.3f}")
        stat_lines.append(f"Edge mean ± std: {stats['boundary_edge_mean_len']:.4f} ± {stats['boundary_edge_std_len']:.4f}")
    if "junction_normal_angle_mean_deg" in stats:
        stat_lines.append(f"Junction normal angle: {stats['junction_normal_angle_mean_deg']:.1f}° ± {stats['junction_normal_angle_std_deg']:.1f}°")

    fig.text(
        0.5, 0.02, "  |  ".join(stat_lines),
        ha="center", fontsize=9, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée : {save_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check — junction faces sur maillage 3D")
    p.add_argument("--mesh", type=str, default=None,
                   help="Chemin vers un fichier OBJ/STL/PLY (défaut : table procédurale)")
    p.add_argument("--shape", type=str, choices=["table", "chair"], default="chair",
                   help="Forme procédurale si --mesh n'est pas fourni")
    p.add_argument("--mask_mode", type=str, default="y_threshold",
                   choices=["z_threshold", "y_threshold", "face_index", "bbox"],
                   help="Méthode de sélection du masque")
    p.add_argument("--y_min", type=float, default=0.55, help="Seuil Y min (pour y_threshold)")
    p.add_argument("--y_max", type=float, default=1.0, help="Seuil Y max (pour y_threshold)")
    p.add_argument("--z_min", type=float, default=-0.1, help="Seuil Z min")
    p.add_argument("--z_max", type=float, default=0.1, help="Seuil Z max")
    p.add_argument("--face_start", type=int, default=0)
    p.add_argument("--face_end", type=int, default=20)
    p.add_argument("--save", type=str, default=None, help="Chemin pour sauvegarder l'image")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Charger ou créer le maillage
    if args.mesh:
        mesh = trimesh.load(args.mesh, force="mesh")
        print(f"Maillage chargé : {args.mesh}  ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
    else:
        mesh = make_chair_mesh() if args.shape == "chair" else make_table_mesh()
        print(f"Maillage procédural '{args.shape}' : {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # 2. Sélectionner le masque
    if args.mask_mode == "z_threshold":
        mask = mask_by_z_threshold(mesh, args.z_min, args.z_max)
    elif args.mask_mode == "y_threshold":
        mask = mask_by_y_threshold(mesh, args.y_min, args.y_max)
    elif args.mask_mode == "face_index":
        mask = mask_by_face_range(mesh, args.face_start, args.face_end)
    elif args.mask_mode == "bbox":
        mask = mask_by_bbox(
            mesh,
            np.array([args.z_min, args.y_min, args.z_min]),
            np.array([args.z_max, args.y_max, args.z_max]),
        )
    else:
        raise ValueError(f"Mode inconnu : {args.mask_mode}")

    print(f"Faces masquées : {mask.sum()} / {len(mesh.faces)}")
    if mask.sum() == 0:
        print("Aucune face masquée ! Ajuste les seuils.")
        return
    if mask.sum() == len(mesh.faces):
        print("Toutes les faces masquées ! Ajuste les seuils.")
        return

    # 3. Extraire les junction faces
    junction_mask, boundary_edges = find_junction_faces(mesh.faces, mask)
    print(f"Junction faces : {junction_mask.sum()}")
    print(f"Boundary edges : {len(boundary_edges)}")

    # 4. Métriques
    stats = boundary_stats(mesh, boundary_edges, junction_mask, mask)
    print("\n--- Métriques bordure ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 5. Visualiser
    visualize_sanity_check(mesh, mask, junction_mask, boundary_edges, stats, args.save)


if __name__ == "__main__":
    main()

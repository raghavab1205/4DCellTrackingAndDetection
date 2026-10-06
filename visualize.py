"""
visualize.py — Visualization utilities for cell tracking.

Creates 2D max-intensity projections of 3D volumes with overlaid
ground-truth or predicted graphs using matplotlib.

Public API
----------
plot_frame_with_detections(frame, centroids, ...)
plot_frame_with_gt(frame, gt_nodes_t, ...)
plot_tracking(frames, graph, t_range, ...)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _normalize_mip(frame_3d: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.5) -> np.ndarray:
    """Create a normalized max-intensity projection from a 3D volume."""
    mip = frame_3d.max(axis=0).astype(np.float32)
    vmin = np.percentile(mip, percentile_low)
    vmax = np.percentile(mip, percentile_high)
    if vmax <= vmin:
        vmax = vmin + 1.0
    mip = np.clip(mip, vmin, vmax)
    mip = (mip - vmin) / (vmax - vmin)
    return mip


def plot_frame_with_detections(
    frame_3d: np.ndarray,
    centroids: np.ndarray = None,
    gt_nodes: list = None,
    title: str = "",
    save_path: str = None,
    figsize: tuple = (8, 8),
):
    """Plot a max-intensity projection with detected and/or GT centroids.

    Parameters
    ----------
    frame_3d : np.ndarray, shape (Z, Y, X)
    centroids : np.ndarray, shape (N, 3), optional
        Predicted centroids in voxel coords (z, y, x).
    gt_nodes : list of dict, optional
        GT nodes with keys 'z', 'y', 'x'.
    title : str
    save_path : str, optional
        If provided, save figure to this path.
    """
    mip = _normalize_mip(frame_3d)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(mip, cmap="gray", origin="upper")

    handles = []

    if gt_nodes is not None and len(gt_nodes) > 0:
        gt_y = [n["y"] for n in gt_nodes]
        gt_x = [n["x"] for n in gt_nodes]
        ax.scatter(gt_x, gt_y, c="lime", s=60, marker="o", linewidths=1.5,
                   edgecolors="white", alpha=0.8, label="GT")
        handles.append(mpatches.Patch(color="lime", label=f"GT ({len(gt_nodes)})"))

    if centroids is not None and len(centroids) > 0:
        ax.scatter(centroids[:, 2], centroids[:, 1], c="red", s=40, marker="x",
                   linewidths=1.5, alpha=0.8, label="Predicted")
        handles.append(mpatches.Patch(color="red", label=f"Pred ({len(centroids)})"))

    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_tracking_pair(
    frame0_3d: np.ndarray,
    frame1_3d: np.ndarray,
    centroids_t0: np.ndarray,
    centroids_t1: np.ndarray,
    matches: list,
    title: str = "",
    save_path: str = None,
    figsize: tuple = (16, 8),
):
    """Plot two consecutive frames side-by-side with tracking links.

    Parameters
    ----------
    frame0_3d, frame1_3d : np.ndarray, shape (Z, Y, X)
    centroids_t0, centroids_t1 : np.ndarray, shape (N, 3) and (M, 3)
    matches : list of (int, int)
        Index pairs from link_frames().
    """
    mip0 = _normalize_mip(frame0_3d)
    mip1 = _normalize_mip(frame1_3d)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    for ax, mip, centroids, t_label in [(ax0, mip0, centroids_t0, "t"),
                                         (ax1, mip1, centroids_t1, "t+1")]:
        ax.imshow(mip, cmap="gray", origin="upper")
        if centroids is not None and len(centroids) > 0:
            ax.scatter(centroids[:, 2], centroids[:, 1], c="cyan", s=30,
                       marker="o", linewidths=1, edgecolors="white", alpha=0.8)
        ax.set_title(f"{t_label} ({len(centroids)} detections)", fontsize=11)

    # Draw lines connecting matched pairs (overlay on right panel)
    for i, j in matches:
        y0, x0 = centroids_t0[i, 1], centroids_t0[i, 2]
        y1, x1 = centroids_t1[j, 1], centroids_t1[j, 2]
        # Draw arrow on right panel from t position to t+1 position
        ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5, alpha=0.6))

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    from data_io import load_volume_frame, load_geff, VOXEL_SCALE, sample_zarr_path, sample_geff_path
    from detector import detect_cells_threshold

    DATA_DIR = "f:/4D cell tracking/biohub-cell-tracking-during-development"
    sample = "44b6_0113de3b"

    zarr_p = sample_zarr_path(DATA_DIR, "train", sample)
    geff_p = sample_geff_path(DATA_DIR, "train", sample)

    frame = load_volume_frame(zarr_p, 0)
    geff = load_geff(geff_p)

    # GT nodes at t=0
    gt_t0 = geff["nodes_df"][geff["nodes_df"]["t"] == 0]
    gt_nodes = [{"z": r["z"], "y": r["y"], "x": r["x"]} for _, r in gt_t0.iterrows()]

    # Detect
    centroids = detect_cells_threshold(frame, VOXEL_SCALE)

    plot_frame_with_detections(
        frame, centroids, gt_nodes,
        title=f"{sample} t=0 — Detections vs GT",
        save_path="f:/4D cell tracking/viz_frame0.png",
    )
    print("Visualization saved.")

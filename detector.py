"""
detector.py — Baseline 3D cell detection using Laplacian of Gaussian (LoG).

Public API
----------
detect_cells(volume_3d, voxel_scale, ...)  -> np.ndarray of shape (N, 3) in voxel coords (z, y, x)

The LoG detector is classical, training-free, and fast. It works well for
roughly-spherical bright blobs (fluorescent nuclei) on a dark background.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import blob_log


def detect_cells(
    volume_3d: np.ndarray,
    voxel_scale: np.ndarray = None,
    min_sigma_um: float = 2.0,
    max_sigma_um: float = 6.0,
    num_sigma: int = 5,
    threshold: float = 0.02,
    overlap: float = 0.5,
    intensity_percentile: float = 5.0,
) -> np.ndarray:
    """Detect cell centroids in a single 3-D volume using LoG blob detection.

    Parameters
    ----------
    volume_3d : np.ndarray, shape (Z, Y, X)
        Single-timepoint fluorescence volume, dtype uint16 or float.
    voxel_scale : np.ndarray, optional
        Physical voxel size in µm as (z, y, x). Default: (1.625, 0.40625, 0.40625).
    min_sigma_um : float
        Minimum expected blob radius in µm.
    max_sigma_um : float
        Maximum expected blob radius in µm.
    num_sigma : int
        Number of sigma values to test between min and max.
    threshold : float
        LoG response threshold (relative to the normalized image max).
        Lower values = more detections.
    overlap : float
        Allowed overlap between blobs (0-1). Higher = more merging.
    intensity_percentile : float
        Percentile threshold for masking out pure background before detection.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Detected centroid coordinates in voxel units (z, y, x), dtype float64.
    """
    if voxel_scale is None:
        voxel_scale = np.array([1.625, 0.40625, 0.40625])

    # Normalize to [0, 1] float
    vol = volume_3d.astype(np.float32)
    # Clip to remove outliers (use 99.9th percentile)
    vmin = np.percentile(vol, intensity_percentile)
    vmax = np.percentile(vol, 99.9)
    if vmax <= vmin:
        vmax = vmin + 1.0
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin)

    # Convert sigma from physical µm to voxel units for each axis
    # LoG in skimage expects isotropic sigma, so we rescale the volume
    # to isotropic resolution first
    min_voxel = min(voxel_scale)
    scale_factors = voxel_scale / min_voxel  # e.g., [4.0, 1.0, 1.0]

    # Resample to isotropic (zoom to make all axes same physical resolution)
    zoom_factors = 1.0 / scale_factors  # zoom DOWN the oversampled axes
    # Actually, we want to zoom UP the under-sampled axes to isotropic
    # z has larger voxel size (1.625 µm) vs y,x (0.40625 µm)
    # So z is undersampled -> zoom UP z by factor 4
    zoom_factors = scale_factors  # zoom z UP, keep y,x at 1
    vol_iso = ndimage.zoom(vol, zoom_factors, order=1)

    # Now sigma is in isotropic voxels (each = min_voxel µm)
    min_sigma_vox = min_sigma_um / min_voxel
    max_sigma_vox = max_sigma_um / min_voxel

    # Run LoG blob detection on isotropic volume
    blobs = blob_log(
        vol_iso,
        min_sigma=min_sigma_vox,
        max_sigma=max_sigma_vox,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )

    if len(blobs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # blobs[:, :3] are (z_iso, y_iso, x_iso) in the isotropic volume
    centroids_iso = blobs[:, :3]

    # Convert back to original voxel coords
    centroids_voxel = centroids_iso / zoom_factors[np.newaxis, :]

    return centroids_voxel.astype(np.float64)


def detect_cells_threshold(
    volume_3d: np.ndarray,
    voxel_scale: np.ndarray = None,
    sigma_um: float = 2.0,
    threshold_percentile: float = 85.0,
    min_size_voxels: int = 10,
) -> np.ndarray:
    """Simple fallback detector: Gaussian smooth + threshold + connected components.

    Faster than LoG but less accurate. Useful as a sanity check.

    Parameters
    ----------
    volume_3d : np.ndarray, shape (Z, Y, X)
    voxel_scale : np.ndarray, optional
    sigma_um : float
        Gaussian smoothing sigma in µm.
    threshold_percentile : float
        Percentile of smoothed volume to use as threshold.
    min_size_voxels : int
        Minimum connected component size to keep.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Centroids in voxel units (z, y, x).
    """
    if voxel_scale is None:
        voxel_scale = np.array([1.625, 0.40625, 0.40625])

    # Gaussian smoothing with anisotropic sigma
    sigma_voxels = sigma_um / voxel_scale
    smoothed = ndimage.gaussian_filter(
        volume_3d.astype(np.float32), sigma=sigma_voxels
    )

    # Threshold
    thresh = np.percentile(smoothed, threshold_percentile)
    binary = smoothed > thresh

    # Connected components
    labeled, n_labels = ndimage.label(binary)

    centroids = []
    for i in range(1, n_labels + 1):
        component = labeled == i
        size = component.sum()
        if size < min_size_voxels:
            continue
        # Center of mass
        com = ndimage.center_of_mass(smoothed, labeled, i)
        centroids.append(com)

    if len(centroids) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.array(centroids, dtype=np.float64)


if __name__ == "__main__":
    from data_io import load_volume_frame, VOXEL_SCALE

    DATA_DIR = "f:/4D cell tracking/biohub-cell-tracking-during-development"
    zarr_path = f"{DATA_DIR}/train/44b6_0113de3b.zarr"

    frame = load_volume_frame(zarr_path, 0)
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Intensity range: [{frame.min()}, {frame.max()}]")

    # Test threshold detector (faster)
    print("\n--- Threshold Detector ---")
    centroids_thresh = detect_cells_threshold(frame, VOXEL_SCALE)
    print(f"Detected {len(centroids_thresh)} cells (threshold)")
    if len(centroids_thresh) > 0:
        print(f"First 5 centroids (z,y,x voxel):\n{centroids_thresh[:5]}")

    # Test LoG detector
    print("\n--- LoG Detector ---")
    centroids_log = detect_cells(frame, VOXEL_SCALE, threshold=0.01)
    print(f"Detected {len(centroids_log)} cells (LoG)")
    if len(centroids_log) > 0:
        print(f"First 5 centroids (z,y,x voxel):\n{centroids_log[:5]}")

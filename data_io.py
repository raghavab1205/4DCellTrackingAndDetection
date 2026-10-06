"""
data_io.py — Read Zarr v3 image volumes and .geff ground-truth graphs.

Zarr v3 is read via raw chunk decompression (blosc/zstd) because zarr-python
2.x on Python 3.10 doesn't support the v3 spec natively.

Public API
----------
load_volume(zarr_path)          -> np.ndarray of shape (T, Z, Y, X), uint16
load_volume_frame(zarr_path, t) -> np.ndarray of shape (Z, Y, X), uint16
load_geff(geff_path)            -> dict with keys 'node_ids', 'nodes_df', 'edges'
get_zarr_metadata(zarr_path)    -> dict parsed from zarr.json
get_geff_metadata(geff_path)    -> dict parsed from zarr.json
list_samples(data_dir, split)   -> list of sample name strings
VOXEL_SCALE                     -> (z, y, x) physical scale in µm
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc, Zstd

# Physical voxel scale in micrometers (from competition spec)
VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x


# ---------------------------------------------------------------------------
# Zarr v3 low-level reader
# ---------------------------------------------------------------------------

def get_zarr_metadata(zarr_path: str) -> dict:
    """Read and parse the top-level zarr.json metadata."""
    with open(os.path.join(zarr_path, "zarr.json"), "r") as f:
        return json.load(f)


def _get_array_meta(zarr_path: str) -> dict:
    """Read the array-level zarr.json at <zarr>/0/zarr.json."""
    with open(os.path.join(zarr_path, "0", "zarr.json"), "r") as f:
        return json.load(f)


def load_volume_frame(zarr_path: str, t: int) -> np.ndarray:
    """Load a single timepoint from a Zarr v3 image volume.

    Parameters
    ----------
    zarr_path : str
        Path to the .zarr directory.
    t : int
        Timepoint index.

    Returns
    -------
    np.ndarray
        3-D array of shape (Z, Y, X), dtype uint16.
    """
    meta = _get_array_meta(zarr_path)
    shape = tuple(meta["shape"])        # (T, Z, Y, X)
    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")

    if t < 0 or t >= shape[0]:
        raise IndexError(f"Timepoint {t} out of range [0, {shape[0]})")

    # Each chunk is one timepoint: chunk key = c/<t>/0/0/0
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")

    # Determine decompressor from codec chain
    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)

    with open(chunk_path, "rb") as f:
        raw = f.read()

    buf = codec.decode(raw)
    arr = np.frombuffer(buf, dtype=dtype).reshape(chunk_shape)
    return arr[0]  # remove leading T=1 dimension -> (Z, Y, X)


def load_volume(zarr_path: str) -> np.ndarray:
    """Load the full 4-D volume (T, Z, Y, X) from a Zarr v3 store.

    This reads all timepoints sequentially. For large volumes, prefer
    load_volume_frame() to process one frame at a time.
    """
    meta = _get_array_meta(zarr_path)
    T = meta["shape"][0]
    frames = [load_volume_frame(zarr_path, t) for t in range(T)]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# .geff (graph exchange format) reader
# ---------------------------------------------------------------------------

def get_geff_metadata(geff_path: str) -> dict:
    """Read and parse the geff-level zarr.json metadata."""
    with open(os.path.join(geff_path, "zarr.json"), "r") as f:
        return json.load(f)


def _read_geff_array(path: str, dtype: str) -> np.ndarray:
    """Read a single zstd-compressed geff array chunk."""
    zstd = Zstd(level=0)
    chunk_file = os.path.join(path, "c", "0")
    if not os.path.isfile(chunk_file):
        # 2-D arrays (edges) use c/0/0
        chunk_file = os.path.join(path, "c", "0", "0")
    with open(chunk_file, "rb") as f:
        raw = f.read()
    return np.frombuffer(zstd.decode(raw), dtype=dtype)


def load_geff(geff_path: str) -> dict:
    """Load a .geff ground-truth graph.

    Returns
    -------
    dict with keys:
        'node_ids'  : np.ndarray of uint64 node IDs
        'nodes_df'  : pd.DataFrame with columns [node_id, t, z, y, x]
        'edges'     : np.ndarray of shape (N, 2), uint64 (source_id, target_id)
        'metadata'  : dict from zarr.json
    """
    meta = get_geff_metadata(geff_path)

    # Node IDs
    node_ids = _read_geff_array(
        os.path.join(geff_path, "nodes", "ids"), dtype="<u8"
    )

    # Node properties
    props = {}
    for prop in ["t", "z", "y", "x"]:
        props[prop] = _read_geff_array(
            os.path.join(geff_path, "nodes", "props", prop, "values"),
            dtype="<i8",
        )

    nodes_df = pd.DataFrame(
        {
            "node_id": node_ids,
            "t": props["t"],
            "z": props["z"],
            "y": props["y"],
            "x": props["x"],
        }
    )

    # Edges (shape (N, 2))
    edges_path = os.path.join(geff_path, "edges", "ids")
    edges_meta_file = os.path.join(edges_path, "zarr.json")
    with open(edges_meta_file, "r") as f:
        edges_meta = json.load(f)
    n_edges = edges_meta["shape"][0]

    edges_raw = _read_geff_array(edges_path, dtype="<u8")
    edges = edges_raw.reshape(n_edges, 2)

    return {
        "node_ids": node_ids,
        "nodes_df": nodes_df,
        "edges": edges,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def list_samples(data_dir: str, split: str = "train") -> list:
    """List all sample names (without extension) in a split directory.

    Parameters
    ----------
    data_dir : str
        Root dataset directory (contains train/ and test/ subdirs).
    split : str
        "train" or "test".

    Returns
    -------
    list of str
        Sorted unique sample names, e.g. ['44b6_0113de3b', ...].
    """
    split_dir = os.path.join(data_dir, split)
    names = set()
    for entry in os.listdir(split_dir):
        if entry.endswith(".zarr"):
            names.add(entry.replace(".zarr", ""))
    return sorted(names)


def sample_zarr_path(data_dir: str, split: str, name: str) -> str:
    """Return the .zarr directory path for a sample."""
    return os.path.join(data_dir, split, f"{name}.zarr")


def sample_geff_path(data_dir: str, split: str, name: str) -> str:
    """Return the .geff directory path for a sample."""
    return os.path.join(data_dir, split, f"{name}.geff")


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def voxel_to_physical(coords_zyx: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates to physical (µm) coordinates.

    Parameters
    ----------
    coords_zyx : np.ndarray, shape (..., 3)
        Coordinates in (z, y, x) voxel units.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Coordinates in physical µm units.
    """
    return coords_zyx * VOXEL_SCALE


def physical_distance(a_zyx: np.ndarray, b_zyx: np.ndarray) -> float:
    """Compute Euclidean distance in physical (µm) space between two voxel coords."""
    a_phys = voxel_to_physical(np.asarray(a_zyx, dtype=float))
    b_phys = voxel_to_physical(np.asarray(b_zyx, dtype=float))
    return float(np.linalg.norm(a_phys - b_phys))


if __name__ == "__main__":
    # Quick smoke test
    DATA_DIR = "f:/4D cell tracking/biohub-cell-tracking-during-development"
    samples = list_samples(DATA_DIR, "train")
    print(f"Found {len(samples)} train samples. First 5: {samples[:5]}")

    sample = samples[0]
    zarr_p = sample_zarr_path(DATA_DIR, "train", sample)
    geff_p = sample_geff_path(DATA_DIR, "train", sample)

    print(f"\n--- Zarr metadata for {sample} ---")
    zmeta = get_zarr_metadata(zarr_p)
    ameta = _get_array_meta(zarr_p)
    print(f"  Shape: {ameta['shape']}, dtype: {ameta['data_type']}")

    frame = load_volume_frame(zarr_p, 0)
    print(f"  Frame 0: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

    print(f"\n--- GEFF metadata for {sample} ---")
    geff = load_geff(geff_p)
    print(f"  Nodes: {len(geff['node_ids'])}, Edges: {geff['edges'].shape[0]}")
    print(f"  t range: [{geff['nodes_df']['t'].min()}, {geff['nodes_df']['t'].max()}]")
    est = geff["metadata"]["attributes"]["geff"]["extra"]["estimated_number_of_nodes"]
    print(f"  Estimated total nodes: {est}")
    print(f"\n  First 5 nodes:")
    print(geff["nodes_df"].head().to_string(index=False))
    print(f"\n  First 5 edges:")
    print(geff["edges"][:5])

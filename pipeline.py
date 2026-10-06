"""
pipeline.py — Main end-to-end tracking pipeline and CSV export.

Runs detector + tracker over all frames of each sample, constructs the
tracking graph, and exports to the Kaggle submission CSV format.

Usage
-----
    python pipeline.py                     # Run on test set, produce submission.csv
    python pipeline.py --eval              # Run on train set with evaluation
    python pipeline.py --eval --max-samples 5  # Quick test on 5 train samples
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from data_io import (
    VOXEL_SCALE,
    list_samples,
    load_geff,
    load_volume_frame,
    sample_geff_path,
    sample_zarr_path,
    _get_array_meta,
)
from detector import detect_cells_threshold
from tracker import TrackGraph, build_tracks, detect_divisions


def run_pipeline_on_sample(
    zarr_path: str,
    voxel_scale: np.ndarray = None,
    max_distance_um: float = 7.0,
    detection_kwargs: dict = None,
    verbose: bool = True,
) -> TrackGraph:
    """Run the full detect → link → division pipeline on one sample.

    Parameters
    ----------
    zarr_path : str
        Path to the .zarr directory.
    voxel_scale : np.ndarray
    max_distance_um : float
    detection_kwargs : dict, optional
        Keyword args passed to detect_cells_threshold().
    verbose : bool

    Returns
    -------
    TrackGraph
    """
    if voxel_scale is None:
        voxel_scale = VOXEL_SCALE
    if detection_kwargs is None:
        detection_kwargs = {}

    # Get volume shape
    meta = _get_array_meta(zarr_path)
    T = meta["shape"][0]

    if verbose:
        print(f"  Processing {T} timepoints...")

    # Detect cells in each frame
    all_centroids = {}
    for t in range(T):
        frame = load_volume_frame(zarr_path, t)
        centroids = detect_cells_threshold(frame, voxel_scale, **detection_kwargs)
        all_centroids[t] = centroids
        if verbose and (t % 20 == 0 or t == T - 1):
            print(f"    t={t:3d}: {len(centroids):4d} detections")

    # Build tracking graph
    graph = build_tracks(all_centroids, voxel_scale, max_distance_um)

    # Detect divisions
    graph = detect_divisions(graph, all_centroids, voxel_scale, max_distance_um)

    if verbose:
        n_divs = sum(1 for n in graph.nodes
                     if len(graph.get_outgoing_edges(n["id"])) >= 2)
        print(f"  Result: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
              f"{n_divs} divisions")

    return graph


def graph_to_submission_rows(
    graph: TrackGraph,
    dataset_name: str,
) -> list:
    """Convert a TrackGraph to submission CSV rows.

    Returns list of dicts with keys matching the submission format.
    """
    rows = []

    # Node rows
    for node in graph.nodes:
        rows.append({
            "dataset": dataset_name,
            "row_type": "node",
            "node_id": node["id"],
            "t": node["t"],
            "z": node["z"],
            "y": node["y"],
            "x": node["x"],
            "source_id": -1,
            "target_id": -1,
        })

    # Edge rows
    for src, tgt in graph.edges:
        rows.append({
            "dataset": dataset_name,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": src,
            "target_id": tgt,
        })

    return rows


def write_submission(all_rows: list, output_path: str):
    """Write all rows to the submission CSV file.

    Parameters
    ----------
    all_rows : list of dict
    output_path : str
    """
    df = pd.DataFrame(all_rows)
    df.insert(0, "id", range(len(df)))
    df = df[["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
             "source_id", "target_id"]]
    df.to_csv(output_path, index=False)
    print(f"\nSubmission written: {output_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="3D+Time Cell Tracking Pipeline")
    parser.add_argument("--data-dir", type=str,
                        default="f:/4D cell tracking/biohub-cell-tracking-during-development",
                        help="Root dataset directory")
    parser.add_argument("--output", type=str, default="f:/4D cell tracking/submission.csv",
                        help="Output CSV path")
    parser.add_argument("--eval", action="store_true",
                        help="Run on train set and evaluate instead of producing test submission")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to process (for quick testing)")
    parser.add_argument("--max-distance", type=float, default=7.0,
                        help="Max linking distance in µm")
    args = parser.parse_args()

    split = "train" if args.eval else "test"
    samples = list_samples(args.data_dir, split)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Pipeline: processing {len(samples)} {split} samples")
    print(f"  Max linking distance: {args.max_distance} µm")
    print()

    all_rows = []
    eval_results = []

    for idx, sample in enumerate(samples):
        t0 = time.time()
        print(f"[{idx+1}/{len(samples)}] {sample}")

        zarr_p = sample_zarr_path(args.data_dir, split, sample)
        graph = run_pipeline_on_sample(
            zarr_p,
            max_distance_um=args.max_distance,
        )

        rows = graph_to_submission_rows(graph, sample)
        all_rows.extend(rows)

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Evaluate if on train set
        if args.eval:
            from metric import evaluate_sample
            geff_p = sample_geff_path(args.data_dir, split, sample)
            gt = load_geff(geff_p)
            result = evaluate_sample(graph, gt)
            eval_results.append(result)
            print(f"  Edge Jaccard: {result['edge_jaccard']:.4f} "
                  f"(TP={result['edge_tp']}, FP={result['edge_fp']}, FN={result['edge_fn']})")
            print(f"  Div  Jaccard: {result['div_jaccard']:.4f} "
                  f"(TP={result['div_tp']}, FP={result['div_fp']}, FN={result['div_fn']})")

        print()

    # Write submission
    write_submission(all_rows, args.output)

    # Aggregate evaluation
    if args.eval and len(eval_results) > 0:
        from metric import evaluate_all
        agg = evaluate_all(eval_results)
        print(f"\n{'='*50}")
        print(f"AGGREGATE RESULTS ({agg['n_samples']} samples)")
        print(f"  Edge Jaccard (weighted): {agg['edge_jaccard_weighted']:.4f}")
        print(f"  Div  Jaccard (micro):    {agg['div_jaccard_micro']:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()

"""
metric.py — Local implementation of the competition evaluation metrics.

Implements:
  - Edge Jaccard: per-sample, then weight-averaged.
  - Division Jaccard: micro-averaged across samples.

The matching procedure mirrors the competition spec:
  1. Per timepoint, match predicted nodes to GT nodes via optimal bipartite
     assignment on scaled Euclidean distance (max 7.0 µm).
  2. A predicted edge (u→v) is TP if both u and v are matched to GT nodes
     that share a GT edge.
  3. Division: a GT division is TP if the predicted graph has the parent
     matched and both daughter lineages covered.

Public API
----------
evaluate_sample(pred_graph, gt_geff, ...) -> dict with edge/division scores
evaluate_all(pred_graphs, data_dir, ...) -> dict with aggregated scores
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from data_io import VOXEL_SCALE, voxel_to_physical, load_geff
from tracker import TrackGraph


def _match_nodes_at_time(
    pred_nodes: list,
    gt_nodes: list,
    voxel_scale: np.ndarray,
    max_distance_um: float = 7.0,
) -> dict:
    """Match predicted nodes to GT nodes at a single timepoint.

    Returns
    -------
    dict : mapping pred_node_id -> gt_node_id for matched pairs.
    """
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return {}

    pred_coords = np.array([[n["z"], n["y"], n["x"]] for n in pred_nodes], dtype=float)
    gt_coords = np.array([[n["z"], n["y"], n["x"]] for n in gt_nodes], dtype=float)

    # Physical distance
    pred_phys = voxel_to_physical(pred_coords)
    gt_phys = voxel_to_physical(gt_coords)

    cost = cdist(pred_phys, gt_phys, metric="euclidean")

    SENTINEL = 1e6
    cost_masked = np.where(cost <= max_distance_um, cost, SENTINEL)

    row_ind, col_ind = linear_sum_assignment(cost_masked)

    matching = {}
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] <= max_distance_um:
            matching[pred_nodes[i]["id"]] = gt_nodes[j]["node_id"]

    return matching


def evaluate_sample(
    pred_graph: TrackGraph,
    gt_geff: dict,
    voxel_scale: np.ndarray = None,
    max_distance_um: float = 7.0,
) -> dict:
    """Evaluate a predicted tracking graph against ground truth for one sample.

    Parameters
    ----------
    pred_graph : TrackGraph
    gt_geff : dict
        Output of data_io.load_geff().
    voxel_scale : np.ndarray, optional
    max_distance_um : float

    Returns
    -------
    dict with keys:
        'edge_tp', 'edge_fp', 'edge_fn', 'edge_jaccard',
        'div_tp', 'div_fp', 'div_fn', 'div_jaccard',
        'n_pred_nodes', 'n_gt_nodes', 'estimated_nodes'
    """
    if voxel_scale is None:
        voxel_scale = VOXEL_SCALE

    gt_df = gt_geff["nodes_df"]
    gt_edges_arr = gt_geff["edges"]
    estimated_nodes = gt_geff["metadata"]["attributes"]["geff"]["extra"]["estimated_number_of_nodes"]

    # Build GT node lookup
    gt_nodes_by_time = {}
    for _, row in gt_df.iterrows():
        t = int(row["t"])
        if t not in gt_nodes_by_time:
            gt_nodes_by_time[t] = []
        gt_nodes_by_time[t].append({
            "node_id": int(row["node_id"]),
            "z": int(row["z"]),
            "y": int(row["y"]),
            "x": int(row["x"]),
        })

    # Build pred node lookup
    pred_nodes_by_time = {}
    for node in pred_graph.nodes:
        t = node["t"]
        if t not in pred_nodes_by_time:
            pred_nodes_by_time[t] = []
        pred_nodes_by_time[t].append(node)

    # Match per timepoint
    all_timepoints = sorted(set(list(gt_nodes_by_time.keys()) + list(pred_nodes_by_time.keys())))
    pred_to_gt = {}  # pred_node_id -> gt_node_id

    for t in all_timepoints:
        pred_t = pred_nodes_by_time.get(t, [])
        gt_t = gt_nodes_by_time.get(t, [])
        matches = _match_nodes_at_time(pred_t, gt_t, voxel_scale, max_distance_um)
        pred_to_gt.update(matches)

    # --- Edge Jaccard ---
    # GT edge set
    gt_edge_set = set()
    for i in range(len(gt_edges_arr)):
        gt_edge_set.add((int(gt_edges_arr[i, 0]), int(gt_edges_arr[i, 1])))

    # Map predicted edges to GT space
    gt_to_pred = {v: k for k, v in pred_to_gt.items()}  # gt_id -> pred_id
    pred_edge_set_mapped = set()
    for src, tgt in pred_graph.edges:
        gt_src = pred_to_gt.get(src)
        gt_tgt = pred_to_gt.get(tgt)
        if gt_src is not None and gt_tgt is not None:
            pred_edge_set_mapped.add((gt_src, gt_tgt))

    edge_tp = len(pred_edge_set_mapped & gt_edge_set)
    edge_fp = len(pred_edge_set_mapped - gt_edge_set)
    edge_fn = len(gt_edge_set - pred_edge_set_mapped)

    # Node count penalty: if we predict more nodes than estimated, add penalty
    n_pred = len(pred_graph.nodes)
    n_gt_annotated = len(gt_df)
    node_overcount = max(0, n_pred - estimated_nodes)

    edge_denom = edge_tp + edge_fp + edge_fn + node_overcount
    edge_jaccard = edge_tp / edge_denom if edge_denom > 0 else 0.0

    # --- Division Jaccard ---
    # GT divisions: nodes with >=2 outgoing edges in GT
    gt_outgoing = {}
    for src, tgt in gt_edge_set:
        if src not in gt_outgoing:
            gt_outgoing[src] = []
        gt_outgoing[src].append(tgt)

    gt_divisions = {src: tgts for src, tgts in gt_outgoing.items() if len(tgts) >= 2}

    # Pred divisions: nodes with >=2 outgoing edges in pred
    pred_outgoing = {}
    for src, tgt in pred_graph.edges:
        if src not in pred_outgoing:
            pred_outgoing[src] = []
        pred_outgoing[src].append(tgt)

    pred_divisions = {src: tgts for src, tgts in pred_outgoing.items() if len(tgts) >= 2}

    # Match divisions
    div_tp = 0
    matched_gt_divs = set()
    for pred_parent, pred_daughters in pred_divisions.items():
        gt_parent = pred_to_gt.get(pred_parent)
        if gt_parent is None or gt_parent not in gt_divisions:
            continue

        gt_daughters = set(gt_divisions[gt_parent])
        pred_daughters_mapped = set()
        for d in pred_daughters:
            gt_d = pred_to_gt.get(d)
            if gt_d is not None:
                pred_daughters_mapped.add(gt_d)

        # Division is TP if at least 2 GT daughters are covered
        if len(pred_daughters_mapped & gt_daughters) >= 2:
            div_tp += 1
            matched_gt_divs.add(gt_parent)

    div_fn = len(gt_divisions) - len(matched_gt_divs)
    div_fp = sum(1 for p_parent in pred_divisions
                 if pred_to_gt.get(p_parent) not in matched_gt_divs)

    div_denom = div_tp + div_fp + div_fn
    div_jaccard = div_tp / div_denom if div_denom > 0 else 0.0

    return {
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_jaccard": edge_jaccard,
        "node_overcount_penalty": node_overcount,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
        "div_jaccard": div_jaccard,
        "n_pred_nodes": n_pred,
        "n_gt_annotated_nodes": n_gt_annotated,
        "estimated_total_nodes": estimated_nodes,
    }


def evaluate_all(
    sample_results: list,
) -> dict:
    """Aggregate per-sample evaluation results.

    Parameters
    ----------
    sample_results : list of dict
        Each dict is the output of evaluate_sample().

    Returns
    -------
    dict with keys:
        'edge_jaccard_weighted': weight-averaged by (TP+FP+FN) per sample.
        'div_jaccard_micro': micro-averaged division Jaccard.
        'per_sample': list of per-sample dicts.
    """
    total_edge_weight = 0
    weighted_edge_jaccard = 0.0

    total_div_tp = 0
    total_div_fp = 0
    total_div_fn = 0

    for res in sample_results:
        w = res["edge_tp"] + res["edge_fp"] + res["edge_fn"] + res["node_overcount_penalty"]
        weighted_edge_jaccard += res["edge_jaccard"] * w
        total_edge_weight += w

        total_div_tp += res["div_tp"]
        total_div_fp += res["div_fp"]
        total_div_fn += res["div_fn"]

    edge_jaccard_weighted = weighted_edge_jaccard / total_edge_weight if total_edge_weight > 0 else 0.0

    div_denom = total_div_tp + total_div_fp + total_div_fn
    div_jaccard_micro = total_div_tp / div_denom if div_denom > 0 else 0.0

    return {
        "edge_jaccard_weighted": edge_jaccard_weighted,
        "div_jaccard_micro": div_jaccard_micro,
        "n_samples": len(sample_results),
        "per_sample": sample_results,
    }


if __name__ == "__main__":
    print("metric.py loaded successfully. Use evaluate_sample() and evaluate_all().")

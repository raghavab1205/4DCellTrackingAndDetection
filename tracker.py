"""
tracker.py — Frame-to-frame linking and division detection.

Public API
----------
link_frames(centroids_t0, centroids_t1, ...) -> list of (i, j) index pairs
build_tracks(all_centroids, ...)             -> TrackGraph (nodes + edges)
detect_divisions(track_graph)                -> list of division events
TrackGraph                                   -> simple graph container
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from data_io import VOXEL_SCALE, voxel_to_physical


class TrackGraph:
    """Simple tracking graph container.

    Attributes
    ----------
    nodes : list of dict
        Each dict has keys: 'id', 't', 'z', 'y', 'x' (voxel coords, int).
    edges : list of tuple
        Each tuple is (source_node_id, target_node_id).
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self._next_id = 1

    def add_node(self, t: int, z: float, y: float, x: float) -> int:
        """Add a node and return its ID."""
        nid = self._next_id
        self._next_id += 1
        self.nodes.append({
            "id": nid,
            "t": t,
            "z": int(round(z)),
            "y": int(round(y)),
            "x": int(round(x)),
        })
        return nid

    def add_edge(self, source_id: int, target_id: int):
        """Add a directed edge from source to target."""
        self.edges.append((source_id, target_id))

    def get_nodes_at_time(self, t: int) -> list:
        """Return all nodes at timepoint t."""
        return [n for n in self.nodes if n["t"] == t]

    def get_outgoing_edges(self, node_id: int) -> list:
        """Return target IDs for outgoing edges from node_id."""
        return [tgt for src, tgt in self.edges if src == node_id]

    def get_incoming_edges(self, node_id: int) -> list:
        """Return source IDs for incoming edges to node_id."""
        return [src for src, tgt in self.edges if tgt == node_id]


def link_frames(
    centroids_t0: np.ndarray,
    centroids_t1: np.ndarray,
    voxel_scale: np.ndarray = None,
    max_distance_um: float = 7.0,
) -> list:
    """Link detections between two consecutive frames using Hungarian matching.

    Parameters
    ----------
    centroids_t0 : np.ndarray, shape (N, 3)
        Centroids at time t in voxel coords (z, y, x).
    centroids_t1 : np.ndarray, shape (M, 3)
        Centroids at time t+1 in voxel coords (z, y, x).
    voxel_scale : np.ndarray, optional
        Physical voxel scale (z, y, x) in µm.
    max_distance_um : float
        Maximum matching distance in µm. Pairs farther apart are not linked.

    Returns
    -------
    list of (int, int)
        Index pairs (i, j) where i indexes centroids_t0 and j indexes centroids_t1.
        Unmatched detections are not included.
    """
    if voxel_scale is None:
        voxel_scale = VOXEL_SCALE

    if len(centroids_t0) == 0 or len(centroids_t1) == 0:
        return []

    # Convert to physical coordinates
    phys_t0 = voxel_to_physical(centroids_t0)
    phys_t1 = voxel_to_physical(centroids_t1)

    # Compute pairwise distance matrix
    cost = cdist(phys_t0, phys_t1, metric="euclidean")

    # Mask out distances beyond threshold
    # Use a large sentinel value for invalid pairs
    SENTINEL = 1e6
    cost_masked = np.where(cost <= max_distance_um, cost, SENTINEL)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_masked)

    # Filter out assignments that exceed the distance threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] <= max_distance_um:
            matches.append((int(i), int(j)))

    return matches


def build_tracks(
    all_centroids: dict,
    voxel_scale: np.ndarray = None,
    max_distance_um: float = 7.0,
) -> TrackGraph:
    """Build a full tracking graph from per-frame detections.

    Parameters
    ----------
    all_centroids : dict
        Mapping from timepoint (int) -> np.ndarray of shape (N_t, 3),
        centroids in voxel coords (z, y, x).
    voxel_scale : np.ndarray, optional
    max_distance_um : float

    Returns
    -------
    TrackGraph
    """
    if voxel_scale is None:
        voxel_scale = VOXEL_SCALE

    graph = TrackGraph()
    timepoints = sorted(all_centroids.keys())

    if len(timepoints) == 0:
        return graph

    # Add nodes for all timepoints, keep track of node IDs per frame
    frame_node_ids = {}  # t -> list of node IDs (same order as centroids)

    for t in timepoints:
        centroids = all_centroids[t]
        ids = []
        for c in centroids:
            nid = graph.add_node(t, c[0], c[1], c[2])
            ids.append(nid)
        frame_node_ids[t] = ids

    # Link consecutive frames
    for i in range(len(timepoints) - 1):
        t0, t1 = timepoints[i], timepoints[i + 1]

        # Only link consecutive timepoints (t1 == t0 + 1)
        if t1 != t0 + 1:
            continue

        centroids_t0 = all_centroids[t0]
        centroids_t1 = all_centroids[t1]
        ids_t0 = frame_node_ids[t0]
        ids_t1 = frame_node_ids[t1]

        matches = link_frames(centroids_t0, centroids_t1, voxel_scale, max_distance_um)

        for idx0, idx1 in matches:
            graph.add_edge(ids_t0[idx0], ids_t1[idx1])

    return graph


def detect_divisions(
    graph: TrackGraph,
    all_centroids: dict,
    voxel_scale: np.ndarray = None,
    max_distance_um: float = 7.0,
) -> TrackGraph:
    """Detect cell divisions by finding unmatched nodes and assigning them
    as second daughters of nearby parent cells.

    A division is detected when a node at time t has no incoming edge
    (unmatched new detection) and its nearest potential parent at t-1
    already has exactly one outgoing edge. The unmatched node is then
    linked as a second daughter, creating a 1->2 split.

    Parameters
    ----------
    graph : TrackGraph
        The tracking graph with 1-to-1 links already established.
    all_centroids : dict
        t -> np.ndarray of shape (N_t, 3) in voxel coords.
    voxel_scale : np.ndarray, optional
    max_distance_um : float

    Returns
    -------
    TrackGraph
        The same graph, modified in place with division edges added.
    """
    if voxel_scale is None:
        voxel_scale = VOXEL_SCALE

    timepoints = sorted(all_centroids.keys())

    for i in range(1, len(timepoints)):
        t = timepoints[i]
        t_prev = timepoints[i - 1]

        if t != t_prev + 1:
            continue

        # Find nodes at t with no incoming edges (unmatched / new)
        nodes_t = graph.get_nodes_at_time(t)
        unmatched = [n for n in nodes_t if len(graph.get_incoming_edges(n["id"])) == 0]

        if len(unmatched) == 0:
            continue

        # Get parent nodes at t-1
        nodes_prev = graph.get_nodes_at_time(t_prev)
        if len(nodes_prev) == 0:
            continue

        # Build arrays for distance computation
        parent_coords = np.array([[n["z"], n["y"], n["x"]] for n in nodes_prev], dtype=float)
        parent_ids = [n["id"] for n in nodes_prev]
        parent_phys = voxel_to_physical(parent_coords)

        for node in unmatched:
            child_coord = np.array([[node["z"], node["y"], node["x"]]], dtype=float)
            child_phys = voxel_to_physical(child_coord)

            # Distance to all potential parents
            dists = cdist(child_phys, parent_phys, metric="euclidean")[0]
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]

            if nearest_dist > max_distance_um:
                continue

            parent_id = parent_ids[nearest_idx]

            # Only create a division if this parent already has exactly 1 outgoing edge
            # (making it a 1->2 split)
            outgoing = graph.get_outgoing_edges(parent_id)
            if len(outgoing) == 1:
                graph.add_edge(parent_id, node["id"])

    return graph


if __name__ == "__main__":
    from data_io import load_volume_frame, VOXEL_SCALE as VS
    from detector import detect_cells_threshold

    DATA_DIR = "f:/4D cell tracking/biohub-cell-tracking-during-development"
    zarr_path = f"{DATA_DIR}/train/44b6_0113de3b.zarr"

    # Detect cells in two consecutive frames
    print("Loading and detecting cells in frames 0 and 1...")
    frame0 = load_volume_frame(zarr_path, 0)
    frame1 = load_volume_frame(zarr_path, 1)

    c0 = detect_cells_threshold(frame0, VS)
    c1 = detect_cells_threshold(frame1, VS)
    print(f"Frame 0: {len(c0)} detections, Frame 1: {len(c1)} detections")

    # Link
    matches = link_frames(c0, c1, VS)
    print(f"Linked {len(matches)} pairs")

    # Build tracks for 3 frames
    print("\nBuilding tracks over 3 frames...")
    frame2 = load_volume_frame(zarr_path, 2)
    c2 = detect_cells_threshold(frame2, VS)
    print(f"Frame 2: {len(c2)} detections")

    all_c = {0: c0, 1: c1, 2: c2}
    graph = build_tracks(all_c, VS)
    graph = detect_divisions(graph, all_c, VS)
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Check for divisions
    for node in graph.nodes:
        out = graph.get_outgoing_edges(node["id"])
        if len(out) >= 2:
            print(f"  Division at node {node['id']} (t={node['t']}): -> {out}")

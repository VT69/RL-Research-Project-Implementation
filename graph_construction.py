"""
graph_construction.py
=====================
Builds the predefined directed graph for GLD2-GNN from the 16-sensor
plantar pressure layout described in the paper (Fig. 3).

Sensor layout (1-indexed in paper, 0-indexed here):
    Left foot  : v0  – v7   (sensors 1–8)
    Right foot : v8  – v15  (sensors 9–16)

Edge rules (from Section III-B-1):
  1. Within each foot, directed edges run heel → toe (walking direction).
  2. Bidirectional edges between metatarsal nodes (v1↔v2, left; v9↔v10, right).
  3. Bidirectional edges between calcaneal nodes  (v5↔v6, left; v13↔v14, right).
  4. Cross-foot edges: v7→v8 and v15→v0  (connect the two feet).

Total: 16 nodes, 26 directed edges (matching the paper).

Outputs
-------
  get_adjacency_matrix()  -> torch.Tensor  [NV, NV]  (binary, float32)
  get_source_matrix()     -> torch.Tensor  [NV, NE]  S[i,j]=1 if node i is source of edge j
  get_destination_matrix()-> torch.Tensor  [NV, NE]  D[i,j]=1 if node i is dest  of edge j
  get_edge_index()        -> torch.Tensor  [2,  NE]  COO format for PyG compatibility
  normalize_matrix()      -> torch.Tensor            row-wise degree normalisation  Λ⁻¹ M
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NV = 16   # number of sensor nodes
EPS = 1e-6  # small value to avoid division by zero in normalisation


# ---------------------------------------------------------------------------
# Edge list  (src, dst)  — 0-indexed
# ---------------------------------------------------------------------------

def _build_edge_list():
    """
    Returns a list of (src, dst) tuples representing the 26 directed edges.

    Left foot nodes  : 0-7   mapped to paper nodes v1-v8
    Right foot nodes : 8-15  mapped to paper nodes v9-v16
    """
    edges = []

    # --- Left foot: walking-direction chain v0 → v7 ---
    # Heel (v0) progresses toward toe (v7)
    left_chain = [(0, 1), (1, 3), (3, 5), (5, 7),
                  (0, 2), (2, 4), (4, 6), (6, 7)]
    edges.extend(left_chain)

    # Bidirectional: metatarsal pair (v1 ↔ v2) and calcaneal pair (v5 ↔ v6)
    edges.extend([(1, 2), (2, 1), (5, 6), (6, 5)])

    # --- Right foot: walking-direction chain v8 → v15 ---
    right_chain = [(8, 9), (9, 11), (11, 13), (13, 15),
                   (8, 10), (10, 12), (12, 14), (14, 15)]
    edges.extend(right_chain)

    # Bidirectional: metatarsal pair (v9 ↔ v10) and calcaneal pair (v13 ↔ v14)
    edges.extend([(9, 10), (10, 9), (13, 14), (14, 13)])

    # --- Cross-foot edges ---
    edges.extend([(7, 8), (15, 0)])

    assert len(edges) == 26, f"Expected 26 edges, got {len(edges)}"
    return edges


# ---------------------------------------------------------------------------
# Core graph tensors
# ---------------------------------------------------------------------------

def get_edge_index():
    """
    Returns the edge index in COO format: shape [2, NE].
    Row 0 = source nodes, Row 1 = destination nodes.
    Compatible with torch_geometric conventions.
    """
    edges = _build_edge_list()
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, NE]
    return edge_index


def get_adjacency_matrix():
    """
    Returns the binary adjacency matrix A of shape [NV, NV].
    A[i, j] = 1.0 if there is a directed edge from node i to node j.
    """
    edges = _build_edge_list()
    A = torch.zeros(NV, NV, dtype=torch.float32)
    for src, dst in edges:
        A[src, dst] = 1.0
    return A  # [NV, NV]


def get_source_matrix():
    """
    Returns the source incidence matrix S of shape [NV, NE].
    S[i, j] = 1.0  if node i is the SOURCE of edge j.

    Used in DyDGN unit to route node features → edge features along
    outgoing edges (equation 13 in the paper).
    """
    edges = _build_edge_list()
    NE = len(edges)
    S = torch.zeros(NV, NE, dtype=torch.float32)
    for j, (src, _) in enumerate(edges):
        S[src, j] = 1.0
    return S  # [NV, NE]


def get_destination_matrix():
    """
    Returns the destination incidence matrix D of shape [NV, NE].
    D[i, j] = 1.0  if node i is the DESTINATION of edge j.

    Used in DyDGN unit to route edge features → node features along
    incoming edges (equation 14 in the paper).
    """
    edges = _build_edge_list()
    NE = len(edges)
    D = torch.zeros(NV, NE, dtype=torch.float32)
    for j, (_, dst) in enumerate(edges):
        D[dst, j] = 1.0
    return D  # [NV, NE]


def normalize_matrix(M: torch.Tensor) -> torch.Tensor:
    """
    Row-wise degree normalisation:  Λ⁻¹ M
    where  Λ[i,i] = sum_j (M[i,j]) + EPS

    Matches equations (11) and (12) in the paper for both static
    source/destination matrices and their dynamic counterparts.

    Args:
        M: float tensor of shape [NV, NE] or [NV, NV]

    Returns:
        Normalised matrix of the same shape.
    """
    row_sums = M.sum(dim=1, keepdim=True) + EPS   # [NV, 1]
    return M / row_sums


def get_normalized_source_matrix():
    """Returns Ŝ = Λ⁻¹ S  (normalised source matrix)."""
    return normalize_matrix(get_source_matrix())


def get_normalized_destination_matrix():
    """Returns D̂ = Λ⁻¹ D  (normalised destination matrix)."""
    return normalize_matrix(get_destination_matrix())


# ---------------------------------------------------------------------------
# Edge value computation  (equation 4 in the paper)
# ---------------------------------------------------------------------------

def compute_edge_values(node_features: torch.Tensor) -> torch.Tensor:
    """
    Computes directed edge values from node features using equation (4):
        e(i,j) = v_dst - v_src

    Args:
        node_features: tensor of shape [B, C, T, NV]
                       B=batch, C=channels, T=time, NV=nodes

    Returns:
        edge_features: tensor of shape [B, C, T, NE]
    """
    edges = _build_edge_list()
    src_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)

    # Index along the NV dimension (last dim)
    v_src = node_features[..., src_idx]   # [B, C, T, NE]
    v_dst = node_features[..., dst_idx]   # [B, C, T, NE]
    return v_dst - v_src                  # [B, C, T, NE]


# ---------------------------------------------------------------------------
# Convenience: return all graph components as a named dict
# ---------------------------------------------------------------------------

def get_graph_components(device: torch.device = torch.device("cpu")) -> dict:
    """
    Returns all static graph components as a dictionary, moved to `device`.

    Keys
    ----
    A       : adjacency matrix          [NV, NV]
    S       : source incidence matrix   [NV, NE]
    D       : destination incidence     [NV, NE]
    S_hat   : normalised S              [NV, NE]
    D_hat   : normalised D              [NV, NE]
    edge_index : COO format             [2,  NE]
    NV      : int  (16)
    NE      : int  (26)
    """
    NE = len(_build_edge_list())
    components = {
        "A":          get_adjacency_matrix(),
        "S":          get_source_matrix(),
        "D":          get_destination_matrix(),
        "S_hat":      get_normalized_source_matrix(),
        "D_hat":      get_normalized_destination_matrix(),
        "edge_index": get_edge_index(),
        "NV":         NV,
        "NE":         NE,
    }
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in components.items()}


# ---------------------------------------------------------------------------
# Quick sanity check  (run this file directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gc = get_graph_components()

    print("=" * 50)
    print("Graph construction sanity check")
    print("=" * 50)
    print(f"  Nodes (NV)            : {gc['NV']}")
    print(f"  Edges (NE)            : {gc['NE']}")
    print(f"  Adjacency matrix      : {gc['A'].shape}  sum={gc['A'].sum().item():.0f}")
    print(f"  Source matrix S       : {gc['S'].shape}")
    print(f"  Destination matrix D  : {gc['D'].shape}")
    print(f"  Normalised S_hat      : {gc['S_hat'].shape}")
    print(f"  Normalised D_hat      : {gc['D_hat'].shape}")
    print(f"  Edge index            : {gc['edge_index'].shape}")

    # Verify each edge appears exactly once in S and D
    assert gc['S'].sum().item() == 26.0, "Each edge should have exactly 1 source"
    assert gc['D'].sum().item() == 26.0, "Each edge should have exactly 1 destination"

    # Verify cross-foot edges exist
    A = gc['A']
    assert A[7, 8].item() == 1.0,  "Missing cross-foot edge v7 → v8"
    assert A[15, 0].item() == 1.0, "Missing cross-foot edge v15 → v0"

    # Verify bidirectional metatarsal edges
    assert A[1, 2].item() == 1.0 and A[2, 1].item() == 1.0, "Missing v1↔v2"
    assert A[5, 6].item() == 1.0 and A[6, 5].item() == 1.0, "Missing v5↔v6"

    # Test edge value computation
    dummy = torch.randn(4, 1, 64, NV)   # B=4, C=1, T=64, NV=16
    ev = compute_edge_values(dummy)
    print(f"\n  Edge value test input : {dummy.shape}")
    print(f"  Edge value output     : {ev.shape}  (expected [4, 1, 64, 26])")

    print("\n  All assertions passed.")

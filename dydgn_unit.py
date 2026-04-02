"""
dydgn_unit.py
=============
Dynamic Directed Graph Network (DyDGN) unit — the second sub-module
inside every DyDGNN block (paper Section III-C-3, Fig. 5).

Purpose
-------
Learns spatial relationships between sensor nodes and pressure-transmission
edges by aggregating three kinds of information at every time slot:

    1. Static/adaptive features — from the predefined directed graph
       (normalised matrices Ŝ, D̂ that become trainable after warm-up)
    2. Dynamic features — from the time-varying adjacency produced by DGLUnit
       (normalised matrices Ŝ^dy, D̂^dy per slot)
    3. Input features — raw node tensor V(t_k) and edge tensor E(t_k)

Node feature update  (Fig. 5b, eq. 15):
    V̂(t_k) = [ V(t_k),
                E(t_k) · Ŝ^T,          ← features along adaptive in-edges
                E(t_k) · D̂^T,          ← features along adaptive out-edges
                E^dy(t_k) · Ŝ^dy^T,    ← features along dynamic in-edges
                E^dy(t_k) · D̂^dy^T ]   ← features along dynamic out-edges
    → linear layer → BN → ReLU

Edge feature update  (Fig. 5a, eq. 16):
    Ê(t_k) = [ E(t_k) ⊕ E^dy(t_k),
                V(t_k) · [Ŝ, Ŝ^dy]^T,  ← source-node features
                V(t_k) · [D̂, D̂^dy]^T ] ← destination-node features
    → two separate linear layers → BN → ReLU

After processing all s slots the outputs are concatenated along the
time dimension to restore shape [B, C_out, T, NV/NE].

Key design details
------------------
* Ŝ and D̂ start as fixed normalised incidence matrices (from graph_construction).
  After `trainable_after` gradient steps they become nn.Parameters so the
  model can learn global adaptive dependencies (paper Section III-C-3).
* Dynamic edge count NE_dy varies per time slot — handled by padding all
  slots to the maximum NE_dy in the batch before concatenation (this is the
  "number of edges adjusted for consistency" mentioned in Section V).
* Separate linear layers for nodes and edges respect their distinct roles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from graph_construction import (
    NV as NV_DEFAULT,
    get_normalized_source_matrix,
    get_normalized_destination_matrix,
    normalize_matrix,
)


# ---------------------------------------------------------------------------
# DyDGN unit
# ---------------------------------------------------------------------------

class DyDGNUnit(nn.Module):
    """
    Dynamic Directed Graph Network unit.

    Args:
        C_in            : number of input channels
        C_out           : number of output channels
        NV              : number of sensor nodes (default 16)
        NE              : number of predefined edges (default 26)
        s               : number of time slots
        trainable_after : global step after which Ŝ, D̂ become trainable
                          parameters (set high to disable, 0 to always train)
    """

    def __init__(
        self,
        C_in:            int,
        C_out:           int,
        NV:              int = NV_DEFAULT,
        NE:              int = 26,
        s:               int = 4,
        trainable_after: int = 5,       # epochs before adaptive matrices unfreeze
    ):
        super().__init__()

        self.C_in  = C_in
        self.C_out = C_out
        self.NV    = NV
        self.NE    = NE
        self.s     = s
        self.trainable_after = trainable_after
        self._step = 0      # internal epoch counter; caller increments via step()

        # ------------------------------------------------------------------
        # Adaptive source / destination matrices  (Ŝ, D̂)
        # Start as fixed buffers; unfreeze to nn.Parameters after warm-up.
        # ------------------------------------------------------------------
        S_hat_init = get_normalized_source_matrix()       # [NV, NE]
        D_hat_init = get_normalized_destination_matrix()  # [NV, NE]

        # Register as buffers first (non-trainable)
        self.register_buffer("S_hat_buf", S_hat_init.clone())
        self.register_buffer("D_hat_buf", D_hat_init.clone())

        # Separate trainable parameters (will be used after warm-up)
        self.S_hat_param = nn.Parameter(S_hat_init.clone(), requires_grad=False)
        self.D_hat_param = nn.Parameter(D_hat_init.clone(), requires_grad=False)

        # ------------------------------------------------------------------
        # Node update linear layer  (eq. 15)
        # Input: 5 × C_in features concatenated along channel dim
        # ------------------------------------------------------------------
        self.node_linear = nn.Linear(5 * C_in, C_out, bias=False)
        self.node_bn     = nn.BatchNorm1d(C_out)

        # ------------------------------------------------------------------
        # Edge update linear layers  (eq. 16)
        # Layer 1 integrates input edges + node info: (3*C_in) → C_out
        # Layer 2 extracts features from predefined + dynamic edges:
        #         input is the concatenated [E, E^dy] portion of Ê
        # ------------------------------------------------------------------
        # Ê = [E ⊕ E^dy , V·[Ŝ,Ŝ^dy]^T , V·[D̂,D̂^dy]^T]
        # channel dims:  [C_in(+C_in_dy)  ,  C_in          ,  C_in      ]
        # We use C_in for both E and E^dy slots (padded to same size)
        self.edge_linear1 = nn.Linear(3 * C_in, C_out, bias=False)
        self.edge_linear2 = nn.Linear(C_out,    C_out, bias=False)
        self.edge_bn      = nn.BatchNorm1d(C_out)

    # ------------------------------------------------------------------
    # Warm-up control
    # ------------------------------------------------------------------

    def step(self):
        """Call once per epoch to advance the internal step counter."""
        self._step += 1
        if self._step >= self.trainable_after:
            self.S_hat_param.requires_grad_(True)
            self.D_hat_param.requires_grad_(True)

    def _get_adaptive_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns current Ŝ and D̂ (fixed buffer or trainable parameter)."""
        if self._step >= self.trainable_after:
            return self.S_hat_param, self.D_hat_param
        return self.S_hat_buf, self.D_hat_buf

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_dynamic_edges(
        E_dy_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Pads dynamic edge tensors to the same NE_dy across all slots so
        they can be processed consistently.

        Also returns padded S_dy and D_dy (already [NV, NE_dy] per slot,
        but may differ in NE_dy between slots).

        Returns lists in the original order, each padded to max_NE_dy.
        """
        max_ne = max(e.shape[-1] for e in E_dy_list)
        padded = []
        for e in E_dy_list:
            ne = e.shape[-1]
            if ne < max_ne:
                pad = torch.zeros(*e.shape[:-1], max_ne - ne,
                                  device=e.device, dtype=e.dtype)
                e = torch.cat([e, pad], dim=-1)
            padded.append(e)
        return padded, max_ne

    @staticmethod
    def _pad_incidence(
        mat_list: List[torch.Tensor], max_ne: int
    ) -> List[torch.Tensor]:
        """Pads [NV, NE_dy] incidence matrices to [NV, max_ne]."""
        padded = []
        for m in mat_list:
            ne = m.shape[1]
            if ne < max_ne:
                pad = torch.zeros(m.shape[0], max_ne - ne,
                                  device=m.device, dtype=m.dtype)
                m = torch.cat([m, pad], dim=1)
            padded.append(m)
        return padded

    # ------------------------------------------------------------------
    # Per-slot update
    # ------------------------------------------------------------------

    def _update_slot(
        self,
        V_k:   torch.Tensor,    # [B, C_in, T_slot, NV]
        E_k:   torch.Tensor,    # [B, C_in, T_slot, NE]
        E_dy_k: torch.Tensor,   # [B, C_in, T_slot, NE_dy]
        S_hat:  torch.Tensor,   # [NV, NE]   adaptive (or fixed)
        D_hat:  torch.Tensor,   # [NV, NE]
        S_dy_k: torch.Tensor,   # [NV, NE_dy]
        D_dy_k: torch.Tensor,   # [NV, NE_dy]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes updated node and edge features for one time slot.

        Returns:
            V_out : [B, C_out, T_slot, NV]
            E_out : [B, C_out, T_slot, NE]
        """
        B, C, T_slot, NV = V_k.shape
        NE     = E_k.shape[-1]
        NE_dy  = E_dy_k.shape[-1]

        # ---- Node feature aggregation (eq. 15) ----
        # E(t_k) · Ŝ^T  →  edge→node along adaptive incoming/outgoing
        # matmul: [B, C, T, NE] x [NE, NV]  =  [B, C, T, NV]
        E_in_adapt  = torch.matmul(E_k,    S_hat.T)   # [B,C,T,NV]
        E_out_adapt = torch.matmul(E_k,    D_hat.T)   # [B,C,T,NV]

        # E^dy(t_k) · Ŝ^dy^T  →  edge→node along dynamic edges
        E_in_dyn    = torch.matmul(E_dy_k, S_dy_k.T)  # [B,C,T,NV]
        E_out_dyn   = torch.matmul(E_dy_k, D_dy_k.T)  # [B,C,T,NV]

        # Concatenate along channel dim → [B, 5*C, T, NV]
        V_hat = torch.cat([V_k, E_in_adapt, E_out_adapt,
                           E_in_dyn, E_out_dyn], dim=1)

        # Linear: treat (B, T, NV) as batch of C-vectors
        # [B, 5C, T, NV] → [B*T*NV, 5C] → linear → [B*T*NV, C_out]
        V_hat = V_hat.permute(0, 2, 3, 1).reshape(B * T_slot * NV, 5 * C)
        V_out = self.node_linear(V_hat)                # [B*T*NV, C_out]

        # BN over C_out features
        V_out = self.node_bn(V_out)
        V_out = F.relu(V_out)
        V_out = V_out.reshape(B, T_slot, NV, self.C_out)
        V_out = V_out.permute(0, 3, 1, 2)             # [B, C_out, T, NV]

        # ---- Edge feature aggregation (eq. 16) ----
        # Concatenate predefined + dynamic edge data: [B, C, T, NE+NE_dy]
        E_concat = torch.cat([E_k, E_dy_k], dim=-1)   # [B,C,T,NE+NE_dy]

        # Source node features: V(t_k) · [Ŝ, Ŝ^dy]
        # [Ŝ, Ŝ^dy]: [NV, NE+NE_dy]
        S_combined = torch.cat([S_hat, S_dy_k], dim=1)  # [NV, NE+NE_dy]
        D_combined = torch.cat([D_hat, D_dy_k], dim=1)  # [NV, NE+NE_dy]

        # V_k: [B,C,T,NV] × [NV, NE+NE_dy] → [B,C,T,NE+NE_dy]
        V_src_info = torch.matmul(V_k, S_combined)
        V_dst_info = torch.matmul(V_k, D_combined)

        # Ê = [E_concat, V_src_info, V_dst_info] → [B, 3C, T, NE+NE_dy]
        E_hat = torch.cat([E_concat, V_src_info, V_dst_info], dim=1)

        # Linear 1: integrate edges + node info
        NE_total = NE + NE_dy
        E_hat = E_hat.permute(0, 2, 3, 1).reshape(B * T_slot * NE_total, 3 * C)
        E_out = self.edge_linear1(E_hat)               # [B*T*NE_total, C_out]

        # Linear 2: extract from predefined + dynamic edges
        E_out = self.edge_linear2(E_out)               # [B*T*NE_total, C_out]
        E_out = self.edge_bn(E_out)
        E_out = F.relu(E_out)
        E_out = E_out.reshape(B, T_slot, NE_total, self.C_out)
        E_out = E_out.permute(0, 3, 1, 2)             # [B, C_out, T, NE_total]

        # Keep only the predefined NE edges (drop dynamic padding columns)
        E_out = E_out[..., :NE]                        # [B, C_out, T, NE]

        return V_out, E_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        V:         torch.Tensor,          # [B, C_in, T, NV]
        E:         torch.Tensor,          # [B, C_in, T, NE]
        S_dy_list: List[torch.Tensor],    # s × [NV, NE_dy(t)]
        D_dy_list: List[torch.Tensor],    # s × [NV, NE_dy(t)]
        E_dy_list: List[torch.Tensor],    # s × [B, C_in, T_slot, NE_dy(t)]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes all s time slots and concatenates results.

        Args:
            V         : [B, C_in, T, NV]   node features (time stream or motion stream)
            E         : [B, C_in, T, NE]   edge features
            S_dy_list : per-slot dynamic source incidence matrices
            D_dy_list : per-slot dynamic destination incidence matrices
            E_dy_list : per-slot dynamic edge feature tensors

        Returns:
            V_out : [B, C_out, T, NV]
            E_out : [B, C_out, T, NE]
        """
        B, C, T, NV = V.shape
        T_slot = T // self.s
        device = V.device

        # Move adaptive matrices to correct device
        S_hat, D_hat = self._get_adaptive_matrices()
        S_hat = S_hat.to(device)
        D_hat = D_hat.to(device)

        # Pad dynamic edge tensors so all slots share the same NE_dy
        E_dy_padded, max_ne_dy = self._pad_dynamic_edges(E_dy_list)
        S_dy_padded = self._pad_incidence(
            [m.to(device) for m in S_dy_list], max_ne_dy
        )
        D_dy_padded = self._pad_incidence(
            [m.to(device) for m in D_dy_list], max_ne_dy
        )

        V_slots, E_slots = [], []

        for k in range(self.s):
            V_k    = V[:, :, k * T_slot: (k + 1) * T_slot, :]   # [B,C,T_slot,NV]
            E_k    = E[:, :, k * T_slot: (k + 1) * T_slot, :]   # [B,C,T_slot,NE]
            E_dy_k = E_dy_padded[k].to(device)                   # [B,C,T_slot,NE_dy]
            S_dy_k = S_dy_padded[k]                               # [NV, NE_dy]
            D_dy_k = D_dy_padded[k]                               # [NV, NE_dy]

            V_k_out, E_k_out = self._update_slot(
                V_k, E_k, E_dy_k, S_hat, D_hat, S_dy_k, D_dy_k
            )
            V_slots.append(V_k_out)
            E_slots.append(E_k_out)

        # Concatenate along temporal dim: list of [B, C_out, T_slot, NV/NE]
        V_out = torch.cat(V_slots, dim=2)   # [B, C_out, T, NV]
        E_out = torch.cat(E_slots, dim=2)   # [B, C_out, T, NE]

        return V_out, E_out


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from graph_construction import get_graph_components
    from dgl_unit import DGLUnit, EdgeGenerationUnit

    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C_in, C_out = 4, 1, 32
    T, NV, NE, s   = 128, 16, 26, 4

    gc       = get_graph_components(device)
    A_static = gc["A"]

    node_data = torch.randn(B, C_in, T, NV)
    edge_data = torch.randn(B, C_in, T, NE)

    print("=" * 55)
    print("dydgn_unit.py sanity check")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Build dynamic adjacency matrices via DGL unit.
    # Current API: DGLUnit(c_in, t_slot, s); forward(node_data) → list[Tensor]
    # ------------------------------------------------------------------
    t_slot   = T // s
    dgl      = DGLUnit(c_in=C_in, t_slot=t_slot, s=s)
    adj_list = dgl(node_data)   # list of s tensors [NV, NV]

    print(f"  DGL unit produced {len(adj_list)} adjacency matrices")
    for i, A in enumerate(adj_list):
        nnz = (A > 0).sum().item()
        print(f"  Slot {i}: shape={A.shape}  non-zero={nnz}")

    # ------------------------------------------------------------------
    # Build dynamic edge features via EdgeGenerationUnit.
    # API: EdgeGenerationUnit(max_dynamic_edges=NE)
    # forward(node_data, adj_matrices) → (dy_edges_list, S_dy_list, D_dy_list)
    # ------------------------------------------------------------------
    edge_gen                            = EdgeGenerationUnit(max_dynamic_edges=NE)
    dy_edges_list, S_dy_list, D_dy_list = edge_gen(node_data, adj_list)

    for i in range(s):
        print(f"  Slot {i}: NE_dy={S_dy_list[i].shape[1]}  "
              f"E_dy={dy_edges_list[i].shape}")

    # ------------------------------------------------------------------
    # Run DyDGN unit.
    # forward(V, E, S_dy_list, D_dy_list, E_dy_list)
    # ------------------------------------------------------------------
    dydgn = DyDGNUnit(C_in=C_in, C_out=C_out, NV=NV, NE=NE, s=s)
    V_out, E_out = dydgn(
        node_data, edge_data, S_dy_list, D_dy_list, dy_edges_list
    )

    print(f"\n  DyDGNUnit output:")
    print(f"    V_out : {V_out.shape}  (expected [4, 32, 128, 16])")
    print(f"    E_out : {E_out.shape}  (expected [4, 32, 128, 26])")

    assert V_out.shape == (B, C_out, T, NV), "Node output shape mismatch"
    assert E_out.shape == (B, C_out, T, NE), "Edge output shape mismatch"

    # Check gradient flows through node_linear and edge_linear layers
    loss = V_out.sum() + E_out.sum()
    loss.backward()

    print(f"\n  Gradient on node_linear.weight : "
          f"{dydgn.node_linear.weight.grad.abs().mean():.6f}")
    print(f"  Gradient on edge_linear1.weight: "
          f"{dydgn.edge_linear1.weight.grad.abs().mean():.6f}")
    print(f"  Gradient on edge_linear2.weight: "
          f"{dydgn.edge_linear2.weight.grad.abs().mean():.6f}")

    # Verify adaptive matrices unfreeze after warm-up
    print(f"\n  Before step(): S_hat_param.requires_grad = "
          f"{dydgn.S_hat_param.requires_grad}")
    for _ in range(dydgn.trainable_after):
        dydgn.step()
    print(f"  After  step(): S_hat_param.requires_grad = "
          f"{dydgn.S_hat_param.requires_grad}")

    print("\n  All assertions passed.")

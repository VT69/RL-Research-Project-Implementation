"""
dgl_unit.py
===========
Dynamic Graph Learning (DGL) unit — the first component inside every
DyDGNN block (paper Section III-C-1, Fig. 4).

Purpose
-------
The predefined directed graph captures static anatomical relationships
between plantar sensors.  During walking, however, the actual pressure
transmission topology changes from frame to frame.  The DGL unit learns
these *dynamic* adjacency matrices by processing node data at each time
slot through five specialised branches.

Five branches
-------------
  Branch 1 : global — uses ALL 16 nodes, learns long-range connectivity
  Branch 2 : left-foot local  — nodes 0-7  only
  Branch 3 : cross-foot left→right — bidirectional inter-foot relationships
  Branch 4 : cross-foot right→left — bidirectional inter-foot relationships
  Branch 5 : right-foot local — nodes 8-15 only

Each branch produces a [NV, NV] adjacency matrix; the five are fused with
learned weights into one matrix per time slot.  Post-processing (equations
6-8) enforces sparsity, top-k selection, and consistency with the
predefined graph structure.

Input / output shapes
---------------------
  Input  : node_data  [B, C_in, T, NV]
  Output : list of s adjacency matrices, each [NV, NV]   (one per time slot)
           dynamic edge data tensor  [B, C_in, T, NE_dy]

where s  = number of time slots (set per DyDGNN block layer)
      NV = 16  (sensor nodes)
      NE_dy varies per time slot (number of learned dynamic edges)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_construction import get_graph_components, NV, normalize_matrix


# ---------------------------------------------------------------------------
# Hyper-parameters matching the paper
# ---------------------------------------------------------------------------

THRESHOLD = 0.1   # equation (6): zero-out entries below this value
TOP_K     = 8     # equation (7): keep top-k edges per node in dynamic graph
D_DIM     = 32    # intermediate dimensionality for cross-foot branches


# ---------------------------------------------------------------------------
# Helper: 1x1 convolution block
# ---------------------------------------------------------------------------

class Conv1x1(nn.Module):
    """
    1x1 convolution over the (C, T) plane for a fixed NV slice.
    Used in branches 1, 2, 5 to map channel x time to adjacency logits.

    Input : [B, C_in, T_slot, N_nodes]
    Output: [B, C_out, T_slot, N_nodes]
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T_slot, N]
        return F.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Branch implementations
# ---------------------------------------------------------------------------

class GlobalBranch(nn.Module):
    """
    Branch 1: learns global NV x NV adjacency from all 16 nodes.

    Two 1x1 conv layers:
      - First  squeezes channel + time to 1
      - Second generates [NV, NV] adjacency logits
    """

    def __init__(self, c_in: int, t_slot: int):
        super().__init__()
        self.conv1  = Conv1x1(c_in, 1)
        self.linear = nn.Linear(NV, NV, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T_slot, NV]
        Returns:
            A_global: [NV, NV]  (averaged over batch)
        """
        h = self.conv1(x)                  # [B, 1, T_slot, NV]
        h = h.squeeze(1).mean(dim=1)       # [B, NV]  average over time slot
        A = self.linear(h)                 # [B, NV]
        # Outer product gives rank-1 adjacency per sample
        A = torch.bmm(A.unsqueeze(2), A.unsqueeze(1))   # [B, NV, NV]
        return A.mean(dim=0)               # [NV, NV]


class LocalFootBranch(nn.Module):
    """
    Branches 2 and 5: learns intra-foot adjacency for one foot's 8 nodes.

    Output is an 8x8 matrix embedded into a full [NV, NV] matrix.
    """

    def __init__(self, c_in: int, t_slot: int, foot: str = "left"):
        super().__init__()
        assert foot in ("left", "right")
        self.foot     = foot
        self.node_idx = list(range(8)) if foot == "left" else list(range(8, 16))
        N_foot        = 8

        self.conv1  = Conv1x1(c_in, 1)
        self.linear = nn.Linear(N_foot, N_foot, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T_slot, NV]
        Returns:
            A_foot: [NV, NV]  non-zero only for this foot's nodes
        """
        idx  = torch.tensor(self.node_idx, device=x.device)
        xf   = x[..., idx]                # [B, C_in, T_slot, 8]

        h = self.conv1(xf)                # [B, 1, T_slot, 8]
        h = h.squeeze(1).mean(dim=1)      # [B, 8]
        A_foot = self.linear(h)           # [B, 8]

        A_foot = torch.bmm(
            A_foot.unsqueeze(2), A_foot.unsqueeze(1)
        )                                 # [B, 8, 8]
        A_foot = A_foot.mean(dim=0)       # [8, 8]

        # Embed into full NV x NV matrix
        A_full = torch.zeros(NV, NV, device=x.device)
        for li, gi in enumerate(self.node_idx):
            for lj, gj in enumerate(self.node_idx):
                A_full[gi, gj] = A_foot[li, lj]

        return A_full                     # [NV, NV]


class CrossFootBranch(nn.Module):
    """
    Branches 3 and 4: learns bidirectional inter-foot relationships.

    Branch 3: A_LR = D_left @ D_right.T   (left to right)
    Branch 4: A_RL = D_right @ D_left.T   (right to left)
    """

    def __init__(self, c_in: int, t_slot: int, direction: str = "LR"):
        super().__init__()
        assert direction in ("LR", "RL")
        self.direction  = direction
        self.conv_left  = Conv1x1(c_in, D_DIM)
        self.conv_right = Conv1x1(c_in, D_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T_slot, NV]
        Returns:
            A_cross: [NV, NV] with cross-foot entries populated
        """
        x_left  = x[..., :8]              # [B, C_in, T_slot, 8]
        x_right = x[..., 8:]              # [B, C_in, T_slot, 8]

        # [B, D_DIM, T_slot, 8] -> average over time -> [B, 8, D_DIM]
        h_left  = self.conv_left(x_left).mean(dim=2).permute(0, 2, 1)
        h_right = self.conv_right(x_right).mean(dim=2).permute(0, 2, 1)

        if self.direction == "LR":
            A_cross = torch.bmm(h_left, h_right.transpose(1, 2))   # [B, 8, 8]
        else:
            A_cross = torch.bmm(h_right, h_left.transpose(1, 2))   # [B, 8, 8]

        A_cross = A_cross.mean(dim=0)     # [8, 8]

        A_full = torch.zeros(NV, NV, device=x.device)
        if self.direction == "LR":
            A_full[:8, 8:] = A_cross      # left nodes -> right nodes
        else:
            A_full[8:, :8] = A_cross      # right nodes -> left nodes

        return A_full                     # [NV, NV]


# ---------------------------------------------------------------------------
# DGL Unit
# ---------------------------------------------------------------------------

class DGLUnit(nn.Module):
    """
    Dynamic Graph Learning unit (paper Section III-C-1).

    Computes a series of dynamic adjacency matrices — one per time slot —
    by fusing five branch outputs with learned scalar weights.

    Post-processing per time slot (equations 6-8):
      1. Zero out entries below THRESHOLD
      2. Keep only top-k entries per row
      3. Zero out entries where the predefined graph A has no edge
         and remove self-loops

    Args:
        c_in   : number of input channels
        t_slot : number of frames per time slot (T // s)
        s      : number of time slots to produce
    """

    def __init__(self, c_in: int, t_slot: int, s: int):
        super().__init__()
        self.s      = s
        self.t_slot = t_slot

        # Five branches
        self.branch1 = GlobalBranch(c_in, t_slot)
        self.branch2 = LocalFootBranch(c_in, t_slot, foot="left")
        self.branch3 = CrossFootBranch(c_in, t_slot, direction="LR")
        self.branch4 = CrossFootBranch(c_in, t_slot, direction="RL")
        self.branch5 = LocalFootBranch(c_in, t_slot, foot="right")

        # Learnable fusion weights (one scalar per branch)
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5.0)

        # Register predefined adjacency as non-trainable buffer
        gc = get_graph_components()
        self.register_buffer("A_pre", gc["A"])   # [NV, NV]

    def _fuse_branches(self, outputs: list) -> torch.Tensor:
        """
        Fuses 5 branch adjacency matrices with softmax-normalised weights.

        Args:
            outputs: list of 5 tensors each [NV, NV]
        Returns:
            A_fused: [NV, NV]
        """
        w       = F.softmax(self.fusion_weights, dim=0)   # [5]
        stacked = torch.stack(outputs, dim=0)             # [5, NV, NV]
        return (w.view(5, 1, 1) * stacked).sum(dim=0)     # [NV, NV]

    def _postprocess(self, A_dy: torch.Tensor) -> torch.Tensor:
        """
        Applies equations (6), (7), (8):
          (6) Zero entries below threshold
          (7) Keep only top-k entries across matrix
          (8) Mask to predefined graph structure; remove self-loops
        """
        # Eq (6): threshold
        A_dy = A_dy * (A_dy >= THRESHOLD).float()

        # Eq (7): global top-k
        if A_dy.sum() > 0:
            flat     = A_dy.view(-1)
            k        = min(TOP_K * NV, flat.numel())
            topk_val = torch.topk(flat, k=k, largest=True).values[-1]
            A_dy     = A_dy * (A_dy >= topk_val).float()

        # Eq (8): enforce predefined structure and remove self-loops
        A_dy = A_dy * self.A_pre
        A_dy.fill_diagonal_(0.0)

        return A_dy

    def forward(self, node_data: torch.Tensor) -> list:
        """
        Args:
            node_data: [B, C_in, T, NV]

        Returns:
            adj_matrices: list of s tensors, each [NV, NV]
        """
        B, C, T, _ = node_data.shape
        t_slot      = T // self.s
        adj_matrices = []

        for k in range(self.s):
            t_start = k * t_slot
            t_end   = t_start + t_slot
            slot    = node_data[:, :, t_start:t_end, :]   # [B, C, t_slot, NV]

            b1 = self.branch1(slot)
            b2 = self.branch2(slot)
            b3 = self.branch3(slot)
            b4 = self.branch4(slot)
            b5 = self.branch5(slot)

            A_fused = self._fuse_branches([b1, b2, b3, b4, b5])
            A_fused = F.relu(A_fused)

            # Normalise to [0, 1]
            max_val = A_fused.max()
            if max_val > 0:
                A_fused = A_fused / max_val

            A_dy = self._postprocess(A_fused)
            adj_matrices.append(A_dy)

        return adj_matrices   # list of s tensors [NV, NV]


# ---------------------------------------------------------------------------
# Edge Generation Unit  (paper Section III-C-2)
# ---------------------------------------------------------------------------

class EdgeGenerationUnit(nn.Module):
    """
    Generates dynamic edge features from learned adjacency matrices.

    For each time slot k:
      1. Identify non-zero entries in A_dy(tk) -> dynamic edge set C(tk)
      2. Compute edge values: e(i,j) = v_dst - v_src  (equation 4)
      3. Build and normalise dynamic source/destination matrices
         S_dy(tk), D_dy(tk)  (equations 9-12)

    No trainable parameters — deterministic transform of adjacency + nodes.
    """

    def __init__(self, max_dynamic_edges: int = 26):
        """
        Args:
            max_dynamic_edges: padding size for NE_dy across slots.
                               Defaults to 26 to match predefined NE.
        """
        super().__init__()
        self.max_ne_dy = max_dynamic_edges

    def forward(
        self,
        node_data:    torch.Tensor,   # [B, C, T, NV]
        adj_matrices: list,           # list of s tensors [NV, NV]
    ):
        """
        Returns
        -------
        dy_edges_list : list of s tensors  [B, C, t_slot, max_ne_dy]
        S_dy_list     : list of s tensors  [NV, max_ne_dy]  normalised
        D_dy_list     : list of s tensors  [NV, max_ne_dy]  normalised
        """
        B, C, T, _ = node_data.shape
        s           = len(adj_matrices)
        t_slot      = T // s

        dy_edges_list = []
        S_dy_list     = []
        D_dy_list     = []

        for k, A_dy in enumerate(adj_matrices):
            t_start = k * t_slot
            t_end   = t_start + t_slot
            slot    = node_data[:, :, t_start:t_end, :]   # [B, C, t_slot, NV]

            edge_coords = A_dy.nonzero(as_tuple=False)    # [NE_dy, 2]
            NE_dy       = edge_coords.size(0)

            if NE_dy == 0:
                dy_edges_list.append(
                    torch.zeros(B, C, t_slot, self.max_ne_dy, device=node_data.device)
                )
                S_dy_list.append(
                    torch.zeros(NV, self.max_ne_dy, device=node_data.device)
                )
                D_dy_list.append(
                    torch.zeros(NV, self.max_ne_dy, device=node_data.device)
                )
                continue

            src_idx = edge_coords[:, 0]   # [NE_dy]
            dst_idx = edge_coords[:, 1]   # [NE_dy]

            # Equation (4)
            v_src = slot[..., src_idx]    # [B, C, t_slot, NE_dy]
            v_dst = slot[..., dst_idx]    # [B, C, t_slot, NE_dy]
            e_dy  = v_dst - v_src         # [B, C, t_slot, NE_dy]

            # Build S_dy and D_dy  (equations 9-10)
            S_dy = torch.zeros(NV, NE_dy, device=node_data.device)
            D_dy = torch.zeros(NV, NE_dy, device=node_data.device)
            for j in range(NE_dy):
                S_dy[src_idx[j], j] = 1.0
                D_dy[dst_idx[j], j] = 1.0

            # Normalise  (equations 11-12)
            S_dy_hat = normalize_matrix(S_dy)
            D_dy_hat = normalize_matrix(D_dy)

            # Pad or truncate to max_ne_dy
            pad = self.max_ne_dy - NE_dy
            if pad > 0:
                e_dy     = F.pad(e_dy,     (0, pad))
                S_dy_hat = F.pad(S_dy_hat, (0, pad))
                D_dy_hat = F.pad(D_dy_hat, (0, pad))
            elif pad < 0:
                e_dy     = e_dy[..., :self.max_ne_dy]
                S_dy_hat = S_dy_hat[:, :self.max_ne_dy]
                D_dy_hat = D_dy_hat[:, :self.max_ne_dy]

            dy_edges_list.append(e_dy)
            S_dy_list.append(S_dy_hat)
            D_dy_list.append(D_dy_hat)

        return dy_edges_list, S_dy_list, D_dy_list


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, C_in, T, s = 4, 1, 128, 4
    t_slot = T // s

    print("=" * 55)
    print("dgl_unit.py sanity check")
    print("=" * 55)

    x = torch.randn(B, C_in, T, NV)

    dgl      = DGLUnit(c_in=C_in, t_slot=t_slot, s=s)
    adj_list = dgl(x)

    print(f"  Input              : {x.shape}")
    print(f"  Time slots         : {len(adj_list)}")
    for k, A in enumerate(adj_list):
        nnz = (A > 0).sum().item()
        print(f"  Slot {k}: shape={A.shape}  non-zero={nnz}")

    egu = EdgeGenerationUnit(max_dynamic_edges=26)
    dy_edges, S_dy_list, D_dy_list = egu(x, adj_list)

    print(f"\n  Edge generation (slot 0):")
    print(f"    dy_edges : {dy_edges[0].shape}")
    print(f"    S_dy_hat : {S_dy_list[0].shape}")
    print(f"    D_dy_hat : {D_dy_list[0].shape}")

    # Gradient flow check
    loss = sum(A.sum() for A in adj_list)
    loss.backward()
    print(f"\n  fusion_weights.grad : {dgl.fusion_weights.grad}")
    print("\n  All checks passed.")

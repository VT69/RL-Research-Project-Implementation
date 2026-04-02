"""
tcn_unit.py
===========
Temporal Convolutional Network (TCN) unit — the third sub-module inside
every DyDGNN block (paper Section III-C-4, equations 17 & 18).

Purpose
-------
After DyDGN captures *spatial* relationships at each time slot, the TCN
unit captures *local temporal* patterns by connecting features of the same
node (or edge) across nearby frames within a sliding window of size Γ.

Sampling strategy (equations 17 & 18):
    N(v_{k,p}) = { v_{t,p} | t = k + l,  |l| ≤ Γ/2,  l ∈ ℤ }
    N(e_{k,q}) = { e_{t,q} | t = k + l,  |l| ≤ Γ/2,  l ∈ ℤ }

This is exactly a 1-D causal convolution along the time axis with
kernel size Γ, applied independently to each node / edge channel.

Key design decisions
--------------------
* Nodes and edges have SEPARATE TCN parameters (distinct weight tensors),
  allowing the model to learn different temporal dynamics for pressure
  values (nodes) vs pressure differences (edges).
* Dilated convolutions are stacked to exponentially grow the receptive
  field without increasing parameters (standard TCN design from Bai et al.
  "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
  for Sequence Modeling", 2018).
* Residual connections stabilise gradient flow across multiple DyDGNN
  blocks (4 blocks × multiple TCN layers can be deep).
* The time dimension is treated as the sequence axis; NV / NE are treated
  as independent "batch" channels so a single Conv1d handles all nodes or
  edges in one pass.

Architecture per TCN unit
--------------------------
  Input [B, C, T, NV]
    → reshape to [B*NV, C, T]           (treat each node as a sequence)
    → TCN stack (dilated Conv1d + BN + ReLU + residual) × n_layers
    → reshape back to [B, C_out, T, NV]

  Same for edges: [B, C, T, NE] → [B*NE, C, T] → ... → [B, C_out, T, NE]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Single dilated residual TCN block
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """
    One dilated causal convolutional residual block.

    Conv1d(kernel=Γ, dilation=d, padding=causal) → BN → ReLU
    Conv1d(kernel=Γ, dilation=d, padding=causal) → BN → ReLU
    + residual (1×1 conv if C_in ≠ C_out)

    Causal padding = (Γ - 1) * d  ensures the output at position t
    only depends on positions ≤ t (no future leakage).
    """

    def __init__(
        self,
        C_in:     int,
        C_out:    int,
        kernel:   int = 3,
        dilation: int = 1,
        dropout:  float = 0.1,
    ):
        super().__init__()
        pad = (kernel - 1) * dilation   # causal left-padding

        self.conv1 = nn.Conv1d(
            C_in, C_out, kernel_size=kernel,
            dilation=dilation, padding=pad, bias=False
        )
        self.bn1   = nn.BatchNorm1d(C_out)

        self.conv2 = nn.Conv1d(
            C_out, C_out, kernel_size=kernel,
            dilation=dilation, padding=pad, bias=False
        )
        self.bn2   = nn.BatchNorm1d(C_out)

        self.drop  = nn.Dropout(dropout)
        self.pad   = pad

        # Residual projection if channel dims differ
        self.residual = (
            nn.Conv1d(C_in, C_out, kernel_size=1, bias=False)
            if C_in != C_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C_in, T]   (N = B*NV or B*NE)
        Returns:
            [N, C_out, T]
        """
        # First conv + trim causal padding
        out = self.conv1(x)
        out = out[..., : x.shape[-1]]    # remove right-side causal padding
        out = F.relu(self.bn1(out))
        out = self.drop(out)

        # Second conv + trim
        out = self.conv2(out)
        out = out[..., : x.shape[-1]]
        out = F.relu(self.bn2(out))
        out = self.drop(out)

        # Residual
        res = self.residual(x)
        return F.relu(out + res)


# ---------------------------------------------------------------------------
# TCN stack for one modality (nodes OR edges)
# ---------------------------------------------------------------------------

class _TCNStack(nn.Module):
    """
    Stack of dilated TCN blocks with exponentially increasing dilation.

    Receptive field = 1 + 2 * (kernel-1) * (2^n_layers - 1)
    For kernel=3, n_layers=3: RF = 1 + 2*2*(8-1) = 29 frames

    Args:
        C_in      : input channels
        C_out     : output channels (same for all layers)
        kernel    : temporal kernel size Γ (paper uses odd integers)
        n_layers  : number of stacked TCN blocks
        dropout   : dropout probability
    """

    def __init__(
        self,
        C_in:     int,
        C_out:    int,
        kernel:   int   = 3,
        n_layers: int   = 3,
        dropout:  float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i           # 1, 2, 4, 8, ...
            in_ch    = C_in if i == 0 else C_out
            layers.append(
                _TCNBlock(in_ch, C_out, kernel=kernel,
                          dilation=dilation, dropout=dropout)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C_in, T]
        Returns:
            [N, C_out, T]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Full TCN unit  (nodes + edges, separate parameters)
# ---------------------------------------------------------------------------

class TCNUnit(nn.Module):
    """
    TCN unit as used inside each DyDGNN block.

    Applies independent temporal convolution stacks to node features
    and edge features, preserving their spatial dimensions (NV, NE).

    Args:
        C_in     : input channels
        C_out    : output channels
        NV       : number of sensor nodes
        NE       : number of predefined edges
        kernel   : temporal kernel size Γ (paper Section III-C-4)
        n_layers : TCN depth (number of dilated blocks)
        dropout  : dropout rate
    """

    def __init__(
        self,
        C_in:     int,
        C_out:    int,
        NV:       int   = 16,
        NE:       int   = 26,
        kernel:   int   = 3,
        n_layers: int   = 3,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.NV = NV
        self.NE = NE

        # Separate stacks for nodes and edges (distinct learned parameters)
        self.node_tcn = _TCNStack(C_in, C_out, kernel, n_layers, dropout)
        self.edge_tcn = _TCNStack(C_in, C_out, kernel, n_layers, dropout)

    def forward(
        self,
        V: torch.Tensor,    # [B, C_in, T, NV]
        E: torch.Tensor,    # [B, C_in, T, NE]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            V: node features  [B, C_in, T, NV]
            E: edge features  [B, C_in, T, NE]

        Returns:
            V_out: [B, C_out, T, NV]
            E_out: [B, C_out, T, NE]
        """
        B, C, T, NV = V.shape
        NE = E.shape[-1]

        # ---- Node TCN ----
        # [B, C, T, NV] → [B, NV, C, T] → [B*NV, C, T]
        V_seq = V.permute(0, 3, 1, 2).reshape(B * NV, C, T)
        V_seq = self.node_tcn(V_seq)                 # [B*NV, C_out, T]
        C_out = V_seq.shape[1]
        V_out = V_seq.reshape(B, NV, C_out, T)
        V_out = V_out.permute(0, 2, 3, 1)            # [B, C_out, T, NV]

        # ---- Edge TCN ----
        # [B, C, T, NE] → [B*NE, C, T]
        E_seq = E.permute(0, 3, 1, 2).reshape(B * NE, C, T)
        E_seq = self.edge_tcn(E_seq)                 # [B*NE, C_out, T]
        E_out = E_seq.reshape(B, NE, C_out, T)
        E_out = E_out.permute(0, 2, 3, 1)            # [B, C_out, T, NE]

        return V_out, E_out


# ---------------------------------------------------------------------------
# Receptive field utility
# ---------------------------------------------------------------------------

def receptive_field(kernel: int, n_layers: int) -> int:
    """
    Computes the temporal receptive field of a TCN stack.

    RF = 1 + 2 * (kernel - 1) * sum(2^i for i in range(n_layers))
       = 1 + 2 * (kernel - 1) * (2^n_layers - 1)

    Example: kernel=3, n_layers=3 → RF = 29 frames
    """
    return 1 + 2 * (kernel - 1) * (2 ** n_layers - 1)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    configs = [
        # (C_in, C_out, T,   NV, NE, kernel, n_layers, label)
        (1,  32,  128, 16, 26, 3, 3, "Block 1 (C_in=1  → C_out=32)"),
        (32, 32,  128, 16, 26, 3, 3, "Block 2 (C_in=32 → C_out=32)"),
        (32, 32,  128, 16, 26, 3, 3, "Block 3 (C_in=32 → C_out=32)"),
        (32, 64,  128, 16, 26, 3, 3, "Block 4 (C_in=32 → C_out=64)"),
    ]

    print("=" * 60)
    print("tcn_unit.py sanity check")
    print("=" * 60)

    for C_in, C_out, T, NV, NE, kern, nlayers, label in configs:
        B = 4
        V = torch.randn(B, C_in, T, NV)
        E = torch.randn(B, C_in, T, NE)

        tcn = TCNUnit(C_in, C_out, NV=NV, NE=NE,
                      kernel=kern, n_layers=nlayers)
        V_out, E_out = tcn(V, E)

        assert V_out.shape == (B, C_out, T, NV), \
            f"Node shape mismatch: {V_out.shape}"
        assert E_out.shape == (B, C_out, T, NE), \
            f"Edge shape mismatch: {E_out.shape}"

        rf = receptive_field(kern, nlayers)

        # Count parameters
        node_params = sum(p.numel() for p in tcn.node_tcn.parameters())
        edge_params = sum(p.numel() for p in tcn.edge_tcn.parameters())

        print(f"\n  {label}")
        print(f"    V: {V.shape} → {V_out.shape}")
        print(f"    E: {E.shape} → {E_out.shape}")
        print(f"    Receptive field : {rf} frames")
        print(f"    Node TCN params : {node_params:,}")
        print(f"    Edge TCN params : {edge_params:,}")

    # Gradient flow check
    print("\n  Gradient flow check ...")
    B, C_in, C_out, T, NV, NE = 4, 32, 32, 128, 16, 26
    V = torch.randn(B, C_in, T, NV, requires_grad=True)
    E = torch.randn(B, C_in, T, NE, requires_grad=True)
    tcn = TCNUnit(C_in, C_out, NV=NV, NE=NE)
    V_out, E_out = tcn(V, E)
    loss = V_out.sum() + E_out.sum()
    loss.backward()
    print(f"    V.grad mean abs : {V.grad.abs().mean():.6f}")
    print(f"    E.grad mean abs : {E.grad.abs().mean():.6f}")

    # Receptive field table
    print("\n  Receptive field vs kernel / depth:")
    print(f"  {'kernel':>8}  {'n_layers':>10}  {'RF (frames)':>12}")
    for k in [3, 5, 7]:
        for n in [2, 3, 4]:
            print(f"  {k:>8}  {n:>10}  {receptive_field(k, n):>12}")

    print("\n  All assertions passed.")

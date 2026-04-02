"""
model.py
========
Full GLD²-GNN model (paper Section III, Fig. 2).

Assembly
--------
  Two-stream input
    ├── Time stream   [B, 1, T, NV]  +  [B, 1, T, NE]
    └── Motion stream [B, 1, T, NV]  +  [B, 1, T, NE]

  Each stream passes through N=4 DyDGNN blocks in sequence:

    DyDGNN block k
    ├── DGLUnit         → s dynamic adjacency matrices
    ├── EdgeGenUnit     → dynamic S^dy, D^dy, E^dy per slot
    ├── DyDGNUnit       → updated V [B, C_out, T, NV]
    │                      updated E [B, C_out, T, NE]
    └── TCNUnit         → V [B, C_out, T, NV]
                          E [B, C_out, T, NE]

  Channel schedule (paper Section IV-A-2):
    Block 1: C_in=1  → C_out=32,  s=4
    Block 2: C_in=32 → C_out=32,  s=8
    Block 3: C_in=32 → C_out=32,  s=8
    Block 4: C_in=32 → C_out=64,  s=8

  Classification head (per stream):
    V: [B, 64, T, NV] → AdaptiveAvgPool over (T, NV) → [B, 64]
    E: [B, 64, T, NE] → AdaptiveAvgPool over (T, NE) → [B, 64]
    concat → [B, 128] → Linear(128, 1) → Sigmoid → p^Ts_c / p^Ms_c

  Two-stream fusion (equation 19):
    p_c = α · p^Ts_c + (1 - α) · p^Ms_c

Usage
-----
  from model import GLD2GNN

  model = GLD2GNN(T=128)
  logits = model(ts_nodes, ts_edges, ms_nodes, ms_edges)
  # logits: [B, 1]  probability of PD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from graph_construction import (
    get_graph_components,
    NV as NV_DEFAULT,
)
from dgl_unit    import DGLUnit, EdgeGenerationUnit
from dydgn_unit  import DyDGNUnit
from tcn_unit    import TCNUnit


# ---------------------------------------------------------------------------
# Single DyDGNN block
# ---------------------------------------------------------------------------

class DyDGNNBlock(nn.Module):
    """
    One complete DyDGNN block:  DGL → EdgeGen → DyDGN → TCN.

    Args:
        C_in            : input channels
        C_out           : output channels
        T               : temporal length of input
        s               : number of time slots for DGL + DyDGN
        NV              : sensor nodes
        NE              : predefined edges
        topk            : top-k edges in DGLUnit
        thresh          : threshold in DGLUnit
        tcn_kernel      : TCN temporal kernel size
        tcn_layers      : TCN depth
        dropout         : dropout in TCN
        trainable_after : warm-up epochs before adaptive matrices unfreeze
    """

    def __init__(
        self,
        C_in:            int,
        C_out:           int,
        T:               int,
        s:               int,
        NV:              int   = NV_DEFAULT,
        NE:              int   = 26,
        topk:            int   = 6,
        thresh:          float = 0.1,
        tcn_kernel:      int   = 3,
        tcn_layers:      int   = 3,
        dropout:         float = 0.1,
        trainable_after: int   = 5,
    ):
        super().__init__()
        assert T % s == 0, f"T={T} must be divisible by s={s}"

        t_slot = T // s
        self.dgl      = DGLUnit(
            c_in=C_in, t_slot=t_slot, s=s,
        )
        self.edge_gen = EdgeGenerationUnit(max_dynamic_edges=NE)
        self.dydgn    = DyDGNUnit(
            C_in=C_in, C_out=C_out,
            NV=NV, NE=NE, s=s,
            trainable_after=trainable_after,
        )
        self.tcn      = TCNUnit(
            C_in=C_out, C_out=C_out,
            NV=NV, NE=NE,
            kernel=tcn_kernel, n_layers=tcn_layers,
            dropout=dropout,
        )

    def step(self):
        """Advance adaptive-matrix warm-up counter (call once per epoch)."""
        self.dydgn.step()

    def forward(
        self,
        V:        torch.Tensor,   # [B, C_in, T, NV]
        E:        torch.Tensor,   # [B, C_in, T, NE]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            V_out : [B, C_out, T, NV]
            E_out : [B, C_out, T, NE]
        """
        # 1. Dynamic graph learning → s adjacency matrices
        adj_list = self.dgl(V)

        # 2. Edge generation → dynamic incidence + edge features
        # EdgeGenerationUnit.forward(node_data, adj_matrices)
        E_dy_list, S_dy_list, D_dy_list = self.edge_gen(V, adj_list)

        # 3. Spatial feature aggregation
        V_out, E_out = self.dydgn(V, E, S_dy_list, D_dy_list, E_dy_list)

        # 4. Temporal feature extraction
        V_out, E_out = self.tcn(V_out, E_out)

        return V_out, E_out


# ---------------------------------------------------------------------------
# Single-stream feature extractor (4 stacked DyDGNN blocks)
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    """
    Processes one stream (time or motion) through 4 DyDGNN blocks and
    returns a fixed-length feature vector per sample.

    Channel schedule from paper Section IV-A-2:
        Block 1 : C_in=1  → C_out=32, s=4
        Block 2 : C_in=32 → C_out=32, s=8
        Block 3 : C_in=32 → C_out=32, s=8
        Block 4 : C_in=32 → C_out=64, s=8

    Classification head:
        V_feat + E_feat concatenated → Linear(128, 1) → Sigmoid
    """

    # Channel / time-slot schedule matching the paper
    BLOCK_CONFIGS = [
        # (C_in, C_out, s)
        (1,  32, 4),
        (32, 32, 8),
        (32, 32, 8),
        (32, 64, 8),
    ]

    def __init__(
        self,
        T:               int,
        NV:              int   = NV_DEFAULT,
        NE:              int   = 26,
        topk:            int   = 6,
        thresh:          float = 0.1,
        tcn_kernel:      int   = 3,
        tcn_layers:      int   = 3,
        dropout:         float = 0.1,
        trainable_after: int   = 5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for C_in, C_out, s in self.BLOCK_CONFIGS:
            assert T % s == 0, (
                f"T={T} must be divisible by s={s} for all blocks. "
                f"Choose T that is divisible by 8 (e.g. 128, 256)."
            )
            self.blocks.append(DyDGNNBlock(
                C_in=C_in, C_out=C_out, T=T, s=s,
                NV=NV, NE=NE,
                topk=topk, thresh=thresh,
                tcn_kernel=tcn_kernel, tcn_layers=tcn_layers,
                dropout=dropout,
                trainable_after=trainable_after,
            ))

        # Final channel dimension after block 4
        C_final = self.BLOCK_CONFIGS[-1][1]   # 64

        # Global average pooling collapses (T, NV) and (T, NE) to scalars
        self.node_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, T, NV] → [B,64,1,1]
        self.edge_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, T, NE] → [B,64,1,1]

        # Classification head: concat node + edge features
        self.classifier = nn.Sequential(
            nn.Linear(C_final * 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def step(self):
        """Propagate warm-up step to all blocks."""
        for block in self.blocks:
            block.step()

    def forward(
        self,
        V:        torch.Tensor,   # [B, 1, T, NV]
        E:        torch.Tensor,   # [B, 1, T, NE]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            V : node features  [B, 1, T, NV]
            E : edge features  [B, 1, T, NE]

        Returns:
            prob : [B, 1]  per-sample PD probability
            feat : [B, 128]  pooled feature vector (for analysis / fusion)
        """
        for block in self.blocks:
            V, E = block(V, E)

        # Pool spatial + temporal dims
        V_feat = self.node_pool(V).flatten(1)   # [B, 64]
        E_feat = self.edge_pool(E).flatten(1)   # [B, 64]
        feat   = torch.cat([V_feat, E_feat], dim=1)   # [B, 128]

        prob   = self.classifier(feat)           # [B, 1]
        return prob, feat


# ---------------------------------------------------------------------------
# Full GLD²-GNN model
# ---------------------------------------------------------------------------

class GLD2GNN(nn.Module):
    """
    Global-Local Dynamic Directed Graph Neural Network (GLD²-GNN).

    Two independent StreamEncoders share the same architecture but have
    separate weights:
        • time_encoder   processes the time stream
        • motion_encoder processes the motion stream

    Final prediction (equation 19):
        p_c = α · p^Ts_c + (1 - α) · p^Ms_c

    where α is a learnable scalar initialised to 0.5.

    Args:
        T               : gait-cycle length (must be divisible by 8)
        NV              : sensor nodes (default 16)
        NE              : predefined edges (default 26)
        topk            : DGL top-k edges per node
        thresh          : DGL adjacency threshold
        tcn_kernel      : TCN temporal kernel size
        tcn_layers      : TCN depth per block
        dropout         : dropout in TCN blocks
        trainable_after : warm-up epochs before adaptive matrices unfreeze
        device          : torch device for static graph tensors
    """

    def __init__(
        self,
        T:               int   = 128,
        NV:              int   = NV_DEFAULT,
        NE:              int   = 26,
        topk:            int   = 6,
        thresh:          float = 0.1,
        tcn_kernel:      int   = 3,
        tcn_layers:      int   = 3,
        dropout:         float = 0.1,
        trainable_after: int   = 5,
        device:          torch.device = torch.device("cpu"),
    ):
        super().__init__()

        encoder_kwargs = dict(
            T=T, NV=NV, NE=NE,
            topk=topk, thresh=thresh,
            tcn_kernel=tcn_kernel, tcn_layers=tcn_layers,
            dropout=dropout,
            trainable_after=trainable_after,
        )

        self.time_encoder   = StreamEncoder(**encoder_kwargs)
        self.motion_encoder = StreamEncoder(**encoder_kwargs)

        # Learnable fusion weight α (equation 19); sigmoid keeps it in (0,1)
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

        # Register static graph tensors as buffers (moved with .to(device))
        gc = get_graph_components(device)
        self.register_buffer("A_static", gc["A"])    # [NV, NV]

    @property
    def alpha(self) -> torch.Tensor:
        """Fusion weight α ∈ (0, 1)."""
        return torch.sigmoid(self.alpha_logit)

    def step(self):
        """
        Advance the adaptive-matrix warm-up counter.
        Call once at the END of each training epoch.
        """
        self.time_encoder.step()
        self.motion_encoder.step()

    def forward(
        self,
        ts_nodes: torch.Tensor,   # [B, 1, T, NV]  time-stream nodes
        ts_edges: torch.Tensor,   # [B, 1, T, NE]  time-stream edges
        ms_nodes: torch.Tensor,   # [B, 1, T, NV]  motion-stream nodes
        ms_edges: torch.Tensor,   # [B, 1, T, NE]  motion-stream edges
    ) -> torch.Tensor:
        """
        Args:
            ts_nodes : [B, 1, T, NV]
            ts_edges : [B, 1, T, NE]
            ms_nodes : [B, 1, T, NV]
            ms_edges : [B, 1, T, NE]

        Returns:
            p_c : [B, 1]  fused PD probability  (threshold at 0.5 to classify)
        """
        # Time stream
        p_ts, _ = self.time_encoder(ts_nodes, ts_edges)      # [B, 1]

        # Motion stream
        p_ms, _ = self.motion_encoder(ms_nodes, ms_edges)    # [B, 1]

        # α-weighted fusion (equation 19)
        alpha = self.alpha
        p_c   = alpha * p_ts + (1.0 - alpha) * p_ms          # [B, 1]

        return p_c

    def predict(
        self,
        ts_nodes: torch.Tensor,
        ts_edges: torch.Tensor,
        ms_nodes: torch.Tensor,
        ms_edges: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Convenience method: returns hard class labels (0=CO, 1=PD).

        Args:
            threshold : decision boundary (default 0.5)

        Returns:
            labels : [B]  LongTensor
        """
        with torch.no_grad():
            prob = self.forward(ts_nodes, ts_edges, ms_nodes, ms_edges)
        return (prob.squeeze(1) >= threshold).long()

    def count_parameters(self) -> dict:
        """Returns parameter counts broken down by component."""
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        total = count(self)
        ts    = count(self.time_encoder)
        ms    = count(self.motion_encoder)

        ts_dgl   = sum(count(b.dgl)   for b in self.time_encoder.blocks)
        ts_dydgn = sum(count(b.dydgn) for b in self.time_encoder.blocks)
        ts_tcn   = sum(count(b.tcn)   for b in self.time_encoder.blocks)

        return {
            "total":             total,
            "time_encoder":      ts,
            "motion_encoder":    ms,
            "time_dgl_units":    ts_dgl,
            "time_dydgn_units":  ts_dydgn,
            "time_tcn_units":    ts_tcn,
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, NV, NE = 4, 128, 16, 26

    print("=" * 60)
    print("model.py sanity check")
    print("=" * 60)

    model = GLD2GNN(T=T, device=device).to(device)

    # Dummy two-stream inputs
    ts_nodes = torch.randn(B, 1, T, NV, device=device)
    ts_edges = torch.randn(B, 1, T, NE, device=device)
    ms_nodes = torch.randn(B, 1, T, NV, device=device)
    ms_edges = torch.randn(B, 1, T, NE, device=device)

    # Forward pass
    p_c = model(ts_nodes, ts_edges, ms_nodes, ms_edges)
    print(f"\n  Output shape    : {p_c.shape}  (expected [4, 1])")
    print(f"  Output range    : [{p_c.min().item():.4f}, {p_c.max().item():.4f}]")
    assert p_c.shape == (B, 1), f"Shape mismatch: {p_c.shape}"
    assert (p_c >= 0).all() and (p_c <= 1).all(), "Probabilities out of [0,1]"

    # Predict
    labels = model.predict(ts_nodes, ts_edges, ms_nodes, ms_edges)
    print(f"  Predicted labels: {labels.tolist()}")
    assert labels.shape == (B,)

    # Alpha
    print(f"  Fusion alpha    : {model.alpha.item():.4f}  (init should be ~0.5)")

    # Parameter counts
    counts = model.count_parameters()
    print(f"\n  Parameter breakdown:")
    for k, v in counts.items():
        print(f"    {k:<22} : {v:>10,}")

    # Gradient flow
    loss = F.binary_cross_entropy(p_c, torch.zeros(B, 1, device=device))
    loss.backward()
    print(f"\n  Loss            : {loss.item():.6f}")
    print(f"  alpha_logit.grad: {model.alpha_logit.grad.item():.6f}")

    # Warm-up step
    print(f"\n  Before step(): S_hat_param trainable = "
          f"{model.time_encoder.blocks[0].dydgn.S_hat_param.requires_grad}")
    for _ in range(5):
        model.step()
    print(f"  After  step(): S_hat_param trainable = "
          f"{model.time_encoder.blocks[0].dydgn.S_hat_param.requires_grad}")

    # DyDGNN block-by-block shape trace (time stream)
    print(f"\n  Block-by-block shape trace (time stream):")
    V = ts_nodes.clone()
    E = ts_edges.clone()
    # Note: DyDGNNBlock.forward(V, E) uses the static graph held internally
    # by DGLUnit as a registered buffer — no need to pass A_static explicitly.
    for i, block in enumerate(model.time_encoder.blocks):
        V, E = block(V, E)
        cfg  = StreamEncoder.BLOCK_CONFIGS[i]
        print(f"    Block {i+1} (C_in={cfg[0]:>2}, C_out={cfg[1]:>2}, s={cfg[2]}): "
              f"V={tuple(V.shape)}  E={tuple(E.shape)}")

    print("\n  All assertions passed.")

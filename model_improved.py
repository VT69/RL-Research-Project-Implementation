"""
model_improved.py
=================
GLD²-GNN+ : Three targeted improvements over the baseline (Wang et al., 2025)

Research gaps addressed
-----------------------
GAP 1 — Fixed scalar α fusion (paper eq. 19)
    The paper uses one global learned scalar α shared across ALL patients.
    This ignores that the relative importance of time vs. motion stream
    varies per subject and per gait phase (early-stage PD differs from
    late-stage PD in which stream is more discriminative).

    FIX: AdaptiveFusion — a small MLP that reads the concatenated
    [time_feat, motion_feat] vector and predicts a SAMPLE-WISE α ∈ (0,1).
    Each sample gets its own fusion weight at inference time.

GAP 2 — No uncertainty estimation
    The baseline outputs a hard probability with no confidence measure.
    Borderline cases (e.g. p=0.58) look the same as clear cases (p=0.97).
    Clinically, knowing "I am uncertain" is as important as the prediction.

    FIX: MC Dropout — keep dropout active at test time, run N stochastic
    forward passes, report mean prediction + std as calibrated uncertainty.
    Zero extra training cost; adds interpretable confidence to every output.

GAP 3 — Unconstrained dynamic graph structure
    The DGL unit is free to learn any edge weights within the predefined
    mask, but nothing encourages sparse, physically meaningful graphs.
    Dense learned graphs are harder to interpret and may overfit.

    FIX: Graph Sparsity Regularisation — add an L1 penalty on the learned
    adjacency matrices to the training loss, encouraging the model to use
    only the most informative pressure-transmission edges.

Usage
-----
    from model_improved import GLD2GNNPlus

    model = GLD2GNNPlus(T=128)

    # Training (returns loss components separately for logging)
    out = model(ts_nodes, ts_edges, ms_nodes, ms_edges)
    loss = out['bce_loss'] + out['sparsity_loss']

    # Inference with uncertainty (MC Dropout, N=30 passes)
    result = model.predict_with_uncertainty(
        ts_nodes, ts_edges, ms_nodes, ms_edges, n_passes=30
    )
    # result['mean']  : [B]  mean prediction
    # result['std']   : [B]  uncertainty (higher = less confident)
    # result['label'] : [B]  hard label at threshold=0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from graph_construction import get_graph_components, NV as NV_DEFAULT
from dgl_unit   import DGLUnit, EdgeGenerationUnit
from dydgn_unit import DyDGNUnit
from tcn_unit   import TCNUnit
from model      import DyDGNNBlock, StreamEncoder   # reuse unchanged blocks


# ---------------------------------------------------------------------------
# GAP 1 FIX: Adaptive sample-wise fusion
# ---------------------------------------------------------------------------

class AdaptiveFusion(nn.Module):
    """
    Replaces the single global α scalar with a per-sample MLP.

    The MLP reads the concatenated pooled features from both streams
    and predicts an α ∈ (0,1) for EACH sample independently.

    Architecture:  [2*feat_dim] → Linear(64) → ReLU → Linear(1) → Sigmoid

    During training the MLP learns which feature patterns indicate
    "time stream is more reliable" vs "motion stream is more reliable",
    adapting the fusion dynamically rather than averaging globally.
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_ts: torch.Tensor,   # [B, feat_dim]
        feat_ms: torch.Tensor,   # [B, feat_dim]
    ) -> torch.Tensor:
        """Returns per-sample α ∈ (0,1)  shape [B, 1]."""
        combined = torch.cat([feat_ts, feat_ms], dim=1)   # [B, 2*feat_dim]
        return self.mlp(combined)                          # [B, 1]


# ---------------------------------------------------------------------------
# GAP 2 FIX: MC Dropout wrapper
# ---------------------------------------------------------------------------

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout — dropout that stays ACTIVE at inference time.

    Standard nn.Dropout is disabled during model.eval().
    This subclass always applies dropout regardless of training mode,
    enabling uncertainty estimation via multiple stochastic forward passes.
    """

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True forces dropout even in eval mode
        return F.dropout(x, p=self.p, training=True)


# ---------------------------------------------------------------------------
# GAP 3 FIX: Sparsity-regularised stream encoder
# ---------------------------------------------------------------------------

class RegularisedStreamEncoder(nn.Module):
    """
    StreamEncoder extended to collect adjacency matrices for sparsity loss.

    During the forward pass, adjacency matrices from all DGL units across
    all 4 blocks are stored so the training loop can compute the L1
    sparsity penalty:

        L_sparsity = λ * mean( |A_dyn| )   summed over all slots and blocks

    This encourages the learned dynamic graphs to use only the most
    informative edges, making the model more interpretable and reducing
    overfitting on small datasets (93+72 subjects is small for GNNs).
    """

    BLOCK_CONFIGS = [
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
        dropout:         float = 0.3,    # raised from 0.1 for MC Dropout
        trainable_after: int   = 5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for C_in, C_out, s in self.BLOCK_CONFIGS:
            self.blocks.append(DyDGNNBlock(
                C_in=C_in, C_out=C_out, T=T, s=s,
                NV=NV, NE=NE,
                topk=topk, thresh=thresh,
                tcn_kernel=tcn_kernel, tcn_layers=tcn_layers,
                dropout=dropout,
                trainable_after=trainable_after,
            ))

        C_final = self.BLOCK_CONFIGS[-1][1]   # 64
        self.node_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.edge_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MC Dropout before classifier
        self.mc_drop = MCDropout(p=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(C_final * 2, 1, bias=True),
            nn.Sigmoid(),
        )

        # Storage for adjacency matrices (populated during forward)
        self.last_adj_list: list = []

    def step(self):
        for block in self.blocks:
            block.step()

    def forward(
        self,
        V:        torch.Tensor,
        E:        torch.Tensor,
        A_static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Returns:
            prob     : [B, 1]   prediction probability
            feat     : [B, 128] pooled feature vector
            adj_list : list of all dynamic adjacency tensors (for sparsity loss)
        """
        self.last_adj_list = []

        for block in self.blocks:
            # Intercept adjacency matrices from DGL unit
            adj = block.dgl(V, A_static)
            self.last_adj_list.extend(adj)

            # Now run the full block (DGL runs again internally — small overhead)
            V, E = block(V, E, A_static)

        V_feat = self.node_pool(V).flatten(1)
        E_feat = self.edge_pool(E).flatten(1)
        feat   = torch.cat([V_feat, E_feat], dim=1)   # [B, 128]

        # MC Dropout applied to features before classification
        feat_dropped = self.mc_drop(feat)
        prob         = self.classifier(feat_dropped)   # [B, 1]

        return prob, feat, self.last_adj_list


# ---------------------------------------------------------------------------
# Full improved model: GLD²-GNN+
# ---------------------------------------------------------------------------

class GLD2GNNPlus(nn.Module):
    """
    GLD²-GNN+ — baseline with all three research gap fixes applied.

    Changes vs. baseline GLD2GNN
    -----------------------------
    1. AdaptiveFusion   replaces scalar alpha_logit
    2. MCDropout        applied to features in each StreamEncoder
    3. Sparsity loss    collected from all DGL units, returned for training

    Forward output
    --------------
    Returns a dict (not a raw tensor) to cleanly separate loss components:
        {
          'prob'          : [B, 1]  fused PD probability
          'bce_loss'      : scalar  binary cross-entropy (if labels given)
          'sparsity_loss' : scalar  L1 on dynamic adjacency matrices
          'alpha'         : [B, 1]  per-sample fusion weights
        }

    For inference without labels, pass labels=None — bce_loss will be 0.
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
        dropout:         float = 0.3,
        trainable_after: int   = 5,
        sparsity_lambda: float = 1e-4,   # λ for L1 graph regularisation
        device:          torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda

        encoder_kwargs = dict(
            T=T, NV=NV, NE=NE,
            topk=topk, thresh=thresh,
            tcn_kernel=tcn_kernel, tcn_layers=tcn_layers,
            dropout=dropout,
            trainable_after=trainable_after,
        )

        # Two independent regularised stream encoders
        self.time_encoder   = RegularisedStreamEncoder(**encoder_kwargs)
        self.motion_encoder = RegularisedStreamEncoder(**encoder_kwargs)

        # GAP 1: adaptive per-sample fusion (feat_dim = 128 from each encoder)
        self.adaptive_fusion = AdaptiveFusion(feat_dim=128)

        # Static graph buffers
        gc = get_graph_components(device)
        self.register_buffer("A_static", gc["A"])

    def step(self):
        """Advance warm-up counter — call once per epoch."""
        self.time_encoder.step()
        self.motion_encoder.step()

    def _sparsity_loss(self, adj_list: list) -> torch.Tensor:
        """
        GAP 3: L1 sparsity penalty on all learned dynamic adjacency matrices.
        Encourages the model to use fewer, more meaningful edges.
        """
        if not adj_list:
            return torch.tensor(0.0, requires_grad=True)
        total = sum(A.abs().mean() for A in adj_list)
        return self.sparsity_lambda * total / len(adj_list)

    def forward(
        self,
        ts_nodes: torch.Tensor,             # [B, 1, T, NV]
        ts_edges: torch.Tensor,             # [B, 1, T, NE]
        ms_nodes: torch.Tensor,             # [B, 1, T, NV]
        ms_edges: torch.Tensor,             # [B, 1, T, NE]
        labels:   Optional[torch.Tensor] = None,  # [B, 1] float, or None
    ) -> Dict[str, torch.Tensor]:
        A = self.A_static

        # Stream encoders
        p_ts, feat_ts, adj_ts = self.time_encoder(ts_nodes, ts_edges, A)
        p_ms, feat_ms, adj_ms = self.motion_encoder(ms_nodes, ms_edges, A)

        # GAP 1: sample-wise adaptive fusion
        alpha = self.adaptive_fusion(feat_ts, feat_ms)     # [B, 1]
        p_c   = alpha * p_ts + (1.0 - alpha) * p_ms       # [B, 1]

        # GAP 3: sparsity loss over all adjacency matrices
        all_adj = adj_ts + adj_ms
        sp_loss = self._sparsity_loss(all_adj)

        # BCE loss (only if labels provided)
        if labels is not None:
            bce_loss = F.binary_cross_entropy(p_c, labels)
        else:
            bce_loss = torch.tensor(0.0, device=p_c.device)

        return {
            "prob":          p_c,
            "bce_loss":      bce_loss,
            "sparsity_loss": sp_loss,
            "alpha":         alpha,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        ts_nodes:  torch.Tensor,
        ts_edges:  torch.Tensor,
        ms_nodes:  torch.Tensor,
        ms_edges:  torch.Tensor,
        n_passes:  int   = 30,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        GAP 2: MC Dropout inference — N stochastic forward passes.

        The model stays in eval() mode for BN layers but MCDropout
        keeps dropout active, producing N different predictions.
        Mean and std across passes give a calibrated uncertainty estimate.

        Args:
            n_passes  : number of stochastic forward passes (30 is standard)
            threshold : decision boundary for hard labels

        Returns dict with keys:
            mean   : [B]  average predicted probability across passes
            std    : [B]  standard deviation (uncertainty)
            label  : [B]  hard prediction  (mean >= threshold)
            alpha  : [B]  mean adaptive fusion weight
            passes : [B, n_passes]  all raw pass probabilities
        """
        self.eval()   # BN uses running stats; MCDropout stays active

        all_probs  = []
        all_alphas = []

        for _ in range(n_passes):
            out = self.forward(ts_nodes, ts_edges, ms_nodes, ms_edges)
            all_probs.append(out["prob"].squeeze(1))    # [B]
            all_alphas.append(out["alpha"].squeeze(1))  # [B]

        passes     = torch.stack(all_probs,  dim=1)   # [B, n_passes]
        alpha_runs = torch.stack(all_alphas, dim=1)   # [B, n_passes]

        mean  = passes.mean(dim=1)       # [B]
        std   = passes.std(dim=1)        # [B]  uncertainty
        label = (mean >= threshold).long()

        return {
            "mean":   mean,
            "std":    std,
            "label":  label,
            "alpha":  alpha_runs.mean(dim=1),
            "passes": passes,
        }

    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "total":            count(self),
            "time_encoder":     count(self.time_encoder),
            "motion_encoder":   count(self.motion_encoder),
            "adaptive_fusion":  count(self.adaptive_fusion),
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, NV, NE = 4, 128, 16, 26
    model = GLD2GNNPlus(T=T, device=device).to(device)

    ts_nodes = torch.randn(B, 1, T, NV, device=device)
    ts_edges = torch.randn(B, 1, T, NE, device=device)
    ms_nodes = torch.randn(B, 1, T, NV, device=device)
    ms_edges = torch.randn(B, 1, T, NE, device=device)
    labels   = torch.randint(0, 2, (B, 1), device=device).float()

    print("=" * 60)
    print("  model_improved.py sanity check")
    print("=" * 60)

    # Forward with labels
    out = model(ts_nodes, ts_edges, ms_nodes, ms_edges, labels)
    print(f"\n  prob          : {out['prob'].shape}  range [{out['prob'].min():.3f}, {out['prob'].max():.3f}]")
    print(f"  bce_loss      : {out['bce_loss'].item():.4f}")
    print(f"  sparsity_loss : {out['sparsity_loss'].item():.6f}")
    print(f"  alpha (sample): {out['alpha'].squeeze().tolist()}")

    # Backprop
    total_loss = out['bce_loss'] + out['sparsity_loss']
    total_loss.backward()
    print(f"\n  Total loss    : {total_loss.item():.4f}  (backprop OK)")

    # MC Dropout uncertainty
    result = model.predict_with_uncertainty(
        ts_nodes, ts_edges, ms_nodes, ms_edges, n_passes=10
    )
    print(f"\n  MC Dropout (10 passes):")
    print(f"    mean  : {result['mean'].tolist()}")
    print(f"    std   : {result['std'].tolist()}")
    print(f"    label : {result['label'].tolist()}")
    print(f"    alpha : {result['alpha'].tolist()}")

    # Parameter counts
    counts = model.count_parameters()
    print(f"\n  Parameters:")
    for k, v in counts.items():
        print(f"    {k:<22}: {v:>10,}")

    print("\n  All checks passed.")

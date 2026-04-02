"""
Page 1 — Research Gap Analysis
Shows what the paper does, what's missing, and how we fixed it.
"""

import streamlit as st
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Gap Analysis", layout="wide")
st.title("Research Gap Analysis")
st.caption("What GLD²-GNN does, what it misses, and what we improved.")
st.markdown("---")

# ── GAP 1 ──────────────────────────────────────────────────────────────────
st.subheader("Gap 1 — Fixed scalar α fusion")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Paper (baseline)")
    st.code("""
# Equation 19 — one global α for ALL patients
p_c = α * p_time + (1 - α) * p_motion

# α is a single nn.Parameter (scalar)
# Same weight regardless of:
#   - Patient's disease stage
#   - Gait characteristics
#   - Motion vs. time signal dominance
    """, language="python")
    st.error(
        "Problem: early-stage PD patients have subtle motion differences "
        "but strong time-domain patterns. Late-stage PD is the opposite. "
        "A single α cannot adapt to this.",
        icon="⚠️",
    )

with col2:
    st.markdown("#### Our fix — AdaptiveFusion MLP")
    st.code("""
# Per-sample α predicted by a small MLP
class AdaptiveFusion(nn.Module):
    def __init__(self, feat_dim=128):
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),         # α ∈ (0, 1)
        )

    def forward(self, feat_ts, feat_ms):
        # Reads BOTH stream features
        # Returns a DIFFERENT α per sample
        combined = torch.cat([feat_ts, feat_ms], dim=1)
        return self.mlp(combined)   # [B, 1]
    """, language="python")
    st.success(
        "Each patient gets their own fusion weight, computed from their "
        "own gait features. The model learns when to trust each stream.",
        icon="✅",
    )

st.markdown("---")

# ── GAP 2 ──────────────────────────────────────────────────────────────────
st.subheader("Gap 2 — No uncertainty estimation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Paper (baseline)")
    st.code("""
# Single deterministic forward pass
def predict(self, ...):
    with torch.no_grad():
        prob = self.forward(...)
    return (prob >= 0.5).long()

# Output: just a label  →  0 or 1
# Problem: p=0.97 looks identical to p=0.53
# No way to flag "I'm not sure about this one"
    """, language="python")
    st.error(
        "A model that says 'PD' with 53% confidence should be treated "
        "very differently from one that says 'PD' with 97% confidence. "
        "The baseline cannot distinguish these.",
        icon="⚠️",
    )

with col2:
    st.markdown("#### Our fix — MC Dropout")
    st.code("""
# MCDropout: dropout stays ON at inference
class MCDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)
        #                            ^^^^^^^^^^^^
        #             Always active, even in eval()

# N stochastic forward passes → distribution
def predict_with_uncertainty(self, ..., n_passes=30):
    probs = [self.forward(...) for _ in range(n_passes)]
    mean  = torch.stack(probs).mean(0)   # prediction
    std   = torch.stack(probs).std(0)    # uncertainty
    return mean, std
    """, language="python")
    st.success(
        "Every prediction now comes with a confidence score. "
        "High std → model is uncertain → flag for clinical review.",
        icon="✅",
    )

st.markdown("---")

# ── GAP 3 ──────────────────────────────────────────────────────────────────
st.subheader("Gap 3 — Unconstrained dynamic graph structure")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Paper (baseline)")
    st.code("""
# DGL unit loss — only classification BCE
loss = criterion(prob, label)

# The adjacency matrices A_dyn can be:
#   - Dense  (many weak spurious edges)
#   - Noisy  (edges without anatomical basis)
#   - Hard to interpret  (why these edges?)

# Only constraint: must match predefined mask
# No pressure toward sparse, meaningful graphs
    """, language="python")
    st.error(
        "Dense learned graphs overfit on a small dataset (165 subjects). "
        "They also make the graph visualisation hard to interpret.",
        icon="⚠️",
    )

with col2:
    st.markdown("#### Our fix — Sparsity Regularisation")
    st.code("""
# L1 penalty on all learned adjacency matrices
def _sparsity_loss(self, adj_list):
    total = sum(A.abs().mean() for A in adj_list)
    return self.λ * total / len(adj_list)

# Training loss becomes:
loss = bce_loss + sparsity_loss

# Effect:
#   - Model prefers fewer, stronger edges
#   - Learned graphs are sparser & cleaner
#   - Each edge that survives is meaningful
#   - Visualisations are more interpretable
    """, language="python")
    st.success(
        "Sparser graphs generalise better on small datasets and "
        "produce cleaner visualisations of which pressure paths matter.",
        icon="✅",
    )

st.markdown("---")

# ── Summary table ──────────────────────────────────────────────────────────
st.subheader("Summary")

st.table({
    "Gap": [
        "Fixed scalar α fusion",
        "No uncertainty output",
        "Dense learned graphs",
    ],
    "Paper behaviour": [
        "Same fusion weight for all patients",
        "Single hard prediction only",
        "Unconstrained adjacency learning",
    ],
    "Our fix": [
        "AdaptiveFusion MLP (per-sample α)",
        "MC Dropout (mean + std across 30 passes)",
        "L1 sparsity regularisation on A_dyn",
    ],
    "Cost": [
        "+~8K parameters (MLP)",
        "30× inference time (manageable)",
        "+1 loss term (λ=1e-4)",
    ],
})

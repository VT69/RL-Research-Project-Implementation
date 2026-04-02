"""
Page 4 — Results Comparison
Bar charts + uncertainty calibration plot comparing baseline vs. improved model.
"""

import sys
import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Results", layout="wide")
st.title("Results Comparison")
st.caption("GLD²-GNN (baseline) vs. GLD²-GNN+ (improved)")
st.markdown("---")

# ── Load results JSON ─────────────────────────────────────────────────────────

results_path = st.sidebar.text_input(
    "Comparison JSON path",
    value="./checkpoints/comparison_results.json",
)

paper_results = {
    "Ga+Ju→Si": {"acc": 0.8125, "f1": 0.8334, "gmean": 0.8340},
    "Ga+Si→Ju": {"acc": 0.8721, "f1": 0.9240, "gmean": 0.9252},
    "Si+Ju→Ga": {"acc": 0.8141, "f1": 0.8757, "gmean": 0.8814},
}

if os.path.exists(results_path):
    with open(results_path) as f:
        data = json.load(f)
    has_results = True
    st.sidebar.success("Results loaded ✓")
else:
    has_results = False
    st.sidebar.warning(
        "No comparison results found.\n"
        "Run `python train_improved.py --mode compare` first.\n"
        "Showing paper baseline numbers only."
    )

# ── Paper results table ───────────────────────────────────────────────────────

st.subheader("Paper reported results (Table IV)")
st.caption("Reproduced from Wang et al., IEEE TNSRE 2025")

cols = st.columns(3)
for i, (exp, vals) in enumerate(paper_results.items()):
    with cols[i]:
        st.markdown(f"**{exp}**")
        st.metric("Accuracy", f"{vals['acc']*100:.2f}%")
        st.metric("F1 score", f"{vals['f1']*100:.2f}%")
        st.metric("G-mean",   f"{vals['gmean']*100:.2f}%")

st.markdown("---")

# ── Comparison bar chart ──────────────────────────────────────────────────────

st.subheader("Baseline vs. Improved: metric comparison")

if has_results:
    b = data["baseline"]
    imp = data["improved"]
    exp_label = f"{'+'}.join(data['train_sets']) → {data['test_set']}"

    metrics     = ["Accuracy", "F1 Score", "G-mean"]
    base_means  = [b["acc"]["mean"]*100,   b["f1"]["mean"]*100,   b["gmean"]["mean"]*100]
    base_stds   = [b["acc"]["std"]*100,    b["f1"]["std"]*100,    b["gmean"]["std"]*100]
    imp_means   = [imp["acc"]["mean"]*100, imp["f1"]["mean"]*100, imp["gmean"]["mean"]*100]
    imp_stds    = [imp["acc"]["std"]*100,  imp["f1"]["std"]*100,  imp["gmean"]["std"]*100]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="GLD²-GNN (baseline)",
        x=metrics, y=base_means,
        error_y=dict(type="data", array=base_stds, visible=True),
        marker_color="#888780",
    ))
    fig.add_trace(go.Bar(
        name="GLD²-GNN+ (improved)",
        x=metrics, y=imp_means,
        error_y=dict(type="data", array=imp_stds, visible=True),
        marker_color="#185FA5",
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Score (%)", range=[70, 100]),
        height=380,
        legend=dict(orientation="h", y=1.12),
        title=f"Experiment: {exp_label}",
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta table
    st.subheader("Improvement delta (improved − baseline)")
    delta_acc   = (imp["acc"]["mean"]   - b["acc"]["mean"])   * 100
    delta_f1    = (imp["f1"]["mean"]    - b["f1"]["mean"])    * 100
    delta_gmean = (imp["gmean"]["mean"] - b["gmean"]["mean"]) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("ΔAccuracy", f"{delta_acc:+.2f}%",  delta_color="normal")
    col2.metric("ΔF1 Score", f"{delta_f1:+.2f}%",   delta_color="normal")
    col3.metric("ΔG-mean",   f"{delta_gmean:+.2f}%", delta_color="normal")

else:
    st.info("Run the comparison script to see your results here.")

st.markdown("---")

# ── Uncertainty analysis ──────────────────────────────────────────────────────

st.subheader("Uncertainty analysis (improved model)")

if has_results and "raw_improved" in data and data["raw_improved"]:
    raw  = data["raw_improved"][0]   # first repeat
    probs  = np.array(raw.get("all_probs",  []))
    stds   = np.array(raw.get("all_stds",   []))
    labels = np.array(raw.get("all_labels", []))
    alphas = np.array(raw.get("alpha_vals", []))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Prediction probability vs. uncertainty**")
        st.caption("Each point = one gait cycle. Color = true label.")

        if len(probs) > 0:
            fig2 = go.Figure()
            for lab, color, name in [(0, "#1D9E75", "Healthy"), (1, "#E24B4A", "PD")]:
                mask = labels == lab
                fig2.add_trace(go.Scatter(
                    x=probs[mask], y=stds[mask],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.6),
                    name=name,
                ))
            fig2.update_layout(
                xaxis_title="Mean probability",
                yaxis_title="Uncertainty (std)",
                height=320, margin=dict(t=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("**Reliability diagram (calibration)**")
        st.caption(
            "Perfect calibration = dots on the diagonal. "
            "Points above = underconfident; below = overconfident."
        )
        cal = raw.get("calibration", {})
        if cal:
            bin_confs = cal.get("bin_confs", [])
            bin_accs  = cal.get("bin_accs",  [])
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", line=dict(dash="dash", color="gray"),
                name="Perfect calibration",
            ))
            fig3.add_trace(go.Scatter(
                x=bin_confs, y=bin_accs,
                mode="markers+lines",
                marker=dict(size=8, color="#378ADD"),
                name="Model calibration",
            ))
            fig3.update_layout(
                xaxis_title="Mean confidence",
                yaxis_title="Fraction positive",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=320, margin=dict(t=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Alpha distribution
    st.markdown("---")
    st.subheader("Adaptive fusion weight α distribution")
    st.caption(
        "α > 0.5 = time stream more informative. "
        "α < 0.5 = motion stream more informative. "
        "Spread shows the model IS adapting per sample."
    )
    if len(alphas) > 0:
        fig4 = go.Figure()
        for lab, color, name in [(0, "#1D9E75", "Healthy (CO)"), (1, "#E24B4A", "PD")]:
            mask = labels == lab
            if mask.sum() > 0:
                fig4.add_trace(go.Histogram(
                    x=alphas[mask], name=name,
                    marker_color=color, opacity=0.7,
                    nbinsx=20,
                ))
        fig4.add_vline(x=0.5, line_dash="dash", line_color="gray",
                       annotation_text="Equal weight")
        fig4.update_layout(
            barmode="overlay",
            xaxis_title="Fusion weight α",
            yaxis_title="Count",
            height=300, margin=dict(t=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

        mean_pd = alphas[labels == 1].mean() if (labels == 1).sum() > 0 else 0
        mean_co = alphas[labels == 0].mean() if (labels == 0).sum() > 0 else 0
        c1, c2 = st.columns(2)
        c1.metric("Mean α (PD patients)",      f"{mean_pd:.3f}")
        c2.metric("Mean α (healthy controls)", f"{mean_co:.3f}")

else:
    st.info("Run the comparison script to see uncertainty analysis here.")

# ── Training curves ───────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Training curves")

if has_results and "raw_improved" in data and data["raw_improved"]:
    history = data["raw_improved"][0].get("history", {})
    if history:
        fig5 = make_subplots(rows=1, cols=2,
                              subplot_titles=["Loss components", "Validation metrics"])

        epochs = list(range(1, len(history.get("train_total_loss", [])) + 1))
        fig5.add_trace(go.Scatter(x=epochs, y=history.get("train_total_loss", []),
                                   name="Total loss", line=dict(color="#E24B4A")), row=1, col=1)
        fig5.add_trace(go.Scatter(x=epochs, y=history.get("train_bce_loss", []),
                                   name="BCE loss",   line=dict(color="#378ADD")), row=1, col=1)
        fig5.add_trace(go.Scatter(x=epochs, y=history.get("train_sparsity_loss", []),
                                   name="Sparsity",   line=dict(color="#EF9F27")), row=1, col=1)

        fig5.add_trace(go.Scatter(x=epochs, y=history.get("val_acc",   []),
                                   name="Val Acc",    line=dict(color="#1D9E75")), row=1, col=2)
        fig5.add_trace(go.Scatter(x=epochs, y=history.get("val_f1",    []),
                                   name="Val F1",     line=dict(color="#7F77DD")), row=1, col=2)
        fig5.add_trace(go.Scatter(x=epochs, y=history.get("val_gmean", []),
                                   name="Val Gmean",  line=dict(color="#D85A30")), row=1, col=2)

        fig5.update_layout(height=320, margin=dict(t=40),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig5, use_container_width=True)

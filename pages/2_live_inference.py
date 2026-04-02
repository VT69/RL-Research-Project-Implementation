"""
Page 2 — Live Inference
Upload a raw VGRF .txt file → prediction + MC Dropout uncertainty.
"""

import sys
import os
import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader     import segment_gait_cycles, resize_cycle, build_two_streams, build_edge_sequence
from model_improved  import GLD2GNNPlus

st.set_page_config(page_title="Live Inference", layout="wide")
st.title("Live Inference")
st.caption("Upload a raw VGRF `.txt` file and get a prediction with uncertainty.")
st.markdown("---")

# ── Model loader ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, T: int = 128):
    device = torch.device("cpu")
    model  = GLD2GNNPlus(T=T, device=device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        return model, True
    return model, False


# ── Sidebar controls ────────────────────────────────────────────────────────

st.sidebar.header("Settings")
ckpt_path  = st.sidebar.text_input(
    "Checkpoint path",
    value="./checkpoints/improved_r0_best.pt",
)
n_passes   = st.sidebar.slider("MC Dropout passes", 10, 50, 30)
T          = st.sidebar.selectbox("Gait cycle length", [128, 256], index=0)
threshold  = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

model, loaded = load_model(ckpt_path, T=T)
if loaded:
    st.sidebar.success("Model loaded ✓")
else:
    st.sidebar.warning("Checkpoint not found — using untrained weights")

# ── File upload ─────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a PhysioNet VGRF .txt file",
    type=["txt"],
    help="Raw whitespace-separated file with 18+ columns. Column 0 = time, columns 1-16 = 16 sensors.",
)

if uploaded is not None:
    # Parse file
    try:
        raw = np.loadtxt(uploaded, dtype=np.float32)
        if raw.ndim == 1 or raw.shape[1] < 17:
            st.error("File format error: expected at least 17 columns (time + 16 sensors).")
            st.stop()

        signal = raw[:, 1:17]   # [T_raw, 16]
        st.success(f"Loaded: {signal.shape[0]} frames × 16 sensors")

    except Exception as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

    # Segment and predict
    cycles = segment_gait_cycles(signal)
    if not cycles:
        st.warning("No gait cycles detected. The file may be too short or have unusual structure.")
        st.stop()

    st.info(f"Detected {len(cycles)} gait cycles. Running inference on all cycles...")

    device = torch.device("cpu")
    all_results = []

    for i, cycle in enumerate(cycles):
        r           = resize_cycle(cycle, T)
        ts, ms      = build_two_streams(r)
        ts_e        = build_edge_sequence(ts)
        ms_e        = build_edge_sequence(ms)

        ts_nodes = torch.from_numpy(ts).unsqueeze(0).unsqueeze(0)   # [1,1,T,16]
        ts_edges = torch.from_numpy(ts_e).unsqueeze(0).unsqueeze(0) # [1,1,T,26]
        ms_nodes = torch.from_numpy(ms).unsqueeze(0).unsqueeze(0)
        ms_edges = torch.from_numpy(ms_e).unsqueeze(0).unsqueeze(0)

        result = model.predict_with_uncertainty(
            ts_nodes, ts_edges, ms_nodes, ms_edges,
            n_passes=n_passes, threshold=threshold,
        )

        all_results.append({
            "cycle":  i + 1,
            "mean":   result["mean"].item(),
            "std":    result["std"].item(),
            "label":  result["label"].item(),
            "alpha":  result["alpha"].item(),
        })

    # ── Aggregate prediction via majority vote ───────────────────────────────
    labels     = [r["label"] for r in all_results]
    means      = [r["mean"]  for r in all_results]
    stds       = [r["std"]   for r in all_results]
    alphas     = [r["alpha"] for r in all_results]

    vote_label = 1 if sum(labels) > len(labels) / 2 else 0
    avg_prob   = float(np.mean(means))
    avg_std    = float(np.mean(stds))
    avg_alpha  = float(np.mean(alphas))

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label_str = "Parkinson's Disease" if vote_label == 1 else "Healthy Control"
        color     = "🔴" if vote_label == 1 else "🟢"
        st.metric("Diagnosis", f"{color} {label_str}")

    with col2:
        st.metric("Mean probability", f"{avg_prob:.3f}")

    with col3:
        confidence = "High" if avg_std < 0.1 else ("Medium" if avg_std < 0.2 else "Low")
        st.metric("Uncertainty (std)", f"{avg_std:.3f}", delta=f"Confidence: {confidence}")

    with col4:
        st.metric("Avg fusion α", f"{avg_alpha:.3f}",
                  help="Closer to 1 = time stream dominant, closer to 0 = motion stream dominant")

    # ── Uncertainty visualisation ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Per-cycle predictions")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(all_results)+1)),
        y=means,
        error_y=dict(type="data", array=stds, visible=True),
        mode="markers+lines",
        marker=dict(
            color=["#E24B4A" if l == 1 else "#1D9E75" for l in labels],
            size=10,
        ),
        name="Prediction ± uncertainty",
    ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Threshold ({threshold})")

    fig.update_layout(
        xaxis_title="Gait cycle",
        yaxis_title="PD probability",
        yaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MC Dropout distribution for first cycle ──────────────────────────────
    st.subheader("MC Dropout pass distribution (first cycle)")
    st.caption(
        "Each dot is one stochastic forward pass. Spread = uncertainty. "
        "Narrow cluster = confident prediction."
    )

    cycle0 = cycles[0]
    r      = resize_cycle(cycle0, T)
    ts, ms = build_two_streams(r)
    ts_e   = build_edge_sequence(ts)
    ms_e   = build_edge_sequence(ms)

    ts_nodes = torch.from_numpy(ts).unsqueeze(0).unsqueeze(0)
    ts_edges = torch.from_numpy(ts_e).unsqueeze(0).unsqueeze(0)
    ms_nodes = torch.from_numpy(ms).unsqueeze(0).unsqueeze(0)
    ms_edges = torch.from_numpy(ms_e).unsqueeze(0).unsqueeze(0)

    res0    = model.predict_with_uncertainty(
        ts_nodes, ts_edges, ms_nodes, ms_edges, n_passes=n_passes
    )
    passes  = res0["passes"].squeeze().tolist()

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=passes, nbinsx=20,
        marker_color="#378ADD", opacity=0.75,
        name="Forward passes",
    ))
    fig2.add_vline(x=threshold, line_dash="dash", line_color="gray",
                   annotation_text="Threshold")
    fig2.add_vline(x=float(np.mean(passes)), line_color="#E24B4A",
                   annotation_text=f"Mean={np.mean(passes):.3f}")
    fig2.update_layout(
        xaxis_title="Predicted probability",
        yaxis_title="Count",
        height=300, margin=dict(t=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw signal preview ───────────────────────────────────────────────────
    with st.expander("Raw VGRF signal preview"):
        fig3 = px.line(
            x=list(range(signal.shape[0])),
            y=signal.mean(axis=1),
            labels={"x": "Frame", "y": "Mean force (N)"},
            title="Mean force across all 16 sensors",
        )
        fig3.update_layout(height=250, margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info(
        "Upload a `.txt` file from the PhysioNet gaitpdb dataset to run inference. "
        "Files are named like `GaCo01_01.txt` (healthy) or `GaPd01_01.txt` (PD)."
    )

    st.markdown("""
    **Expected file format:**
    ```
    0.00  0.0  0.0  ...  (18+ whitespace-separated columns)
    0.01  1.2  0.8  ...
    ...
    ```
    Column 0 = timestamp, columns 1–16 = 16 sensor readings (N), columns 17+ = totals.
    """)

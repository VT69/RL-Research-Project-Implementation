"""
Page 3 — Dynamic Graph Viewer
Visualises how the learned pressure-transmission graph changes
across gait phases, for a selected VGRF file.
"""

import sys
import os
import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from graph_construction import get_graph_components, NV as NV_CONST, _build_edge_list
from data_loader        import segment_gait_cycles, resize_cycle
from model_improved     import GLD2GNNPlus
from dgl_unit           import DGLUnit

st.set_page_config(page_title="Graph Viewer", layout="wide")
st.title("Dynamic Graph Viewer")
st.caption(
    "Watch how the learned pressure-transmission graph evolves "
    "across time slots of a gait cycle."
)
st.markdown("---")

# ── Sensor positions (approximate plantar layout, normalised 0-1) ────────────
# Left foot: nodes 0-7 (left side of plot), Right foot: nodes 8-15 (right side)
SENSOR_XY = {
    # Left foot (x ~ 0.1-0.4, heel at bottom, toe at top)
    0:  (0.15, 0.10),  # L-heel lateral
    1:  (0.25, 0.10),  # L-heel medial
    2:  (0.15, 0.30),  # L-midfoot lateral
    3:  (0.25, 0.30),  # L-midfoot medial
    4:  (0.15, 0.55),  # L-metatarsal lateral
    5:  (0.25, 0.55),  # L-metatarsal medial
    6:  (0.15, 0.80),  # L-toe lateral
    7:  (0.25, 0.80),  # L-toe medial
    # Right foot (x ~ 0.6-0.9, mirrored)
    8:  (0.75, 0.10),
    9:  (0.85, 0.10),
    10: (0.75, 0.30),
    11: (0.85, 0.30),
    12: (0.75, 0.55),
    13: (0.85, 0.55),
    14: (0.75, 0.80),
    15: (0.85, 0.80),
}

GAIT_PHASE_LABELS = [
    "Initial contact",
    "Loading response",
    "Mid-stance",
    "Terminal stance",
    "Pre-swing",
    "Initial swing",
    "Mid-swing",
    "Terminal swing",
]


@st.cache_resource
def load_model_and_gc(ckpt_path: str, T: int = 128):
    device = torch.device("cpu")
    model  = GLD2GNNPlus(T=T, device=device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    gc = get_graph_components(device)
    return model, gc, device


def adjacency_to_plotly(
    A_dyn:    np.ndarray,   # [NV, NV]
    A_static: np.ndarray,   # [NV, NV]
    node_vals: np.ndarray,  # [NV]  pressure values for node colour
    title:    str,
) -> go.Figure:
    """Renders one adjacency snapshot as a plotly figure."""
    fig = go.Figure()

    nv = A_dyn.shape[0]
    xs = [SENSOR_XY[i][0] for i in range(nv)]
    ys = [SENSOR_XY[i][1] for i in range(nv)]

    # Predefined edges (light gray)
    for i in range(nv):
        for j in range(nv):
            if A_static[i, j] > 0:
                fig.add_trace(go.Scatter(
                    x=[xs[i], xs[j]], y=[ys[i], ys[j]],
                    mode="lines",
                    line=dict(color="rgba(180,180,180,0.3)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))

    # Dynamic learned edges (coloured by weight)
    for i in range(nv):
        for j in range(nv):
            w = float(A_dyn[i, j])
            if w > 0.01:
                fig.add_trace(go.Scatter(
                    x=[xs[i], xs[j]], y=[ys[i], ys[j]],
                    mode="lines",
                    line=dict(
                        color=f"rgba(55, 138, 221, {min(w, 1.0):.2f})",
                        width=max(1, w * 6),
                    ),
                    showlegend=False, hoverinfo="skip",
                ))

    # Nodes coloured by pressure value
    norm_vals = (node_vals - node_vals.min()) / (node_vals.max() - node_vals.min() + 1e-8)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=18,
            color=norm_vals.tolist(),
            colorscale="RdBu_r",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Pressure", thickness=12, len=0.6),
            line=dict(width=1.5, color="white"),
        ),
        text=[f"v{i}" for i in range(nv)],
        textposition="middle center",
        textfont=dict(size=9, color="white"),
        hovertext=[
            f"Node {i}<br>Pressure: {node_vals[i]:.2f}"
            for i in range(nv)
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    # Foot labels
    fig.add_annotation(x=0.20, y=0.02, text="Left foot",  showarrow=False,
                        font=dict(size=11, color="gray"))
    fig.add_annotation(x=0.80, y=0.02, text="Right foot", showarrow=False,
                        font=dict(size=11, color="gray"))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.08, 1.0]),
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")
ckpt_path = st.sidebar.text_input("Checkpoint path", "./checkpoints/improved_r0_best.pt")
T         = st.sidebar.selectbox("Gait cycle length", [128, 256], index=0)
s         = st.sidebar.selectbox("Time slots (s)", [4, 8], index=1)

model, gc, device = load_model_and_gc(ckpt_path, T=T)
A_static  = gc["A"].numpy()

# ── File upload or demo mode ─────────────────────────────────────────────────

uploaded = st.file_uploader("Upload a VGRF .txt file (optional)", type=["txt"])

if uploaded is not None:
    raw    = np.loadtxt(uploaded, dtype=np.float32)
    signal = raw[:, 1:17]
    cycles = segment_gait_cycles(signal)
    if cycles:
        cycle = resize_cycle(cycles[0], T)
    else:
        st.warning("No cycles detected — using synthetic data.")
        cycle = np.random.rand(T, 16).astype(np.float32) * 100
else:
    st.info("No file uploaded — showing synthetic gait data for demonstration.")
    np.random.seed(42)
    t_arr = np.linspace(0, 2 * np.pi, T)
    cycle = np.zeros((T, 16), dtype=np.float32)
    for i in range(16):
        phase = (i / 16) * np.pi
        cycle[:, i] = (np.sin(t_arr + phase) + 1) * 50 + np.random.rand(T) * 10

# ── Run DGL unit to get dynamic adjacency matrices ───────────────────────────

node_data = torch.from_numpy(cycle).unsqueeze(0).unsqueeze(0)  # [1,1,T,16]
A_static_t = gc["A"]

dgl_unit = DGLUnit(C_in=1, T=T, s=s, NV=16, topk=6)
dgl_unit.eval()

with torch.no_grad():
    adj_list = dgl_unit(node_data, A_static_t)   # list of s tensors [16,16]

adj_arrays  = [A.numpy() for A in adj_list]
T_slot      = T // s

# ── Slot selector ─────────────────────────────────────────────────────────────

st.subheader("Gait phase adjacency snapshots")
st.caption(
    "Blue edges = learned dynamic connections. "
    "Gray edges = predefined anatomy. "
    "Node color = relative pressure (red=high, blue=low)."
)

phase_labels = GAIT_PHASE_LABELS[:s] if s <= 8 else [f"Slot {i+1}" for i in range(s)]

tab_labels = [f"Phase {i+1}: {phase_labels[i]}" for i in range(s)]
tabs       = st.tabs(tab_labels)

for k, tab in enumerate(tabs):
    with tab:
        slot_nodes = cycle[k * T_slot: (k + 1) * T_slot, :]   # [T_slot, 16]
        mean_pressure = slot_nodes.mean(axis=0)                 # [16]

        fig = adjacency_to_plotly(
            A_dyn    = adj_arrays[k],
            A_static = A_static,
            node_vals = mean_pressure,
            title    = f"{phase_labels[k]} — learned graph snapshot",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats for this slot
        nnz  = int((adj_arrays[k] > 0.01).sum())
        mean_w = float(adj_arrays[k][adj_arrays[k] > 0.01].mean()) if nnz > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Dynamic edges", nnz)
        c2.metric("Mean edge weight", f"{mean_w:.3f}")
        c3.metric("Max pressure node", f"v{int(mean_pressure.argmax())}")

# ── Edge count across all slots ───────────────────────────────────────────────

st.markdown("---")
st.subheader("Dynamic edge count across gait phases")
st.caption("How many pressure-transmission edges the model activates per phase.")

edge_counts = [(adj_arrays[k] > 0.01).sum() for k in range(s)]

fig_bar = go.Figure(go.Bar(
    x=phase_labels,
    y=edge_counts,
    marker_color="#378ADD",
    text=edge_counts,
    textposition="outside",
))
fig_bar.update_layout(
    yaxis_title="Dynamic edges", height=280, margin=dict(t=10),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Predefined vs. learned comparison ─────────────────────────────────────────

st.markdown("---")
st.subheader("Predefined graph vs. typical learned graph")
col1, col2 = st.columns(2)

dummy_pressure = np.ones(16) * 50

with col1:
    fig_static = adjacency_to_plotly(
        A_dyn    = np.zeros((16, 16)),
        A_static = A_static,
        node_vals = dummy_pressure,
        title    = "Predefined directed graph (26 fixed edges)",
    )
    st.plotly_chart(fig_static, use_container_width=True)

with col2:
    # Average learned graph across all slots
    avg_adj = np.mean(adj_arrays, axis=0)
    fig_dyn = adjacency_to_plotly(
        A_dyn    = avg_adj,
        A_static = A_static,
        node_vals = dummy_pressure,
        title    = "Average learned dynamic graph",
    )
    st.plotly_chart(fig_dyn, use_container_width=True)

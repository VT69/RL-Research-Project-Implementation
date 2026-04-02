"""
dashboard/app.py
================
GLD²-GNN+ Research Dashboard — Streamlit entry point.

Run:
    cd dashboard
    streamlit run app.py

Pages (in sidebar):
    1. Research Gap Analysis    — paper vs. proposed, side by side
    2. Live Inference           — upload VGRF file, get prediction + uncertainty
    3. Dynamic Graph Viewer     — animated adjacency matrices per gait phase
    4. Results Comparison       — Acc / F1 / Gmean bar charts baseline vs. improved
"""

import streamlit as st

st.set_page_config(
    page_title  = "GLD²-GNN+ Dashboard",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.title("GLD²-GNN+: Parkinson's Disease Detection")
st.caption(
    "Unofficial implementation & improvement of *Wang et al., IEEE TNSRE 2025*"
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset subjects", "165", help="93 PD + 72 healthy controls")
    st.metric("Sensor nodes", "16", help="8 per foot, plantar pressure")

with col2:
    st.metric("Baseline Acc (Ga+Ju→Si)", "81.25%", help="From paper Table IV")
    st.metric("Improved Acc (Ga+Ju→Si)", "—", help="Run train_improved.py first")

with col3:
    st.metric("Model params (baseline)", "~0.46M")
    st.metric("New: uncertainty output", "Yes ✓", help="MC Dropout, 30 passes")

st.markdown("---")
st.markdown(
    """
    ### How to use this dashboard

    Use the **sidebar** to navigate between pages:

    | Page | What it shows |
    |------|---------------|
    | Research Gap Analysis | Side-by-side comparison of the paper's limitations and our fixes |
    | Live Inference | Upload a raw VGRF `.txt` file and get a live PD prediction with confidence |
    | Dynamic Graph Viewer | Watch the learned pressure-transmission graph evolve across a gait cycle |
    | Results Comparison | Bar charts of Acc / F1 / G-mean: baseline vs. improved model |

    ### Quick start

    ```bash
    # 1. Train both models (runs baseline + improved, saves comparison JSON)
    python train_improved.py --mode compare --data_root ./data --epochs 120

    # 2. Launch dashboard
    cd dashboard
    streamlit run app.py
    ```
    """
)

st.info(
    "This is an academic research project implementation. "
    "Not intended for clinical use.",
    icon="ℹ️",
)

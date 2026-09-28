"""
app.py - AI Hub Model Picker (Streamlit UI)

Type an AI use case in plain English, get the best-matching Qualcomm AI Hub
models from a local catalog. Fully offline once dependencies are installed.

Run with:  streamlit run app.py
"""

import plotly.graph_objects as go
import streamlit as st

import embed_utils as eu

st.set_page_config(page_title="AI Hub Model Picker", page_icon="🔎", layout="centered")


# ---------------------------------------------------------------------------
# Cached resources: load the catalog and its embeddings only once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model catalog and embeddings...")
def load_resources():
    catalog = eu.load_catalog()
    vectors = eu.embed_catalog()  # reuses embeddings.npy if it already exists
    return catalog, vectors


catalog, catalog_vecs = load_resources()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔎 AI Hub Model Picker")
st.write(
    "Describe what you want to build. The matching runs **locally on this "
    "device** - no query or data is sent to the cloud."
)

query = st.text_input(
    "Describe your AI use case",
    placeholder="e.g. transcribe and summarize customer support calls",
)
search_clicked = st.button("Search", type="primary")

# ---------------------------------------------------------------------------
# Search + results
# ---------------------------------------------------------------------------
if search_clicked:
    if not query.strip():
        st.warning("Please describe a use case first.")
    else:
        query_vec = eu.embed_query(query)
        indices, scores = eu.rank_matches(query_vec, catalog_vecs, top_k=5)

        st.subheader("Top matches")
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            model = catalog[int(idx)]
            match_pct = max(0.0, float(score)) * 100  # cosine similarity as %

            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f"**{rank}. {model['name']}**  \n`{model['category']}`")
                right.metric("Match", f"{match_pct:.0f}%")
                st.write(model["description"])
                c1, c2 = st.columns(2)
                c1.markdown(f"**Latency:** ~{model['latency_ms']:g} ms")
                c2.markdown(f"**Power:** {model['power_note']}")

        st.caption(
            "Latency and power figures are example values - to be replaced with "
            "real AI Hub Workbench profiling results."
        )

# ---------------------------------------------------------------------------
# CPU vs NPU timing expander
# ---------------------------------------------------------------------------
# TODO (Snapdragon NPU port): this panel is a STAND-IN for a true QNN-vs-CPU
# comparison. Right now it times the same embedding twice on the sentence-
# transformers library - once on its default backend and once forced to CPU.
# On a normal laptop both resolve to the CPU, so the two bars will be roughly
# equal. Once the embedding model is compiled through Qualcomm AI Hub and run
# via ONNX Runtime's QNN execution provider on a Snapdragon device, replace
# the "default backend" measurement with the QNN (Hexagon NPU) measurement.
# See README.md -> "Porting to Snapdragon NPU".
with st.expander("Show CPU vs NPU inference timing"):
    st.write(
        "Times the same query embedding on two backends. Until this runs on "
        "real Snapdragon hardware, both bars use the CPU (see the TODO in "
        "`app.py`)."
    )
    if st.button("Run timing comparison"):
        sample = query.strip() or "transcribe and summarize customer support calls"
        with st.spinner("Timing..."):
            default_ms = eu.time_embedding(sample, device=None)
            cpu_ms = eu.time_embedding(sample, device="cpu")

        fig = go.Figure(
            go.Bar(
                x=["Default backend", "CPU-only"],
                y=[default_ms, cpu_ms],
                text=[f"{default_ms:.1f} ms", f"{cpu_ms:.1f} ms"],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Query embedding time (lower is better)",
            yaxis_title="milliseconds",
            height=360,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, width="stretch")

"""
web/app.py
===========
Interactive Streamlit GUI for SatQuery AI.

Features
--------
* Upload single image, bi-temporal pair, or cross-modal (optical+SAR) pair.
* Natural-language query box with example-query shortcuts.
* Dual/split raster viewer with bounding-box and change/class-mask overlays.
* Live tool-execution pipeline graph (from `agent.orchestrator.
  AgentController.pipeline_graph`).
* Confidence metrics panel.
* One-click PDF and GeoJSON report export (`web.report_generator`).

Run with:  streamlit run web/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run web/app.py` to resolve the `satquery_ai` package
# root regardless of the working directory it's launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core import config
from core.geo_io import read_image
from agent.orchestrator import AgentController
from web.report_generator import generate_pdf_report, generate_geojson_report

config.ensure_dirs()

st.set_page_config(page_title="SatQuery AI", layout="wide", page_icon="🛰️")

EXAMPLE_QUERIES = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
]


@st.cache_resource
def get_controller() -> AgentController:
    return AgentController()


def _save_upload(uploaded_file) -> str:
    dest = config.UPLOAD_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def _display_array_for_plot(path: str) -> np.ndarray:
    """Load and return an RGB-displayable (H, W, 3) uint8 array from any
    supported raster, for quick preview / overlay rendering."""
    img = read_image(path)
    from core.geo_io import normalize_optical, normalize_sar
    if img.modality == "unknown":
        from core.validator import classify_modality
        img.modality = classify_modality(img)
    if img.modality == "sar":
        norm = normalize_sar(img, config.THRESHOLDS["sar_db_min"], config.THRESHOLDS["sar_db_max"])
        gray = norm[0]
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        norm = normalize_optical(img, config.THRESHOLDS["optical_percentile_low"],
                                   config.THRESHOLDS["optical_percentile_high"])
        if norm.shape[0] >= 3:
            rgb = np.transpose(norm[:3], (1, 2, 0))
        else:
            gray = norm[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _render_overlay(path: str, boxes: list, title: str):
    arr = _display_array_for_plot(path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    colors = {"water": "#1f77ff", "builtup": "#ff4136", "vegetation": "#2ecc40",
              "change": "#ffdc00", "road": "#b10dc9", "salient": "#ff851b"}
    for b in boxes:
        c = colors.get(b.get("label"), "#ffffff")
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                  linewidth=2, edgecolor=c, facecolor="none")
        ax.add_patch(rect)
        ax.text(b["xmin"], max(b["ymin"] - 4, 0), f"{b.get('label','')} {b.get('score',0):.2f}",
                color=c, fontsize=8, weight="bold")
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("🛰️ SatQuery AI")
    st.caption("Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis")

    with st.sidebar:
        st.header("1. Input Configuration")
        input_mode = st.radio(
            "Select input configuration",
            ["Single Image", "Bi-Temporal Pair (Change Detection)", "Cross-Modal Pair (Optical + SAR)"],
        )
        st.divider()
        st.header("Tool Registry")
        try:
            controller_preview = get_controller()
            for tool in controller_preview.pipeline_graph():
                with st.expander(tool["name"]):
                    st.write(f"**Tasks:** {', '.join(tool['task_types'])}")
                    st.write(f"**Input configs:** {', '.join(tool['input_configs'])}")
                    st.caption(tool["description"])
        except Exception as exc:
            st.warning(f"Tool registry unavailable: {exc}")

    image_paths = []
    declared_pair_type = None

    if input_mode == "Single Image":
        f = st.file_uploader("Upload GeoTIFF/TIFF/PNG/JPEG", type=["tif", "tiff", "png", "jpg", "jpeg"])
        if f:
            image_paths = [_save_upload(f)]

    elif input_mode == "Bi-Temporal Pair (Change Detection)":
        declared_pair_type = config.INPUT_CONFIG_BI_TEMPORAL
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.file_uploader("Time T1 image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t1")
        with col2:
            f2 = st.file_uploader("Time T2 image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="t2")
        if f1 and f2:
            image_paths = [_save_upload(f1), _save_upload(f2)]

    else:  # Cross-Modal Pair
        declared_pair_type = config.INPUT_CONFIG_CROSS_MODAL
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.file_uploader("Optical / Multispectral image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="opt")
        with col2:
            f2 = st.file_uploader("SAR image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="sar")
        if f1 and f2:
            image_paths = [_save_upload(f1), _save_upload(f2)]

    st.header("2. Natural-Language Query")
    example = st.selectbox("Example queries (optional)", ["(type your own below)"] + EXAMPLE_QUERIES)
    default_text = "" if example == "(type your own below)" else example
    query = st.text_area("Query", value=default_text, height=80)

    run = st.button("🚀 Run SatQuery AI", type="primary", disabled=not (image_paths and query.strip()))

    if run:
        controller = get_controller()
        with st.spinner("Running agentic pipeline..."):
            trace = controller.run_query(image_paths, query, declared_pair_type=declared_pair_type)
        st.session_state["last_trace"] = trace
        st.session_state["last_image_paths"] = image_paths

    if "last_trace" in st.session_state:
        trace = st.session_state["last_trace"]
        image_paths = st.session_state.get("last_image_paths", [])

        st.header("3. Results")
        status_color = "green" if trace["status"] == "success" else "red"
        st.markdown(f"**Status:** :{status_color}[{trace['status']}]  |  "
                    f"**Task:** `{trace['selected_task']}`  |  "
                    f"**Input config:** `{trace['input_configuration']}`")

        st.subheader("Evidence-Grounded Answer")
        st.success(trace["evidence_grounded_answer"])

        conf = trace["confidence"]["final_score"]
        if conf is not None:
            st.metric("Confidence", f"{conf:.0%}")
            st.progress(min(max(conf, 0.0), 1.0))

        boxes = trace["spatial_evidence"]["bounding_boxes"]
        if image_paths:
            st.subheader("Visual Evidence")
            cols = st.columns(len(image_paths))
            for i, (col, p) in enumerate(zip(cols, image_paths)):
                with col:
                    _render_overlay(p, boxes if i == 0 else [], title=Path(p).name)

        if trace["warnings"]:
            st.subheader("Warnings")
            for w in trace["warnings"]:
                st.warning(w)

        with st.expander("🔍 Full Auditable Execution Trace (JSON)"):
            st.json(trace)

        st.subheader("4. Export Report")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report"):
                pdf_path = generate_pdf_report(trace, image_paths)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=Path(pdf_path).name, mime="application/pdf")
        with col2:
            geojson_str = generate_geojson_report(trace, image_paths)
            st.download_button("⬇️ Download GeoJSON", geojson_str, file_name="satquery_result.geojson",
                                mime="application/geo+json")


if __name__ == "__main__":
    main()

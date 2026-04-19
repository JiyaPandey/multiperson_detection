import time
import os
import sys

import streamlit as st

# Ensure project root is importable when running from src/ui
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.multi_cam_multi_person import run_pipeline as run_multi
from scripts.multi_cam_single_person import run_pipeline as run_reid
from scripts.single_cam import run_pipeline as run_single

st.set_page_config(layout="wide", page_title="Multi-Camera Analytics Dashboard")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(1000px 500px at 15% -5%, rgba(16, 185, 129, 0.16), transparent 60%),
                radial-gradient(1000px 600px at 100% -10%, rgba(56, 189, 248, 0.16), transparent 55%),
                linear-gradient(135deg, #020617, #0f172a);
    color: #f8fafc;
}

.card {
    background: rgba(255, 255, 255, 0.06);
    padding: 24px;
    border-radius: 20px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-bottom: 26px;
    box-shadow: 0 16px 30px rgba(2, 6, 23, 0.35);
}

.title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 0.2px;
}

.subtitle {
    margin-top: 4px;
    color: #94a3b8;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
}

.desc {
    color: #cbd5e1;
    margin-bottom: 14px;
}

.legend-wrap {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 14px 0 18px 0;
}

.legend-pill {
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid rgba(255, 255, 255, 0.25);
}

[data-testid="stImage"] img {
    animation: fadeIn 260ms ease-in;
    border-radius: 12px;
}

@keyframes fadeIn {
    from { opacity: 0.55; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0px); }
}
</style>
""",
    unsafe_allow_html=True,
)

if "playing" not in st.session_state:
    st.session_state.playing = True

st.markdown('<div class="title">Multi-Camera Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-time Person Detection, Tracking, and Re-Identification</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="legend-wrap">
  <span class="legend-pill" style="background: rgba(34,197,94,0.22);">Scenario 1: Single Cam Multi-Person</span>
  <span class="legend-pill" style="background: rgba(15,23,42,0.42);">Scenario 2: Cross-Camera ReID</span>
  <span class="legend-pill" style="background: rgba(56,189,248,0.22);">Scenario 3: Multi-Cam Multi-Person</span>
  <span class="legend-pill" style="background: rgba(245,158,11,0.22);">Heatmap: Activity Intensity</span>
  <span class="legend-pill" style="background: rgba(168,85,247,0.22);">2D Map: Ground Position Path</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Controls")
    if st.button("Play" if not st.session_state.playing else "Pause", use_container_width=True):
        st.session_state.playing = not st.session_state.playing

    refresh_ms = st.slider("Refresh (ms)", min_value=20, max_value=200, value=35, step=5)

    selected_scenarios = st.multiselect(
        "Scenario Selector",
        options=["Scenario 1", "Scenario 2", "Scenario 3"],
        default=["Scenario 1", "Scenario 2", "Scenario 3"],
    )

    status = "Running" if st.session_state.playing else "Paused"
    st.caption(f"Status: {status}")


def safe_run(func):
    try:
        return func()
    except Exception as exc:
        msg = str(exc).strip()
        if not msg:
            msg = "No message"
        return None, {"error": f"{type(exc).__name__}: {msg}"}


def render_scenario(title, desc, frame, stats, color="white"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="color:{color};">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="desc">{desc}</div>', unsafe_allow_html=True)

    if "error" in stats:
        st.error(f"Pipeline error: {stats['error']}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if frame is not None:
        st.image(frame, use_container_width=True)

    metric_col, heat_col, map_col = st.columns(3)

    with metric_col:
        st.metric("Active IDs", int(stats.get("active_ids", 0)))
        st.metric("Total IDs", int(stats.get("total_ids", 0)))
        st.metric("Frame", int(stats.get("frame", 0)))

    with heat_col:
        if stats.get("heatmap") is not None:
            st.image(stats.get("heatmap"), use_container_width=True)
        else:
            st.caption("Heatmap not available for this scenario")

    with map_col:
        if stats.get("map") is not None:
            st.image(stats.get("map"), use_container_width=True)
        else:
            st.caption("2D map not available for this scenario")

    st.markdown("</div>", unsafe_allow_html=True)


f1, s1 = safe_run(run_single) if "Scenario 1" in selected_scenarios else (None, {})
f2, s2 = safe_run(run_reid) if "Scenario 2" in selected_scenarios else (None, {})
f3, s3 = safe_run(run_multi) if "Scenario 3" in selected_scenarios else (None, {})

if "Scenario 1" in selected_scenarios:
    render_scenario(
        "Scenario 1: Multi-Person Detection",
        "Detects and tracks multiple people in a single camera feed.",
        f1,
        s1,
        "#22c55e",
    )

if "Scenario 2" in selected_scenarios:
    render_scenario(
        "Scenario 2: Cross-Camera ReID",
        "Tracks the same person across camera splits with global identity consistency.",
        f2,
        s2,
        "#e2e8f0",
    )

if "Scenario 3" in selected_scenarios:
    render_scenario(
        "Scenario 3: Multi-Cam Multi-Person",
        "Tracks multiple people across multiple cameras with ReID matching.",
        f3,
        s3,
        "#38bdf8",
    )

if st.session_state.playing:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
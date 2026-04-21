import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.multi_cam_multi_person import run_pipeline as run_multi
from scripts.multi_cam_single_person import run_pipeline as run_reid
from scripts.single_cam import run_pipeline as run_single

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Multi-Camera Analytics Dashboard",
    page_icon="🎯",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Global CSS — Tesla-style dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset / Base ────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #05080f !important;
    color: #e2e8f0;
}

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #080d18 0%, #0b1220 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.18) !important;
    padding: 1.2rem 1rem !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stToggle label {
    font-size: 0.78rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.06rem;
}

.sidebar-heading {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14rem;
    text-transform: uppercase;
    color: #6366f1;
    margin: 1rem 0 0.55rem 0;
    display: block;
}
.sidebar-logo {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.06rem;
    margin-bottom: 0.15rem;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-bottom: 0.6rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.15);
    margin: 0.8rem 0;
}

/* ── Centered content wrapper ────────────────────────────────────────────── */
.dash-wrapper {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ── Dashboard header ────────────────────────────────────────────────────── */
.dash-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 0 0.75rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.12);
    margin-bottom: 1.4rem;
}
.dash-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02rem;
    color: #f1f5f9;
}
.dash-badge {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12rem;
    text-transform: uppercase;
    color: #a5b4fc;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(99,102,241,0.32);
    border-radius: 5px;
    padding: 3px 9px;
    animation: pulse-badge 2.4s ease-in-out infinite;
}
@keyframes pulse-badge {
    0%,100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    50%      { box-shadow: 0 0 8px 2px rgba(99,102,241,0.35); }
}

/* ── Section labels ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.16rem;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── Video card ──────────────────────────────────────────────────────────── */
.video-card {
    position: relative;
    width: 880px;
    background: #04070e;
    border-radius: 16px;
    border: 1px solid rgba(99,102,241,0.28);
    overflow: hidden;
    box-shadow:
        0 0 0 1px rgba(99,102,241,0.08),
        0 8px 40px rgba(0,0,0,0.7),
        0 0 60px rgba(99,102,241,0.07);
    margin: 0 0 1.2rem 0;
}
.video-card video {
    width: 880px !important;
    height: 495px !important;
    object-fit: cover !important;
    border-radius: 0 !important;
    display: block;
}

/* ── Live badge ──────────────────────────────────────────────────────────── */
.vid-live-badge {
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.14rem;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 4px;
    border: 1px solid;
}
.vid-live-badge.live {
    color: #4ade80;
    background: rgba(74,222,128,0.12);
    border-color: rgba(74,222,128,0.35);
    animation: pulse-badge 2s ease-in-out infinite;
}

/* ── Analytics cards ─────────────────────────────────────────────────────── */
.analytics-card {
    background: #0a0f1e;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 15px 15px 12px 15px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.45);
    transition: border-color 0.25s, box-shadow 0.25s;
    min-height: 220px;
}
.analytics-card:hover {
    border-color: rgba(99,102,241,0.42);
    box-shadow: 0 4px 24px rgba(0,0,0,0.55), 0 0 28px rgba(99,102,241,0.1);
}
.analytics-card [data-testid="stImage"] img {
    width: 100% !important;
    max-height: 190px !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 10px !important;
    display: block;
}
.analytics-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 0.35rem;
    height: 185px;
    color: #1e2d45;
    font-size: 0.72rem;
    letter-spacing: 0.06rem;
    border: 1px dashed rgba(99,102,241,0.12);
    border-radius: 10px;
}

/* ── Metrics row ─────────────────────────────────────────────────────────── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.1rem;
}

/* ── Error state ─────────────────────────────────────────────────────────── */
.error-card {
    background: rgba(127,29,29,0.25);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px;
    padding: 14px 18px;
    color: #fca5a5;
    font-size: 0.85rem;
}

/* ── Streamlit metric override ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0a0f1e !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,102,241,0.38) !important;
    box-shadow: 0 0 22px rgba(99,102,241,0.1) !important;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1rem !important;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
}

/* ── Streamlit tabs override ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 2px;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    margin-bottom: 1rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08rem !important;
    text-transform: uppercase !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 6px 14px !important;
    border: none !important;
    transition: color 0.18s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.1) !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.28); border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Scenario registry  (backend unchanged)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Scenario 1: Single Camera Multi-Person": {
        "runner": run_single,
        "layout": "single",
        "camera_count": 1,
    },
    "Scenario 2: 4-Cam Single Person ReID": {
        "runner": run_reid,
        "layout": "quad",
        "camera_count": 4,
    },
    "Scenario 3: Multi-Cam Multi-Person": {
        "runner": run_multi,
        "layout": "tri",
        "camera_count": 3,
    },
}

# ---------------------------------------------------------------------------
# Shared state: video loop (main thread) → analytics fragment
# ---------------------------------------------------------------------------
_shared_state: dict = {
    "stats": {},
    "error": None,
}


# ---------------------------------------------------------------------------
# Pipeline generator helper
# ---------------------------------------------------------------------------
def _make_pipeline_gen(runner_func):
    """
    Wraps a run_pipeline() function (which returns (frame, stats) each call)
    into an infinite generator that yields (frame, stats) on every next().
    Errors are caught, surfaced via _shared_state, and a fallback frame is
    yielded so the caller's loop never crashes.
    """
    while True:
        try:
            frame, stats = runner_func()
            _shared_state["error"] = None
            if frame is not None and isinstance(frame, np.ndarray):
                yield frame, stats
        except Exception as exc:
            msg = str(exc).strip() or "No message"
            _shared_state["error"] = f"{type(exc).__name__}: {msg}"
            _shared_state["stats"] = {}
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                err_frame,
                f"Pipeline error: {msg[:55]}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (239, 68, 68),
                2,
            )
            yield err_frame, {}


# ---------------------------------------------------------------------------
# Backend helpers  (unchanged logic)
# ---------------------------------------------------------------------------
def _overlay_rect(img, x1, y1, x2, y2, alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 12, 18), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def split_quad(frame):
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    return [
        frame[0:h2, 0:w2],
        frame[0:h2, w2:w],
        frame[h2:h, 0:w2],
        frame[h2:h, w2:w],
    ]


def split_tri(frame):
    h, w = frame.shape[:2]
    h2 = h // 2
    w2 = w // 2
    wq = w // 4
    top = frame[0:h2, :]
    bottom = frame[h2:h, :]
    cam1 = top[:, 0:w2]
    cam2 = top[:, w2:w]
    cam3 = bottom[:, wq:(w - wq)]
    return cam1, cam2, cam3


def _image_stretch(target, image, channels=None, clamp=False):
    """Render image with width='stretch' and fallback for older Streamlit versions."""
    kwargs = {"width": "stretch"}
    if channels is not None:
        kwargs["channels"] = channels
    if clamp:
        kwargs["clamp"] = True

    try:
        target.image(image, **kwargs)
    except TypeError:
        # Streamlit < width='stretch' support
        fallback_kwargs = {"use_container_width": True}
        if channels is not None:
            fallback_kwargs["channels"] = channels
        if clamp:
            fallback_kwargs["clamp"] = True
        target.image(image, **fallback_kwargs)


# ---------------------------------------------------------------------------
# ── UI MODULES ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def render_analytics(stats: dict, scenario_key: str):
    """Heatmap + 2D Map — reads from stats dict only, never from video stream."""
    st.markdown("### Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Heatmap")
        if stats.get("heatmap") is not None:
            _image_stretch(st, stats["heatmap"])
        else:
            st.info("No heatmap available")

    with col2:
        st.markdown("#### 2D Map")
        if stats.get("map") is not None:
            _image_stretch(st, stats["map"])
        else:
            st.info("No map available")


def render_metrics(stats: dict):
    """Horizontal metrics row — Active IDs / Total IDs / Frame."""
    st.markdown(
        '<span class="section-label" style="margin-top:1.4rem;display:block;">Metrics</span>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.metric("\U0001f7e2 Active IDs", int(stats.get("active_ids", 0)))
    with m2:
        st.metric("\U0001f4cb Total IDs",  int(stats.get("total_ids",  0)))
    with m3:
        st.metric("\U0001f39e Frame",       int(stats.get("frame",      0)))


# ---------------------------------------------------------------------------
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.1rem;">'
        '<div class="sidebar-logo">🎯 SURVEILLANCE</div>'
        '<span class="vid-live-badge live" style="font-size:0.55rem;">● Running</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-sub">Multi-Camera Analytics v2</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-heading">Scenario</span>', unsafe_allow_html=True)
    scenario_key = st.radio(
        "scenario_radio",
        options=list(SCENARIOS.keys()),
        label_visibility="collapsed",
        format_func=lambda s: s,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-heading">Settings</span>', unsafe_allow_html=True)

    confidence = st.slider("Confidence", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

# ---------------------------------------------------------------------------
# ── MAIN CONTENT ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown('<div class="dash-wrapper">', unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="dash-header">'
    '<span class="dash-title">Multi-Camera Analytics Dashboard</span>'
    '<span class="dash-badge">Live</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ── VIDEO PLAYER — zero-flicker st.empty() loop ──────────────────────────
with st.container():
    st.markdown("### Live Feed")
    frame_placeholder = st.empty()

# ── ANALYTICS — render into one placeholder to avoid duplicated blocks
analytics_placeholder = st.empty()


def _analytics_panel():
    with analytics_placeholder.container():
        stats = _shared_state.get("stats", {})
        error = _shared_state.get("error")
        if error:
            st.markdown(
                f'<div class="error-card">⚠ Pipeline error: {error}</div>',
                unsafe_allow_html=True,
            )
        render_analytics(stats, scenario_key)
        render_metrics(stats)


_analytics_panel()

st.markdown("</div>", unsafe_allow_html=True)  # close dash-wrapper

# ── PIPELINE INITIALISATION ────────────────────────────────────────────────────────
# Keep one generator per scenario in session_state so pipeline state (model
# weights, video file position, tracking history) persists across frames.
# When the user picks a different scenario the key changes, triggering a full
# Streamlit rerun which re-enters this block and starts a fresh generator.
if (
    "pipeline_key" not in st.session_state
    or st.session_state.pipeline_key != scenario_key
):
    st.session_state.pipeline_key = scenario_key
    st.session_state.pipeline = _make_pipeline_gen(
        SCENARIOS[scenario_key]["runner"]
    )

# ── MAIN VIDEO LOOP ───────────────────────────────────────────────────────────
# Blocking loop — updates video_placeholder in-place on every frame.
# NO st.rerun() is used, so no full-page flicker occurs.
# Streamlit will interrupt this loop naturally when the user interacts
# (e.g. changes scenario), executing a clean page rerun.
while True:
    try:
        frame, stats = next(st.session_state.pipeline)
        _image_stretch(frame_placeholder, frame, channels="BGR", clamp=True)
        _shared_state["stats"] = stats
        _analytics_panel()
    except StopIteration:
        # Generator exhausted (video looped) — restart
        st.session_state.pipeline = _make_pipeline_gen(
            SCENARIOS[scenario_key]["runner"]
        )
    time.sleep(0.03)  # ~30 FPS cap

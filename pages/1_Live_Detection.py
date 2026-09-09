"""
pages/1_Live_Detection.py — SHIELD Live & Upload Detection
===========================================================
Features:
  • Upload dashcam image → instant YOLO + weather + risk analysis
  • Upload dashcam video → frame-by-frame analysis with progress bar
  • Webcam snapshot → one-shot detection from device camera
  • Annotated result image with colour-coded bounding boxes
  • Hazard table, risk scorecard, weather card
  • Feeds ETL pipeline → persists to DB → ingests report into RAG
"""

import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import settings
from etl_pipeline import ETLPipeline
from rag_pipeline import RAGPipeline
from weather_service import WeatherService
from yolo_detector import YOLODetector

st.set_page_config(
    page_title = "Live Detection — SHIELD",
    page_icon  = "🎯",
    layout     = "wide",
)

# ── Shared CSS (re-import for page isolation) ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
:root { --accent:#00e5ff; --bg3:#161b22; --border:#21262d; --muted:#8b949e; --text:#e6edf3; }
html, body, [class*="css"] { background:#0a0c10; color:var(--text); font-family:'Inter',sans-serif; }
h1,h2,h3 { font-family:'Rajdhani',sans-serif !important; letter-spacing:0.04em; }
.stButton>button { background:transparent; border:1px solid var(--accent); color:var(--accent); font-family:'Rajdhani',sans-serif; letter-spacing:0.06em; border-radius:4px; transition:all 0.2s; }
.stButton>button:hover { background:var(--accent); color:#000; box-shadow:0 0 16px rgba(0,229,255,0.35); }
[data-testid="metric-container"] { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:12px 16px !important; }
[data-testid="metric-container"] label { color:var(--muted) !important; font-size:0.72rem; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Share Tech Mono'; color:var(--accent) !important; }
section[data-testid="stSidebar"] { background:#0f1318 !important; border-right:1px solid var(--border); }
.hazard-row { background:#161b22; border:1px solid #21262d; border-radius:6px; padding:8px 14px; margin:4px 0; font-family:'Share Tech Mono'; font-size:0.85rem; }
.weather-card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px 20px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "etl_result"  not in st.session_state: st.session_state.etl_result  = None
if "rag"         not in st.session_state: st.session_state.rag         = RAGPipeline()
if "last_session_id" not in st.session_state: st.session_state.last_session_id = None

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    city      = st.text_input("🌍 City (for weather)", value=settings.WEATHER_CITY)
    speed_kmh = st.slider("🚗 Speed (km/h)", 0, 180, 60, step=5)
    confidence= st.slider("🎯 YOLO Confidence", 0.2, 0.9, settings.YOLO_CONFIDENCE, step=0.05)
    st.divider()
    if st.button("🗑️ Clear Last Result"):
        st.session_state.etl_result = None
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 8px;'>
    <span style='font-family:Rajdhani; font-size:2.2rem; font-weight:700; letter-spacing:0.06em;'>
        🎯 Live Detection
    </span>
    <span style='color:#8b949e; font-size:0.9rem; margin-left:16px;'>
        Upload dashcam footage or use your webcam
    </span>
</div>
""", unsafe_allow_html=True)

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["📷  Image Upload", "🎬  Video Upload", "📹  Webcam Snapshot"])


def _run_image_pipeline(image_bytes: bytes, filename: str) -> None:
    """Run full ETL pipeline on image bytes and store result in session state."""
    with st.spinner("🔍 Running YOLO detection + weather fusion..."):
        etl    = ETLPipeline()
        result = etl.process_image(
            image_bytes  = image_bytes,
            filename     = filename,
            city         = city,
            speed_kmh    = float(speed_kmh),
        )
    st.session_state.etl_result     = result
    st.session_state.last_session_id = result.session_id

    # Ingest report into RAG
    if result.report_text:
        st.session_state.rag.ingest_report(result.report_text, source=result.session_name)

    if result.errors:
        st.error(f"ETL errors: {result.errors}")
    else:
        st.success(f"✅ Analysis complete — Session `{result.session_name}`")


# ── IMAGE TAB ─────────────────────────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader(
        "Drop a dashcam image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
    )

    col_demo, col_run = st.columns([3, 1])
    with col_run:
        run_btn = st.button("▶  Analyse Image", use_container_width=True, key="run_img")

    if run_btn and uploaded:
        _run_image_pipeline(uploaded.read(), uploaded.name)
    elif run_btn and not uploaded:
        st.warning("Please upload an image first.")


# ── VIDEO TAB ─────────────────────────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader(
        "Drop a dashcam video clip here (mp4, avi, mov)",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
    sample_every = st.slider("Sample every N frames", 5, 30, 10, key="sample_slider")

    if st.button("▶  Analyse Video", use_container_width=False, key="run_vid"):
        if vid_file:
            with st.spinner("🎬 Processing video frames... this may take a moment."):
                etl    = ETLPipeline()
                result = etl.process_video(
                    video_bytes  = vid_file.read(),
                    filename     = vid_file.name,
                    city         = city,
                    speed_kmh    = float(speed_kmh),
                    sample_every = sample_every,
                )
            st.session_state.etl_result      = result
            st.session_state.last_session_id = result.session_id
            if result.report_text:
                st.session_state.rag.ingest_report(result.report_text, source=result.session_name)
            st.success(f"✅ Processed {result.frames_processed} frames — Session `{result.session_name}`")
        else:
            st.warning("Please upload a video first.")


# ── WEBCAM TAB ────────────────────────────────────────────────────────────────
with tab_cam:
    st.info("📸 Captures a single frame from your webcam, then runs the full pipeline.")

    cam_img = st.camera_input("Take a snapshot")

    if cam_img and st.button("▶  Analyse Snapshot", key="run_cam"):
        _run_image_pipeline(cam_img.getvalue(), "webcam_snapshot.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

result = st.session_state.etl_result

if result and result.session_id != -1:
    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:Rajdhani; font-size:1.6rem; font-weight:700; margin-bottom:12px;'>
        📋 Analysis Results — <span style='color:#00e5ff;'>{result.session_name}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    risk_colours = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🟠","CRITICAL":"🔴"}
    m1.metric("Risk Level",    f"{risk_colours.get(result.risk_level,'')} {result.risk_level}")
    m2.metric("Total Score",   f"{result.avg_risk:.1f} / 100")
    m3.metric("Peak Score",    f"{result.peak_risk:.1f} / 100")
    m4.metric("Hazards Found", result.hazards_found)
    m5.metric("Frames",        result.frames_processed)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    # ── Annotated image (last frame) ──────────────────────────────────────────
    with left:
        st.markdown("#### 🖼️ Annotated Frame")
        if result.frame_analyses:
            analysis = result.frame_analyses[-1]
            assess   = result.assessments[-1]

            # Re-run annotation if we have the jpeg_b64, else show placeholder
            if analysis.jpeg_b64:
                import base64
                img_bytes = base64.b64decode(analysis.jpeg_b64)
                st.image(img_bytes, use_column_width=True, caption="Last analysed frame")
            else:
                st.info("Annotated frame not available (video mode without frame capture).")

            # Per-signal scores
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("👁️ Vision",   f"{assess.vision_score:.1f}")
            sc2.metric("🌦️ Weather",  f"{assess.weather_score:.1f}")
            sc3.metric("🚗 Speed",    f"{assess.speed_score:.1f}")

    # ── Right panel ───────────────────────────────────────────────────────────
    with right:
        # Hazard table
        st.markdown("#### ⚠️ Detected Hazards")
        if result.frame_analyses:
            detections = result.frame_analyses[-1].detections
            if detections:
                for det in sorted(detections, key=lambda d: -d.danger_weight):
                    colour = (
                        "#ff4655" if det.danger_weight >= 0.8 else
                        "#ff8c00" if det.danger_weight >= 0.6 else
                        "#ffd700" if det.danger_weight >= 0.4 else
                        "#39d353"
                    )
                    st.markdown(f"""
                    <div class='hazard-row'>
                        <span style='color:{colour}; font-weight:700;'>▶ {det.label.upper()}</span>
                        &nbsp;&nbsp;conf: {det.confidence:.0%}
                        &nbsp;&nbsp;danger: {det.danger_weight:.1f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#8b949e; padding:8px;'>✅ No road hazards detected</div>", unsafe_allow_html=True)

        # Weather card
        st.markdown("#### 🌦️ Weather Conditions")
        if result.weather:
            w = result.weather
            st.markdown(f"""
            <div class='weather-card'>
                <div style='font-size:1.5rem; margin-bottom:8px;'>{w.get('condition','—')} &nbsp;·&nbsp; {w.get('temperature','—')}°C</div>
                <div style='font-size:0.82rem; color:#8b949e; line-height:1.8;'>
                    💧 Humidity: {w.get('humidity','—')}%<br>
                    💨 Wind: {w.get('wind_speed','—')} m/s<br>
                    👁️ Visibility: {round(w.get('visibility',10000)/1000,1)} km<br>
                    🌧️ Rain (1h): {w.get('rain_1h', 0)} mm<br>
                    ⚠️ Weather Risk: <strong style='color:#00e5ff;'>{w.get('weather_risk','—')}/100</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("#### 💡 Recommendations")
        if result.assessments:
            for rec in result.assessments[-1].recommendations:
                st.markdown(f"""
                <div style='background:#0f1318; border-left:3px solid #00e5ff; padding:8px 14px;
                            border-radius:0 6px 6px 0; margin:4px 0; font-size:0.85rem;'>
                    {rec}
                </div>
                """, unsafe_allow_html=True)

    # ── Alerts ────────────────────────────────────────────────────────────────
    if result.assessments and result.assessments[-1].alerts:
        st.markdown("#### 🚨 Active Alerts")
        for alert in result.assessments[-1].alerts:
            colour = "#ff4655" if alert["type"] == "CRITICAL" else "#ff8c00"
            st.markdown(f"""
            <div style='background:rgba(255,70,85,0.08); border-left:4px solid {colour};
                        padding:10px 16px; border-radius:0 6px 6px 0; margin:4px 0; font-size:0.88rem;'>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)

    # ── Trip report download ───────────────────────────────────────────────────
    if result.report_text:
        st.markdown("---")
        st.download_button(
            "📄 Download Trip Report",
            data      = result.report_text,
            file_name = f"{result.session_name}_report.txt",
            mime      = "text/plain",
        )

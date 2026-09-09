"""
pages/2_Risk_Dashboard.py — SHIELD Risk Dashboard
==================================================
Features:
  • Live risk score gauge (Plotly)
  • Per-signal breakdown bar chart (vision / weather / speed)
  • Frame-by-frame risk timeline (for video sessions)
  • Hazard frequency pie chart
  • Weather condition panel
  • Pulls from st.session_state.etl_result or DB history
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from db import get_session
from models import TripSession, RiskEvent, HazardDetection, WeatherSnapshot

st.set_page_config(
    page_title = "Risk Dashboard — SHIELD",
    page_icon  = "⚡",
    layout     = "wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
:root{--accent:#00e5ff;--bg:#0a0c10;--bg2:#0f1318;--bg3:#161b22;--border:#21262d;--muted:#8b949e;--text:#e6edf3;}
html,body,[class*="css"]{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Rajdhani',sans-serif !important;letter-spacing:0.04em;}
section[data-testid="stSidebar"]{background:var(--bg2) !important;border-right:1px solid var(--border);}
[data-testid="metric-container"]{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px 16px !important;}
[data-testid="metric-container"] label{color:var(--muted) !important;font-size:0.72rem;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Share Tech Mono';color:var(--accent) !important;}
.stButton>button{background:transparent;border:1px solid var(--accent);color:var(--accent);font-family:'Rajdhani';border-radius:4px;transition:all 0.2s;}
.stButton>button:hover{background:var(--accent);color:#000;}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor = "#0a0c10",
    plot_bgcolor  = "#0f1318",
    font          = dict(color="#e6edf3", family="Share Tech Mono"),

    yaxis         = dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin        = dict(l=32, r=32, t=40, b=32),
)


def risk_colour(score: float) -> str:
    if score >= 80: return "#ff4655"
    if score >= 60: return "#ff8c00"
    if score >= 30: return "#ffd700"
    return "#39d353"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 8px;'>
    <span style='font-family:Rajdhani;font-size:2.2rem;font-weight:700;letter-spacing:0.06em;'>
        ⚡ Risk Dashboard
    </span>
</div>
""", unsafe_allow_html=True)

# ── Session selector ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Session")
    with get_session() as db:
        sessions = db.query(TripSession).order_by(TripSession.started_at.desc()).limit(30).all()

    if not sessions:
        st.info("No sessions yet. Run detection first.")
        st.stop()

    session_labels = {f"[{s.id}] {s.session_name}  ({s.risk_level or '?'})": s.id for s in sessions}
    chosen_label   = st.selectbox("Select session", list(session_labels.keys()))
    chosen_id      = session_labels[chosen_label]

    # If there's a fresh result in session_state, auto-select it
    if st.session_state.get("last_session_id"):
        fresh_label = next(
            (k for k, v in session_labels.items() if v == st.session_state.last_session_id), None
        )
        if fresh_label:
            chosen_id = st.session_state.last_session_id

    st.divider()
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Load session data ─────────────────────────────────────────────────────────
with get_session() as db:
    session   = db.query(TripSession).filter(TripSession.id == chosen_id).first()
    risk_rows = db.query(RiskEvent).filter(RiskEvent.session_id == chosen_id).all()
    det_rows  = db.query(HazardDetection).filter(HazardDetection.session_id == chosen_id).all()
    wx_rows   = db.query(WeatherSnapshot).filter(WeatherSnapshot.session_id == chosen_id).all()

if not session:
    st.error("Session not found.")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
risk_icons = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🟠","CRITICAL":"🔴"}
m1.metric("Risk Level",     f"{risk_icons.get(session.risk_level,'')} {session.risk_level or '?'}")
m2.metric("Avg Risk Score", f"{session.avg_risk_score or 0:.1f} / 100")
m3.metric("Peak Score",     f"{session.peak_risk_score or 0:.1f} / 100")
m4.metric("Total Hazards",  len(det_rows))
m5.metric("Frames",         session.frame_count or 0)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Gauge + Signal Breakdown ──────────────────────────────────────────
g_col, b_col = st.columns([1, 1.6])

with g_col:
    score = session.avg_risk_score or 0
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 50, "increasing": {"color": "#ff4655"}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":   {"color": risk_colour(score)},
            "steps": [
                {"range": [0,  30], "color": "#0d1f14"},
                {"range": [30, 60], "color": "#1f1a0d"},
                {"range": [60, 80], "color": "#1f130d"},
                {"range": [80,100], "color": "#1f0d0e"},
            ],
            "threshold": {"line": {"color": "#ff4655", "width": 3}, "value": 80},
        },
        title = {"text": "AVG RISK SCORE", "font": {"size": 14, "color": "#8b949e"}},
        number = {"suffix": "/100", "font": {"size": 36, "color": risk_colour(score)}},
    ))
    fig_gauge.update_layout(**_DARK, height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

with b_col:
    if risk_rows:
        last  = risk_rows[-1]
        df_bar = pd.DataFrame({
            "Signal":  ["👁️ Vision", "🌦️ Weather", "🚗 Speed"],
            "Score":   [last.vision_score or 0, last.weather_score or 0, last.speed_score or 0],
            "Colour":  [
                risk_colour(last.vision_score  or 0),
                risk_colour(last.weather_score or 0),
                risk_colour(last.speed_score   or 0),
            ],
        })
        fig_bar = go.Figure(go.Bar(
            x            = df_bar["Score"],
            y            = df_bar["Signal"],
            orientation  = "h",
            marker_color = df_bar["Colour"],
            text         = df_bar["Score"].apply(lambda x: f"{x:.1f}"),
            textposition = "outside",
        ))
        fig_bar.update_layout(
          title="Signal Breakdown (last frame)",
          height=300,
          xaxis=dict(range=[0,110]),
    template="plotly_dark"
)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No risk events in this session.")

# ── Row 2: Timeline + Pie ─────────────────────────────────────────────────────
t_col, p_col = st.columns([2, 1])

with t_col:
    if len(risk_rows) > 1:
        df_time = pd.DataFrame({
            "Frame": [r.frame_id for r in risk_rows],
            "Total": [r.total_score for r in risk_rows],
            "Vision": [r.vision_score or 0 for r in risk_rows],
            "Weather":[r.weather_score or 0 for r in risk_rows],
        })
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_time["Frame"], y=df_time["Total"],
            mode="lines", name="Total Risk",
            line=dict(color="#00e5ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
        ))
        fig_line.add_trace(go.Scatter(
            x=df_time["Frame"], y=df_time["Vision"],
            mode="lines", name="Vision",
            line=dict(color="#ff4655", width=1, dash="dot"),
        ))
        fig_line.add_trace(go.Scatter(
            x=df_time["Frame"], y=df_time["Weather"],
            mode="lines", name="Weather",
            line=dict(color="#ffd700", width=1, dash="dot"),
        ))
        # Threshold lines
        fig_line.add_hline(y=80, line_dash="dash", line_color="#ff4655", opacity=0.4, annotation_text="CRITICAL")
        fig_line.add_hline(y=60, line_dash="dash", line_color="#ff8c00", opacity=0.4, annotation_text="HIGH")
        fig_line.update_layout(**_DARK, title="Risk Score Timeline", height=300,
                               yaxis=dict(range=[0, 110]))
        fig_line.update_layout(legend=dict(bgcolor="#0f1318", bordercolor="#21262d", borderwidth=1))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Timeline requires multiple frames (upload a video).")

with p_col:
    if det_rows:
        label_counts = {}
        for d in det_rows:
            label_counts[d.label] = label_counts.get(d.label, 0) + 1

        fig_pie = go.Figure(go.Pie(
            labels  = list(label_counts.keys()),
            values  = list(label_counts.values()),
            hole    = 0.5,
            marker  = dict(colors=px.colors.qualitative.Dark24),
            textfont= dict(size=11),
        ))
        fig_pie.update_layout(**_DARK, title="Hazard Distribution", height=300,
                              showlegend=True,
                              legend=dict(bgcolor="#0f1318", bordercolor="#21262d"))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No hazards detected in this session.")

# ── Weather panel ─────────────────────────────────────────────────────────────
if wx_rows:
    wx = wx_rows[-1]
    st.markdown("---")
    st.markdown("### 🌦️ Weather Conditions")

    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wc1.metric("Condition",   wx.condition or "—")
    wc2.metric("Temperature", f"{wx.temperature or 0:.1f} °C")
    wc3.metric("Humidity",    f"{wx.humidity or 0} %")
    wc4.metric("Visibility",  f"{round((wx.visibility or 10000)/1000,1)} km")
    wc5.metric("Weather Risk",f"{wx.weather_risk or 0:.1f} / 100")

# ── Recommendations ───────────────────────────────────────────────────────────
if risk_rows and risk_rows[-1].recommendation:
    st.markdown("---")
    st.markdown("### 💡 Safety Recommendations")
    for rec in risk_rows[-1].recommendation.split(" | "):
        if rec.strip():
            st.markdown(f"""
            <div style='background:#0f1318;border-left:3px solid #00e5ff;padding:8px 14px;
                        border-radius:0 6px 6px 0;margin:4px 0;font-size:0.85rem;'>
                {rec.strip()}
            </div>
            """, unsafe_allow_html=True)

# ── Raw detections table ──────────────────────────────────────────────────────
with st.expander("🔍 Raw Hazard Detections"):
    if det_rows:
        df_det = pd.DataFrame([{
            "Frame":   d.frame_id,
            "Label":   d.label,
            "Confidence": f"{d.confidence:.0%}",
            "Danger Weight": d.danger_weight,
            "BBox":    f"[{d.bbox_x1},{d.bbox_y1},{d.bbox_x2},{d.bbox_y2}]",
        } for d in det_rows])
        st.dataframe(df_det, use_container_width=True, hide_index=True)
    else:
        st.write("No detections.")

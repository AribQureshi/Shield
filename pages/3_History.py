"""
pages/3_History.py — SHIELD Trip History
=========================================
Features:
  • All past sessions list with risk level badges
  • Aggregate stats: total trips, avg risk, most common hazard
  • Per-session expandable detail: hazards, weather, recommendations
  • Download trip report per session
  • Delete session button
  • Plotly bar chart — risk scores across sessions
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import timezone

from config import settings
from db import get_session
from models import TripSession, HazardDetection, RiskEvent, WeatherSnapshot

st.set_page_config(
    page_title = "Trip History — SHIELD",
    page_icon  = "📊",
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
.session-card{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:14px 18px;margin:6px 0;transition:border-color 0.2s;}
.session-card:hover{border-color:var(--accent);}
</style>
""", unsafe_allow_html=True)

_DARK = dict(
    paper_bgcolor="#0a0c10", plot_bgcolor="#0f1318",
    font=dict(color="#e6edf3", family="Share Tech Mono"),
    xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
    margin=dict(l=32, r=32, t=40, b=32),
)

def risk_colour(level: str) -> str:
    return {"LOW":"#39d353","MEDIUM":"#ffd700","HIGH":"#ff8c00","CRITICAL":"#ff4655"}.get(level or "", "#8b949e")

def risk_badge(level: str) -> str:
    colours = {"LOW":("rgba(57,211,83,0.15)","#39d353"),
               "MEDIUM":("rgba(255,215,0,0.15)","#ffd700"),
               "HIGH":("rgba(255,140,0,0.15)","#ff8c00"),
               "CRITICAL":("rgba(255,70,85,0.15)","#ff4655")}
    bg, fg = colours.get(level or "LOW", ("#21262d","#8b949e"))
    return (f"<span style='background:{bg};color:{fg};border:1px solid {fg};"
            f"padding:2px 10px;border-radius:20px;font-family:Share Tech Mono;"
            f"font-size:0.75rem;'>{level or '?'}</span>")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 8px;'>
    <span style='font-family:Rajdhani;font-size:2.2rem;font-weight:700;letter-spacing:0.06em;'>
        📊 Trip History
    </span>
</div>
""", unsafe_allow_html=True)

# ── Load all sessions ─────────────────────────────────────────────────────────
with get_session() as db:
    sessions = db.query(TripSession).order_by(TripSession.started_at.desc()).all()

if not sessions:
    st.info("No trips recorded yet. Go to **Live Detection** to analyse your first dashcam footage.")
    st.stop()

# ── Aggregate stats ───────────────────────────────────────────────────────────
total_trips   = len(sessions)
avg_risk      = round(sum(s.avg_risk_score or 0 for s in sessions) / total_trips, 1)
critical_trips= sum(1 for s in sessions if s.risk_level == "CRITICAL")
total_hazards = sum(s.frame_count or 0 for s in sessions)

with get_session() as db:
    all_dets = db.query(HazardDetection).all()

label_counts: dict[str, int] = {}
for d in all_dets:
    label_counts[d.label] = label_counts.get(d.label, 0) + 1
top_hazard = max(label_counts, key=label_counts.get) if label_counts else "None"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Trips",     total_trips)
m2.metric("Avg Risk Score",  f"{avg_risk} / 100")
m3.metric("Critical Trips",  critical_trips)
m4.metric("Top Hazard",      top_hazard.capitalize())

st.markdown("<br>", unsafe_allow_html=True)

# ── Risk across sessions chart ────────────────────────────────────────────────
df_sess = pd.DataFrame([{
    "Session":  s.session_name[:20],
    "Avg Risk": s.avg_risk_score or 0,
    "Peak Risk":s.peak_risk_score or 0,
    "Level":    s.risk_level or "LOW",
} for s in reversed(sessions[-15:])])   # last 15, oldest first

fig_hist = go.Figure()
fig_hist.add_trace(go.Bar(
    x=df_sess["Session"], y=df_sess["Avg Risk"],
    name="Avg Risk",
    marker_color=[risk_colour(l) for l in df_sess["Level"]],
    opacity=0.85,
))
fig_hist.add_trace(go.Scatter(
    x=df_sess["Session"], y=df_sess["Peak Risk"],
    name="Peak Risk", mode="lines+markers",
    line=dict(color="#00e5ff", width=2),
    marker=dict(size=6),
))
fig_hist.add_hline(y=80, line_dash="dash", line_color="#ff4655", opacity=0.4)
fig_hist.update_layout(**_DARK, title="Risk Scores Across Sessions", height=280,
                       barmode="overlay", xaxis_tickangle=-30)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 Filters")
    filter_level = st.multiselect(
        "Risk Level",
        ["LOW","MEDIUM","HIGH","CRITICAL"],
        default=["LOW","MEDIUM","HIGH","CRITICAL"]
    )
    sort_by = st.selectbox("Sort by", ["Newest first", "Highest risk", "Most hazards"])

    st.divider()

    if st.button("🗑️ Delete ALL sessions", type="secondary"):
        with get_session() as db:
            sessions = db.query(TripSession).all()
            for s in sessions:
                db.delete(s)
            db.commit()

        st.success("All sessions deleted.")
        st.rerun()

# ── Filter + sort ─────────────────────────────────────────────────────────────
filtered = [s for s in sessions if (s.risk_level or "LOW") in filter_level]
if sort_by == "Highest risk":
    filtered.sort(key=lambda s: s.avg_risk_score or 0, reverse=True)
elif sort_by == "Most hazards":
    filtered.sort(key=lambda s: s.frame_count or 0, reverse=True)

st.markdown(f"### 🗂️ Sessions ({len(filtered)} shown)")

# ── Session cards ─────────────────────────────────────────────────────────────
for sess in filtered:
    ts = sess.started_at
    if ts and ts.tzinfo is None:
        ts_str = ts.strftime("%d %b %Y  %H:%M")
    else:
        ts_str = ts.strftime("%d %b %Y  %H:%M") if ts else "—"

    with st.expander(
        f"{'🔴' if sess.risk_level=='CRITICAL' else '🟠' if sess.risk_level=='HIGH' else '🟡' if sess.risk_level=='MEDIUM' else '🟢'}"
        f"  {sess.session_name}   ·   Risk {sess.avg_risk_score or 0:.1f}/100   ·   {ts_str}"
    ):
        col_l, col_r = st.columns([1.5, 1])

        with col_l:
            st.markdown(f"**Mode:** `{sess.mode}`  &nbsp; **City:** `{sess.city or '—'}`")
            st.markdown(f"**Frames analysed:** `{sess.frame_count}`  &nbsp; **Avg risk:** `{sess.avg_risk_score or 0:.1f}`  &nbsp; **Peak:** `{sess.peak_risk_score or 0:.1f}`")
            st.markdown(f"**Risk level:** {risk_badge(sess.risk_level)}", unsafe_allow_html=True)

            # Hazards in this session
            with get_session() as db:
                dets = db.query(HazardDetection).filter(HazardDetection.session_id == sess.id).all()
            if dets:
                det_counts: dict[str, int] = {}
                for d in dets:
                    det_counts[d.label] = det_counts.get(d.label, 0) + 1
                st.markdown("**Hazards:** " + "  ·  ".join(f"`{k}` ×{v}" for k,v in sorted(det_counts.items())))
            else:
                st.markdown("**Hazards:** None detected")

        with col_r:
            # Weather summary
            with get_session() as db:
                wx = db.query(WeatherSnapshot).filter(WeatherSnapshot.session_id == sess.id).first()
            if wx:
                st.markdown(f"""
                <div style='background:#0f1318;border:1px solid #21262d;border-radius:6px;padding:10px 14px;font-size:0.82rem;line-height:1.8;'>
                    🌡️ {wx.temperature or 0:.1f}°C &nbsp;·&nbsp; {wx.condition or '—'}<br>
                    👁️ Visibility: {round((wx.visibility or 10000)/1000,1)} km<br>
                    💨 Wind: {wx.wind_speed or 0} m/s<br>
                    ⚠️ Weather risk: <strong style='color:#00e5ff;'>{wx.weather_risk or 0:.1f}/100</strong>
                </div>
                """, unsafe_allow_html=True)

            # Last recommendation
            with get_session() as db:
                ev = db.query(RiskEvent).filter(RiskEvent.session_id == sess.id).order_by(RiskEvent.id.desc()).first()
            if ev and ev.recommendation:
                first_rec = ev.recommendation.split(" | ")[0]
                st.markdown(f"""
                <div style='background:#0f1318;border-left:3px solid #00e5ff;padding:8px 12px;
                            border-radius:0 4px 4px 0;margin-top:8px;font-size:0.82rem;'>
                    💡 {first_rec}
                </div>
                """, unsafe_allow_html=True)

        # Report download
        report_path = Path(settings.ETL_REPORTS_DIR) / f"{sess.session_name}.txt"
        if report_path.exists():
            st.download_button(
                "📄 Download Report",
                data      = report_path.read_text(),
                file_name = report_path.name,
                mime      = "text/plain",
                key       = f"dl_{sess.id}",
            )

        # Delete this session
        if st.button("🗑️ Delete", key=f"del_{sess.id}", type="secondary"):
            with get_session() as db:
                s = db.query(TripSession).filter(TripSession.id == sess.id).first()
                if s:
                    db.delete(s)
                    db.commit()
            st.rerun()

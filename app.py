"""
app.py — SHIELD Streamlit Application Entry Point
==================================================
Run with:  streamlit run app.py

This is the home / landing page.
All feature pages live in pages/ directory (auto-discovered by Streamlit).
"""

import os

import streamlit as st

from config import settings
from db import check_connection, init_db

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "SHIELD — Road Safety AI",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0a0c10;
    --bg2:       #0f1318;
    --bg3:       #161b22;
    --border:    #21262d;
    --accent:    #00e5ff;
    --accent2:   #ff4655;
    --accent3:   #ffd700;
    --green:     #39d353;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --font-head: 'Rajdhani', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
    --font-body: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background:  var(--bg);
    color:       var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: var(--font-head) !important;
    letter-spacing: 0.04em;
    color: var(--text) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:    var(--bg3);
    border:        1px solid var(--border);
    border-radius: 8px;
    padding:       12px 16px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: var(--font-mono); color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
    background:    transparent;
    border:        1px solid var(--accent);
    color:         var(--accent);
    font-family:   var(--font-head);
    font-size:     1rem;
    letter-spacing:0.06em;
    border-radius: 4px;
    transition:    all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent);
    color:      #000;
    box-shadow: 0 0 16px rgba(0,229,255,0.35);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg2); border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"]      { font-family: var(--font-head); color: var(--muted); letter-spacing: 0.05em; }
.stTabs [aria-selected="true"]    { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border:        1px dashed var(--border);
    border-radius: 8px;
    background:    var(--bg2);
}

/* ── Selectbox / input ── */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Custom card component ── */
.shield-card {
    background:    var(--bg3);
    border:        1px solid var(--border);
    border-radius: 8px;
    padding:       20px 24px;
    margin-bottom: 12px;
}
.shield-card:hover { border-color: var(--accent); transition: border-color 0.2s; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.risk-low      { background: rgba(57,211,83,0.15);  color: #39d353; border: 1px solid #39d353; }
.risk-medium   { background: rgba(255,215,0,0.15);  color: #ffd700; border: 1px solid #ffd700; }
.risk-high     { background: rgba(255,140,0,0.15);  color: #ff8c00; border: 1px solid #ff8c00; }
.risk-critical { background: rgba(255,70,85,0.15);  color: #ff4655; border: 1px solid #ff4655; }

/* ── Alert box ── */
.shield-alert {
    padding: 10px 16px;
    border-radius: 6px;
    border-left: 4px solid var(--accent2);
    background: rgba(255,70,85,0.08);
    font-size: 0.88rem;
    margin: 6px 0;
}

/* ── Mono stat ── */
.mono-stat {
    font-family: var(--font-mono);
    color: var(--accent);
    font-size: 2rem;
    line-height: 1;
}

/* ── Divider ── */
.shield-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Init DB (idempotent) ──────────────────────────────────────────────────────
init_db()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 24px;'>
        <div style='font-family:Rajdhani; font-size:2rem; font-weight:700; color:#00e5ff; letter-spacing:0.15em;'>
            🛡️ SHIELD
        </div>
        <div style='font-size:0.7rem; color:#8b949e; letter-spacing:0.2em; text-transform:uppercase;'>
            Smart Hazard Intelligence &<br>Early Life-saving Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # DB status
    db_ok = check_connection()
    st.markdown(
        f"**Database** &nbsp; {'🟢 Connected' if db_ok else '🔴 Error'}",
        unsafe_allow_html=True,
    )

    # API key status
    has_groq    = bool(settings.GROQ_API_KEY)
    has_weather = bool(settings.WEATHER_API_KEY)
    st.markdown(f"**Groq API** &nbsp; {'🟢 Set' if has_groq else '🔴 Missing'}", unsafe_allow_html=True)
    st.markdown(f"**Weather API** &nbsp; {'🟢 Set' if has_weather else '🔴 Missing'}", unsafe_allow_html=True)

    st.divider()
    st.caption(f"v{settings.APP_VERSION} · {settings.ENVIRONMENT}")


# ── Home Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 16px;'>
    <div style='font-family:Rajdhani; font-size:3.2rem; font-weight:700; letter-spacing:0.06em; line-height:1.1;'>
        Smart Hazard Intelligence<br>
        <span style='color:#00e5ff;'>& Early Life-saving Detection</span>
    </div>
    <div style='color:#8b949e; font-size:1rem; margin-top:12px; max-width:600px;'>
        Real-time road hazard detection powered by YOLOv8, live weather fusion,
        and an AI risk engine — giving drivers actionable safety intelligence, frame by frame.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='shield-divider'></div>", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='shield-card'>
        <div style='font-size:1.8rem;'>🎯</div>
        <div style='font-family:Rajdhani; font-size:1.1rem; font-weight:600; margin:8px 0 4px;'>Live Detection</div>
        <div style='color:#8b949e; font-size:0.82rem;'>YOLOv8 real-time hazard detection from webcam or uploaded dashcam footage</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='shield-card'>
        <div style='font-size:1.8rem;'>⚡</div>
        <div style='font-family:Rajdhani; font-size:1.1rem; font-weight:600; margin:8px 0 4px;'>Risk Dashboard</div>
        <div style='color:#8b949e; font-size:0.82rem;'>Fused vision + weather + speed risk scores with real-time charts and alerts</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='shield-card'>
        <div style='font-size:1.8rem;'>📊</div>
        <div style='font-family:Rajdhani; font-size:1.1rem; font-weight:600; margin:8px 0 4px;'>Trip History</div>
        <div style='color:#8b949e; font-size:0.82rem;'>Full session history, hazard breakdowns, and downloadable reports</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='shield-card'>
        <div style='font-size:1.8rem;'>🤖</div>
        <div style='font-family:Rajdhani; font-size:1.1rem; font-weight:600; margin:8px 0 4px;'>Safety Chatbot</div>
        <div style='color:#8b949e; font-size:0.82rem;'>RAG-powered AI that answers questions about your trips and road conditions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='shield-divider'></div>", unsafe_allow_html=True)

# ── Quick start ───────────────────────────────────────────────────────────────
st.markdown("### ⚡ Quick Start")
q1, q2 = st.columns([2, 1])
with q1:
    st.markdown("""
    1. Set `WEATHER_API_KEY` and optionally `OPENAI_API_KEY` in your `.env` file
    2. Navigate to **Live Detection** → upload a dashcam image or video
    3. View the fused risk score on the **Risk Dashboard**
    4. Ask questions about your trip in the **Safety Chatbot**
    """)
with q2:
    if not has_groq or not has_weather:        
        st.warning("⚠️ Some API keys missing — see sidebar for status.")
    else:
        st.success("✅ All systems configured and ready.")

    if st.button("🚀 Start Detection →", use_container_width=True):
        st.switch_page("pages/1_Live_Detection.py")
st.write("DEBUG GROQ:", settings.GROQ_API_KEY)
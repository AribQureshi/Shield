"""
pages/4_Chat.py — SHIELD Safety Chatbot
=========================================
Features:
  • RAG-powered chat grounded on trip reports
  • Persistent conversation history per page session
  • Session selector — load any past trip into RAG context
  • Suggested prompts for quick start
  • Clear history / clear documents buttons
  • Chunk count indicator + ready state
"""

import streamlit as st
from pathlib import Path

from config import settings
from db import get_session
from models import TripSession
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title = "Safety Chatbot — SHIELD",
    page_icon  = "🤖",
    layout     = "wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
:root{--accent:#00e5ff;--bg:#0a0c10;--bg2:#0f1318;--bg3:#161b22;--border:#21262d;--muted:#8b949e;--text:#e6edf3;}
html,body,[class*="css"]{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Rajdhani',sans-serif !important;letter-spacing:0.04em;}
section[data-testid="stSidebar"]{background:var(--bg2) !important;border-right:1px solid var(--border);}
.stButton>button{background:transparent;border:1px solid var(--accent);color:var(--accent);font-family:'Rajdhani';border-radius:4px;transition:all 0.2s;}
.stButton>button:hover{background:var(--accent);color:#000;}
[data-testid="stChatMessage"]{background:var(--bg3) !important;border:1px solid var(--border);border-radius:8px;margin:6px 0;}
.stChatInput textarea{background:var(--bg2) !important;border:1px solid var(--border) !important;color:var(--text) !important;border-radius:6px !important;}
.chip{display:inline-block;background:var(--bg3);border:1px solid var(--border);border-radius:20px;
      padding:4px 14px;font-size:0.78rem;color:var(--muted);cursor:pointer;margin:3px;transition:all 0.15s;}
.chip:hover{border-color:var(--accent);color:var(--accent);}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag"           not in st.session_state: st.session_state.rag           = RAGPipeline()
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "chat_session_id" not in st.session_state: st.session_state.chat_session_id = None

rag: RAGPipeline = st.session_state.rag

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Load Trip Report")

    with get_session() as db:
        sessions = db.query(TripSession).order_by(TripSession.started_at.desc()).limit(30).all()

    if sessions:
        sess_labels = {f"[{s.id}] {s.session_name}": s for s in sessions}
        chosen_label = st.selectbox("Select trip to load into chat", ["— none —"] + list(sess_labels.keys()))

        if chosen_label != "— none —" and st.button("📥 Load into RAG"):
            sess = sess_labels[chosen_label]
            report_path = Path(settings.ETL_REPORTS_DIR) / f"{sess.session_name}.txt"

            if report_path.exists():
                n = rag.ingest_report(report_path.read_text(), source=sess.session_name)

                st.session_state.chat_session_id = sess.id
                st.session_state.rag_loaded = True   # ✅ ADD THIS LINE

                st.success(f"Loaded {n} chunks from '{sess.session_name}'")
            else:
             st.warning("Report file not found. Re-run detection to regenerate.")
    else:
        st.info("No trips yet. Run detection first.")

    st.divider()

    # Status
    st.markdown(f"**Chunks loaded:** `{rag.chunk_count}`")
    st.markdown(f"**Chat turns:**    `{rag.history_length}`")
    st.markdown(
        "**RAG ready:** " + ("🟢 Yes" if rag.chunk_count > 0 else "🔴 No — load a trip"),
        unsafe_allow_html=True,
    )

    st.divider()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear chat"):
            rag.clear_history()
            st.session_state.chat_messages = []
            st.rerun()
    with col_c2:
        if st.button("🗑️ Clear docs"):
            rag.clear_documents()
            st.session_state.chat_messages = []
            st.rerun()

        if not settings.GROQ_API_KEY:       
            st.warning("⚠️ OPENAI_API_KEY not set.\nChatbot requires an OpenAI key.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 8px;'>
    <span style='font-family:Rajdhani;font-size:2.2rem;font-weight:700;letter-spacing:0.06em;'>
        🤖 Safety Chatbot
    </span>
    <span style='color:#8b949e;font-size:0.9rem;margin-left:16px;'>
        Ask SHIELD anything about your trip, road conditions, or safety advice
    </span>
</div>
""", unsafe_allow_html=True)

# ── Suggested prompts ─────────────────────────────────────────────────────────
suggestions = [
    "What hazards were detected in my trip?",
    "How dangerous was the weather?",
    "What safety recommendations do you have?",
    "What was my overall risk score?",
    "Were any pedestrians detected?",
    "How can I drive safer in rain?",
]

st.markdown("**Quick prompts:**")
cols = st.columns(3)
for i, prompt in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(prompt, key=f"sug_{i}", use_container_width=True):
            st.session_state._pending_prompt = prompt

# ── Chat messages ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
# Handle pending prompt from quick buttons
pending = st.session_state.pop("_pending_prompt", None)
user_input = st.chat_input("Ask about your trip…") or pending

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            response = rag.ask(
                query      = user_input,
                session_id = st.session_state.chat_session_id,
            )
        st.markdown(response)

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.chat_messages:
    st.markdown("""
    <div style='text-align:center;padding:60px 0;color:#8b949e;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-family:Rajdhani;font-size:1.2rem;margin:12px 0 6px;'>SHIELD is ready to help</div>
        <div style='font-size:0.85rem;'>
            Load a trip report from the sidebar, then ask a question above<br>
            or click one of the quick prompts to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

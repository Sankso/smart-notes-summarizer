#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Notes Summarizer — Streamlit Web UI (Clean Edition)
Run: streamlit run streamlit_app.py
"""

import os
import sys
import time
import tempfile
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from agent.agent import SmartSummarizerAgent

# ── Page Config ──
st.set_page_config(
    page_title="Smart Notes Summarizer",
    page_icon="🧠",
    layout="centered",
)

# ── Minimal Dark CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0f1117;
    --card: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #8b5cf6;
}

.stApp { 
    font-family: 'Inter', sans-serif !important; 
    background: var(--bg) !important;
}
h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--text) !important; }

/* title */
.app-title {
    font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem;
    background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-sub { font-size: 0.85rem; color: var(--muted) !important; margin-bottom: 1.5rem; }

/* cards */
.result-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
}
.result-card h4 { margin: 0 0 0.6rem 0; font-size: 0.85rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.05em; }
.result-card p, .result-card span { color: var(--text) !important; }

/* chips */
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
    padding: 0.25rem 0.7rem; border-radius: 50px; font-size: 0.78rem; font-weight: 500;
    background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.25); color: #c4b5fd !important;
}
.chip-cyan { background: rgba(6,182,212,0.12); border-color: rgba(6,182,212,0.25); color: #67e8f9 !important; }
.chip-green { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.25); color: #6ee7b7 !important; }

/* stats row */
.stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat {
    flex: 1; min-width: 100px; text-align: center; padding: 0.8rem;
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: var(--accent) !important; }
.stat-lbl { font-size: 0.7rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* route badge */
.badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 50px;
    font-size: 0.72rem; font-weight: 600;
}
.badge-local { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7 !important; }
.badge-gemini { background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.3); color: #c4b5fd !important; }

/* button */
div.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; width: 100%;
}
div.stButton > button:hover { box-shadow: 0 6px 20px rgba(139,92,246,0.3) !important; }
</style>
""", unsafe_allow_html=True)

# ── Log Capture ──
class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    def emit(self, record):
        self.logs.append(self.format(record))

# ── Cache Model ──
@st.cache_resource(show_spinner="Loading models...")
def load_agent():
    return SmartSummarizerAgent()

# ──────────────────────────────────────
#  UI
# ──────────────────────────────────────

st.markdown('<div class="app-title">🧠 Smart Notes Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Paste text or upload a PDF → get an AI-generated summary with keywords</div>', unsafe_allow_html=True)

# Input
tab_text, tab_pdf = st.tabs(["📝 Text", "📄 PDF"])

with tab_text:
    input_text = st.text_area("Paste your text", height=180, placeholder="Paste lecture notes, articles, or any text here...", label_visibility="collapsed")

with tab_pdf:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

# Controls
col1, col2 = st.columns([3, 1])
with col1:
    length = st.select_slider("Length", options=["short", "normal", "long"], value="normal")
with col2:
    st.write("")
    run = st.button("🚀 Summarize")

# ──────────────────────────────────────
#  Process
# ──────────────────────────────────────

if run:
    has_text = bool(input_text and input_text.strip())
    has_pdf = uploaded_file is not None

    if not has_text and not has_pdf:
        st.warning("Please provide some text or upload a PDF.")
        st.stop()

    # Capture logs
    log_handler = LogCapture()
    log_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    for name in ["", "agent.agent", "agent.brain", "agent.executor", "agent.planner", "agent.keyword_extractor", "agent.pdf_processor"]:
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.INFO)
        lgr.addHandler(log_handler)

    with st.spinner("Running pipeline..."):
        agent = load_agent()

        if has_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            result = agent.summarize_pdf(tmp_path, summary_length=length)
            try: os.unlink(tmp_path)
            except: pass
        else:
            result = agent.summarize_text(input_text, summary_length=length)

    # Cleanup loggers
    for name in ["", "agent.agent", "agent.brain", "agent.executor", "agent.planner", "agent.keyword_extractor", "agent.pdf_processor"]:
        logging.getLogger(name).removeHandler(log_handler)

    # ── Results ──
    st.markdown("---")

    # Stats
    stats = result.get("stats", {})
    st.markdown(f"""
    <div class="stats">
        <div class="stat"><div class="stat-val">{stats.get('input_chars',0):,}</div><div class="stat-lbl">Characters</div></div>
        <div class="stat"><div class="stat-val">{stats.get('num_chunks',0)}</div><div class="stat-lbl">Chunks</div></div>
        <div class="stat"><div class="stat-val">{stats.get('chunks_local',0)}</div><div class="stat-lbl">FLAN-T5</div></div>
        <div class="stat"><div class="stat-val">{stats.get('chunks_gemini',0)}</div><div class="stat-lbl">Gemini</div></div>
        <div class="stat"><div class="stat-val">{stats.get('processing_time_seconds',0):.1f}s</div><div class="stat-lbl">Time</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="result-card">
        <h4>📝 Summary</h4>
        <p>{result.get('summary', 'No summary generated.')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Keywords
    keywords = result.get("keywords", [])
    if keywords:
        method_cls = {"YAKE": "chip", "RAKE": "chip chip-cyan", "TF-IDF": "chip chip-green"}
        chips = " ".join(
            f'<span class="{method_cls.get(kw.get("method",""), "chip")}">{kw["keyword"]}</span>'
            for kw in keywords
        )
        st.markdown(f'<div class="result-card"><h4>🏷️ Keywords</h4><div class="chips">{chips}</div></div>', unsafe_allow_html=True)


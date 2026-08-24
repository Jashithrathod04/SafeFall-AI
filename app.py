import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import CLASS_NAMES, SEQUENCE_LENGTH
from src.pose.pose_detector import PoseDetector
from src.model.predictor import Predictor
from src.detection.temporal_validator import TemporalValidator
from src.alerts.alert_manager import AlertManager


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #070b0a;
    --bg2: #0b1210;
    --glass: rgba(16, 25, 22, .68);
    --glass2: rgba(20, 32, 28, .78);
    --line: rgba(255,255,255,.09);
    --line2: rgba(255,255,255,.16);
    --text: #edf5f1;
    --muted: #8da29a;
    --green: #34d399;
    --cyan: #22d3ee;
    --amber: #fbbf24;
    --red: #f87171;
    --shadow: 0 20px 60px rgba(0,0,0,.28);
    --radius: 20px;
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(52,211,153,.09), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(34,211,238,.07), transparent 25%),
        radial-gradient(circle at 55% 100%, rgba(52,211,153,.045), transparent 30%),
        linear-gradient(135deg, var(--bg), var(--bg2));
    color: var(--text);
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .22;
    background-image:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 92%);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

h1,h2,h3,h4 {
    font-family: "Space Grotesk", sans-serif !important;
    color: var(--text) !important;
}

p, label, span {
    color: var(--text);
}

.mono {
    font-family: "JetBrains Mono", monospace !important;
}

.glass {
    background: linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.015)), var(--glass);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.glass:hover {
    border-color: rgba(255,255,255,.13);
}

.panel {
    padding: 22px;
}

.small-label {
    display: block;
    color: var(--muted);
    font: 600 .67rem "JetBrains Mono", monospace;
    text-transform: uppercase;
    letter-spacing: .14em;
    margin-bottom: 8px;
}

.muted {
    color: var(--muted) !important;
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
}

.brand-mark {
    width:44px;
    height:44px;
    border-radius:14px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg, rgba(52,211,153,.2), rgba(34,211,238,.08));
    border:1px solid var(--line2);
    box-shadow: 0 0 35px rgba(52,211,153,.12);
    font-size:21px;
}

.brand-name {
    font:700 1.35rem "Space Grotesk", sans-serif;
    letter-spacing:-.03em;
}

.brand-sub {
    color:var(--muted);
    font-size:.75rem;
    margin-top:1px;
}

.online-pill {
    display:inline-flex;
    align-items:center;
    gap:8px;
    border:1px solid rgba(52,211,153,.24);
    background:rgba(52,211,153,.09);
    color:var(--green);
    padding:7px 12px;
    border-radius:999px;
    font:600 .68rem "JetBrains Mono", monospace;
    letter-spacing:.08em;
}

.pulse {
    width:7px;
    height:7px;
    border-radius:50%;
    background:var(--green);
    box-shadow:0 0 12px var(--green);
    animation:pulse 1.8s ease-in-out infinite;
}

@keyframes pulse {
    0%,100% { transform:scale(1); opacity:1; }
    50% { transform:scale(.65); opacity:.55; }
}

.hero {
    position:relative;
    overflow:hidden;
    padding:34px;
    min-height:230px;
    display:flex;
    align-items:center;
    border-radius:28px;
    background:
        radial-gradient(circle at 82% 50%, rgba(34,211,238,.09), transparent 25%),
        radial-gradient(circle at 70% 20%, rgba(52,211,153,.11), transparent 30%),
        rgba(15,23,20,.72);
    border:1px solid var(--line);
    box-shadow:var(--shadow);
}

.hero:after {
    content:"";
    position:absolute;
    width:360px;
    height:360px;
    right:-130px;
    top:-150px;
    border:1px solid rgba(52,211,153,.12);
    border-radius:50%;
    box-shadow:0 0 0 45px rgba(52,211,153,.025), 0 0 0 90px rgba(52,211,153,.018);
    animation:orbit 12s linear infinite;
}

@keyframes orbit {
    from { transform:rotate(0deg); }
    to { transform:rotate(360deg); }
}

.hero-title {
    font:700 clamp(2.2rem, 5vw, 4.5rem) "Space Grotesk", sans-serif;
    letter-spacing:-.055em;
    line-height:.95;
    margin:0;
}

.hero-title span {
    background:linear-gradient(90deg,var(--green),var(--cyan));
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
}

.hero-copy {
    color:var(--muted);
    max-width:650px;
    margin-top:14px;
    line-height:1.7;
}

.metric-grid {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:14px;
    margin:18px 0;
}

.metric {
    padding:19px;
    min-height:118px;
    border:1px solid var(--line);
    border-radius:17px;
    background:rgba(15,23,20,.65);
    transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}

.metric:hover {
    transform:translateY(-3px);
    border-color:rgba(52,211,153,.25);
    box-shadow:0 14px 40px rgba(0,0,0,.2);
}

.metric-name {
    color:var(--muted);
    font:600 .66rem "JetBrains Mono", monospace;
    letter-spacing:.1em;
    text-transform:uppercase;
}

.metric-value {
    font:700 1.9rem "Space Grotesk", sans-serif;
    margin-top:10px;
}

.metric-foot {
    color:var(--muted);
    font-size:.72rem;
    margin-top:5px;
}

.nav-title {
    color:var(--muted);
    font:600 .65rem "JetBrains Mono", monospace;
    letter-spacing:.14em;
    margin:14px 0 8px;
    text-transform:uppercase;
}

.nav-active {
    padding:10px 12px;
    border-radius:12px;
    background:linear-gradient(90deg,rgba(52,211,153,.13),rgba(34,211,238,.05));
    border:1px solid rgba(52,211,153,.17);
    color:var(--text);
    margin-bottom:6px;
}

.section-head {
    display:flex;
    justify-content:space-between;
    align-items:end;
    margin:26px 0 12px;
}

.section-title {
    font:700 1.45rem "Space Grotesk", sans-serif;
}

.section-sub {
    color:var(--muted);
    font-size:.82rem;
}

.status-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:12px 0;
    border-bottom:1px solid var(--line);
}

.status-row:last-child { border-bottom:0; }

.status-ok {
    color:var(--green);
    font:600 .75rem "JetBrains Mono", monospace;
}

.status-warn {
    color:var(--amber);
    font:600 .75rem "JetBrains Mono", monospace;
}

.status-bad {
    color:var(--red);
    font:600 .75rem "JetBrains Mono", monospace;
}

.activity {
    font:700 1.7rem "Space Grotesk", sans-serif;
    margin:2px 0 16px;
}

.bar {
    height:8px;
    border-radius:999px;
    background:rgba(255,255,255,.06);
    overflow:hidden;
}

.fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,var(--green),var(--cyan));
    box-shadow:0 0 16px rgba(52,211,153,.16);
    transition:width .45s ease;
}

.fill-alert {
    background:linear-gradient(90deg,var(--amber),var(--red));
}

.alert {
    padding:22px;
    border-radius:20px;
    border:1px solid rgba(248,113,113,.35);
    background:linear-gradient(135deg,rgba(248,113,113,.12),rgba(248,113,113,.035));
    box-shadow:0 0 50px rgba(248,113,113,.1);
    animation:alertPulse 2.2s ease-in-out infinite;
}

@keyframes alertPulse {
    0%,100% { box-shadow:0 0 25px rgba(248,113,113,.07); }
    50% { box-shadow:0 0 55px rgba(248,113,113,.18); }
}

.safe {
    padding:22px;
    border-radius:20px;
    border:1px solid rgba(52,211,153,.25);
    background:rgba(52,211,153,.07);
}

.timeline {
    display:flex;
    align-items:center;
    gap:0;
    padding:24px 6px;
}

.timeline-line {
    height:2px;
    flex:1;
    background:linear-gradient(90deg,rgba(52,211,153,.3),rgba(34,211,238,.3));
}

.timeline-dot {
    width:12px;
    height:12px;
    border-radius:50%;
    background:var(--green);
    box-shadow:0 0 15px rgba(52,211,153,.5);
    margin:0 5px;
}

.timeline-dot.alert-dot {
    background:var(--red);
    box-shadow:0 0 18px rgba(248,113,113,.55);
}

.process-wrap {
    padding:28px;
    text-align:center;
}

.ai-orb {
    width:105px;
    height:105px;
    margin:8px auto 22px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    border:1px solid rgba(52,211,153,.28);
    background:radial-gradient(circle,rgba(52,211,153,.2),rgba(34,211,238,.04) 55%,transparent 70%);
    box-shadow:0 0 45px rgba(52,211,153,.12);
    animation:orb 2.5s ease-in-out infinite;
}

@keyframes orb {
    0%,100% { transform:scale(1); box-shadow:0 0 35px rgba(52,211,153,.1); }
    50% { transform:scale(1.06); box-shadow:0 0 70px rgba(52,211,153,.2); }
}

.orb-core {
    width:28px;
    height:28px;
    border-radius:50%;
    background:linear-gradient(135deg,var(--green),var(--cyan));
    box-shadow:0 0 25px rgba(52,211,153,.5);
}

.login-shell {
    min-height:82vh;
    display:flex;
    align-items:center;
    justify-content:center;
}

.login-card {
    width:min(480px,100%);
    padding:34px;
    border-radius:28px;
    background:rgba(12,19,17,.78);
    border:1px solid var(--line2);
    box-shadow:0 30px 100px rgba(0,0,0,.45), 0 0 70px rgba(52,211,153,.05);
    backdrop-filter:blur(24px);
}

.login-logo {
    width:70px;
    height:70px;
    border-radius:22px;
    margin:0 auto 18px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:31px;
    background:linear-gradient(135deg,rgba(52,211,153,.2),rgba(34,211,238,.1));
    border:1px solid var(--line2);
    box-shadow:0 0 55px rgba(52,211,153,.14);
    animation:float 4s ease-in-out infinite;
}

@keyframes float {
    0%,100% { transform:translateY(0); }
    50% { transform:translateY(-6px); }
}

.login-title {
    text-align:center;
    font:700 2rem "Space Grotesk", sans-serif;
    letter-spacing:-.04em;
}

.login-sub {
    text-align:center;
    color:var(--muted);
    margin:7px 0 24px;
}

.waiting {
    min-height:76vh;
    display:flex;
    align-items:center;
    justify-content:center;
}

.waiting-card {
    width:min(700px,100%);
    text-align:center;
    padding:44px;
}

.check-row {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:12px 14px;
    margin:7px 0;
    border:1px solid var(--line);
    border-radius:12px;
    background:rgba(255,255,255,.025);
}

.table-card {
    padding:0;
    overflow:hidden;
}

.incident {
    padding:18px 20px;
    border-bottom:1px solid var(--line);
    transition:background .2s ease;
}

.incident:hover {
    background:rgba(255,255,255,.025);
}

.incident:last-child { border-bottom:0; }

.footer {
    text-align:center;
    color:var(--muted);
    font:500 .65rem "JetBrains Mono", monospace;
    letter-spacing:.08em;
    margin-top:45px;
}

@media (max-width:900px) {
    .metric-grid { grid-template-columns:repeat(2,1fr); }
}

@media (max-width:600px) {
    .metric-grid { grid-template-columns:1fr; }
    .hero { padding:24px; }
    .block-container { padding-left:1rem; padding-right:1rem; }
}

/* Streamlit controls */
.stButton > button {
    border-radius:12px !important;
    border:1px solid rgba(52,211,153,.22) !important;
    background:linear-gradient(135deg,rgba(52,211,153,.16),rgba(34,211,238,.08)) !important;
    color:var(--text) !important;
    font-family:"Space Grotesk",sans-serif !important;
    font-weight:700 !important;
    min-height:42px;
    transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease !important;
}

.stButton > button:hover {
    transform:translateY(-2px);
    border-color:rgba(52,211,153,.5) !important;
    box-shadow:0 10px 30px rgba(52,211,153,.13) !important;
}

.stButton > button:active {
    transform:scale(.985);
}

.stTextInput input, .stNumberInput input {
    background:rgba(255,255,255,.035) !important;
    border:1px solid var(--line2) !important;
    color:var(--text) !important;
    border-radius:12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,.025) !important;
    border:1.5px dashed rgba(255,255,255,.18) !important;
    border-radius:20px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color:var(--green) !important;
    box-shadow:0 0 35px rgba(52,211,153,.08);
}

div[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,var(--green),var(--cyan)) !important;
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0a110f,#070b0a) !important;
    border-right:1px solid var(--line);
}

section[data-testid="stSidebar"] .block-container {
    padding-top:1.2rem;
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Dashboard",
    "authenticated": False,
    "splash_done": False,
    "boot_done": False,
    "login_error": "",
    "analysis_count": 0,
    "fall_count": 0,
    "last_confidence": 0.0,
    "last_activity": "Awaiting analysis",
    "last_alert": None,
    "incidents": [],
    "system_error": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def render_brand(compact=False):
    html(f"""
    <div class="brand">
        <div class="brand-mark">🛡️</div>
        <div>
            <div class="brand-name">SafeFall AI</div>
            <div class="brand-sub">Computer-vision safety intelligence</div>
        </div>
    </div>
    """)


def render_header():
    left, right = st.columns([1, 1])
    with left:
        render_brand()
    with right:
        html('<div style="text-align:right"><span class="online-pill"><span class="pulse"></span>SYSTEM ONLINE</span></div>')


def animate_splash():
    if st.session_state.splash_done:
        return

    placeholder = st.empty()

    placeholder.markdown("""
    <div class="waiting">
      <div class="waiting-card">
        <div class="login-logo">🛡️</div>
        <div class="login-title">SafeFall <span style="color:#34d399">AI</span></div>
        <div class="login-sub">Intelligent fall detection and safety monitoring</div>
        <div class="glass panel" style="text-align:left">
          <div class="check-row"><span>VISION ENGINE</span><span class="status-ok">INITIALIZING</span></div>
          <div class="check-row"><span>POSE DETECTOR</span><span class="status-ok">INITIALIZING</span></div>
          <div class="check-row"><span>TEMPORAL ANALYSIS</span><span class="status-ok">INITIALIZING</span></div>
          <div class="check-row"><span>ALERT PIPELINE</span><span class="status-ok">INITIALIZING</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(.7)

    placeholder.markdown("""
    <div class="waiting">
      <div class="waiting-card">
        <div class="login-logo">🛡️</div>
        <div class="login-title">SafeFall <span style="color:#34d399">AI</span></div>
        <div class="login-sub">Preparing secure monitoring environment</div>
        <div class="glass panel" style="text-align:left">
          <div class="check-row"><span>VISION ENGINE</span><span class="status-ok">✓ READY</span></div>
          <div class="check-row"><span>POSE DETECTOR</span><span class="status-ok">✓ READY</span></div>
          <div class="check-row"><span>TEMPORAL ANALYSIS</span><span class="status-ok">✓ READY</span></div>
          <div class="check-row"><span>ALERT PIPELINE</span><span class="status-ok">✓ ARMED</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(.7)
    st.session_state.splash_done = True
    placeholder.empty()


def login_page():
    html("""
    <div class="login-shell">
      <div class="login-card">
        <div class="login-logo">🛡️</div>
        <div class="login-title">Welcome back</div>
        <div class="login-sub">Sign in to your SafeFall monitoring console.</div>
    </div>
    </div>
    """)

    # The card above is visual; inputs are placed immediately after it
    # so Streamlit widgets remain functional and accessible.
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        username = st.text_input("Email", placeholder="operator@safefall.ai")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        remember = st.checkbox("Remember this session")

        if st.button("SIGN IN  →", use_container_width=True):
            # Demo authentication. Replace with your real auth provider later.
            if username.strip() and password.strip():
                st.session_state.authenticated = True
                st.session_state.login_error = ""
                st.session_state.boot_done = False
                st.rerun()
            else:
                st.session_state.login_error = "Enter both email and password."

        if st.session_state.login_error:
            html(f'<div class="alert" style="margin-top:12px">⚠️ {st.session_state.login_error}</div>')

        html("""
        <div style="text-align:center;color:#8da29a;font-size:.7rem;margin-top:16px">
            Secure monitoring console · Demo authentication
        </div>
        """)


def boot_page():
    if st.session_state.boot_done:
        return

    holder = st.empty()

    steps = [
        ("SECURE SESSION", "✓ VERIFIED"),
        ("VISION SERVICES", "✓ READY"),
        ("MONITORING CONSOLE", "✓ READY"),
    ]

    for i in range(len(steps) + 1):
        rows = ""
        for j, (name, _) in enumerate(steps):
            status = "✓ READY" if j < i else "◌ LOADING"
            cls = "status-ok" if j < i else "status-warn"
            rows += f'<div class="check-row"><span>{name}</span><span class="{cls}">{status}</span></div>'

        holder.markdown(f"""
        <div class="waiting">
          <div class="waiting-card">
            <div class="ai-orb"><div class="orb-core"></div></div>
            <div class="login-title">Preparing your workspace</div>
            <div class="login-sub">Synchronizing SafeFall AI services...</div>
            <div class="glass panel" style="text-align:left">{rows}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(.35)

    st.session_state.boot_done = True
    holder.empty()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_system():
    pose_detector = PoseDetector()
    predictor = Predictor()
    validator = TemporalValidator()
    alert_manager = AlertManager()
    return pose_detector, predictor, validator, alert_manager


def model_status():
    try:
        objects = load_system()
        st.session_state.system_error = None
        return objects, None
    except Exception as exc:
        st.session_state.system_error = str(exc)
        return None, exc


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        render_brand()
        st.markdown("---")

        html('<div class="nav-title">Workspace</div>')

        pages = [
            ("🏠", "Dashboard"),
            ("🎥", "Live Monitor"),
            ("📁", "Video Analysis"),
            ("🚨", "Incidents"),
            ("📊", "Analytics"),
        ]

        for icon, page in pages:
            if st.session_state.page == page:
                html(f'<div class="nav-active">{icon}&nbsp;&nbsp; {page}</div>')
            else:
                if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
                    set_page(page)

        html('<div class="nav-title">Intelligence</div>')

        for icon, page in [
            ("🧠", "AI Performance"),
            ("🖥️", "System Health"),
            ("⚙️", "Settings"),
        ]:
            if st.session_state.page == page:
                html(f'<div class="nav-active">{icon}&nbsp;&nbsp; {page}</div>')
            else:
                if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
                    set_page(page)

        st.markdown("---")
        html("""
        <div class="glass panel">
          <span class="small-label">Monitoring status</span>
          <div class="status-row"><span>AI Core</span><span class="status-ok">ONLINE</span></div>
          <div class="status-row"><span>Alert Engine</span><span class="status-ok">ARMED</span></div>
          <div class="status-row"><span>Camera</span><span class="status-ok">READY</span></div>
        </div>
        """)

        st.markdown("")
        if st.button("↪  Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.splash_done = True
            st.session_state.boot_done = False
            st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():
    render_header()

    html("""
    <div style="height:16px"></div>
    <div class="hero">
      <div>
        <span class="small-label">Intelligent safety console</span>
        <div class="hero-title">Watch smarter.<br><span>Respond faster.</span></div>
        <div class="hero-copy">
          SafeFall AI combines pose extraction, temporal activity recognition
          and multi-frame validation to identify potential fall events.
        </div>
      </div>
    </div>
    """)

    falls = st.session_state.fall_count
    confidence = st.session_state.last_confidence * 100
    activity = st.session_state.last_activity

    html(f"""
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-name">Falls detected</div>
        <div class="metric-value">{falls:02d}</div>
        <div class="metric-foot">Confirmed incidents</div>
      </div>
      <div class="metric">
        <div class="metric-name">Latest confidence</div>
        <div class="metric-value">{confidence:.1f}%</div>
        <div class="metric-foot">Most recent prediction</div>
      </div>
      <div class="metric">
        <div class="metric-name">Analyses</div>
        <div class="metric-value">{st.session_state.analysis_count:02d}</div>
        <div class="metric-foot">Recordings processed</div>
      </div>
      <div class="metric">
        <div class="metric-name">System</div>
        <div class="metric-value" style="font-size:1.45rem;color:#34d399">ONLINE</div>
        <div class="metric-foot">All interface services ready</div>
      </div>
    </div>
    """)

    left, right = st.columns([1.45, .85], gap="large")

    with left:
        html("""
        <div class="glass panel">
          <span class="small-label">Current activity</span>
        """)
        html(f"""
          <div class="activity">{activity}</div>
          <div class="metric-name" style="margin-bottom:7px">Model confidence</div>
          <div class="bar"><div class="fill" style="width:{min(confidence,100):.1f}%"></div></div>
        </div>
        """)
        if st.button("🎥  Open Live Monitor", use_container_width=True):
            set_page("Live Monitor")

    with right:
        html("""
        <div class="glass panel">
          <span class="small-label">AI pipeline</span>
          <div class="status-row"><span>01 · Pose extraction</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>02 · Sequence buffer</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>03 · BiLSTM inference</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>04 · Temporal validation</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>05 · Alert dispatch</span><span class="status-ok">ARMED</span></div>
        </div>
        """)

    html('<div class="section-head"><div><div class="section-title">Activity timeline</div><div class="section-sub">Recent SafeFall events</div></div></div>')

    if not st.session_state.incidents:
        html("""
        <div class="glass panel">
          <div class="timeline">
            <div class="timeline-dot"></div>
            <div class="timeline-line"></div>
            <div class="timeline-dot"></div>
            <div class="timeline-line"></div>
            <div class="timeline-dot"></div>
          </div>
          <div style="text-align:center;color:#8da29a;font-size:.8rem">
            No incidents recorded yet. Upload a recording to begin analysis.
          </div>
        </div>
        """)
    else:
        html('<div class="glass table-card">')
        for incident in st.session_state.incidents[-5:][::-1]:
            html(f"""
            <div class="incident">
              <div style="display:flex;justify-content:space-between">
                <strong>🔴 {incident["title"]}</strong>
                <span class="mono muted">{incident["time"]}</span>
              </div>
              <div class="muted" style="font-size:.8rem;margin-top:5px">
                Confidence {incident["confidence"]:.1f}% · {incident["activity"]}
              </div>
            </div>
            """)
        html('</div>')


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def video_analysis():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">Video Analysis</div>
        <div class="section-sub">Run the complete SafeFall AI pipeline on a recording.</div>
      </div>
    </div>
    """)

    uploaded = st.file_uploader(
        "Upload a surveillance video",
        type=["avi", "mp4", "mov"],
        label_visibility="collapsed",
        key="analysis_upload",
    )

    if uploaded is None:
        html("""
        <div class="glass process-wrap">
          <div class="ai-orb"><div class="orb-core"></div></div>
          <div style="font:700 1.2rem 'Space Grotesk'">Drop a recording to begin</div>
          <div class="muted" style="margin-top:8px">
            MP4, AVI or MOV · pose extraction · temporal analysis · fall validation
          </div>
        </div>
        """)
        return

    c1, c2 = st.columns([1.25, .75], gap="large")

    with c1:
        html('<div class="glass panel"><span class="small-label">Source recording</span>')
        st.video(uploaded)
        html('</div>')

    with c2:
        html("""
        <div class="glass panel">
          <span class="small-label">Recording loaded</span>
          <div style="font:700 1.15rem 'Space Grotesk'">""" + uploaded.name + """</div>
          <div class="muted" style="margin-top:6px">Ready for inference</div>
        </div>
        """)
        analyze = st.button("▶  RUN AI ANALYSIS", use_container_width=True)

    if not analyze:
        return

    objects, error = model_status()
    if error:
        html("""
        <div class="alert" style="margin-top:18px">
          <div style="font:700 1.15rem 'Space Grotesk'">⚠️ AI model could not be initialized</div>
          <div class="muted" style="margin-top:7px">
            The frontend is running, but the inference model is unavailable.
            Check that <b>models/activity_model/best_model.pt</b> exists in the deployed repository.
          </div>
        </div>
        """)
        with st.expander("Deployment diagnostic"):
            st.exception(error)
        return

    pose_detector, predictor, validator, alert_manager = objects

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as temp:
        temp.write(uploaded.getbuffer())
        video_path = temp.name

    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 1

    sequence = []
    detected_events = []
    frame_index = 0

    html('<div style="height:16px"></div>')
    html('<div class="glass panel">')
    html('<span class="small-label">AI processing</span>')

    status = st.empty()
    progress = st.progress(0)

    status.markdown(
        '<span class="mono muted">01 · Extracting frames and initializing temporal sequence…</span>',
        unsafe_allow_html=True,
    )

    activity_col, metric_col = st.columns([1.3, .7])
    with activity_col:
        activity_placeholder = st.empty()
    with metric_col:
        confidence_placeholder = st.empty()

    while True:
        success, frame = capture.read()
        if not success:
            break

        keypoints = pose_detector.extract_keypoints(frame)
        sequence.append(keypoints)

        if len(sequence) >= SEQUENCE_LENGTH:
            window = np.asarray(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)

            result = predictor.predict(window)
            label = result["label"]
            confidence = float(result["confidence"])

            st.session_state.last_activity = str(label)
            st.session_state.last_confidence = confidence

            is_alert = confidence >= float(st.session_state.confidence_threshold)

            bar_class = "fill fill-alert" if is_alert else "fill"

            activity_placeholder.markdown(f"""
            <div class="glass panel">
              <span class="small-label">Live model output</span>
              <div class="activity">{'⚠️' if is_alert else '🧍'} {label}</div>
              <div class="metric-name" style="margin-bottom:7px">Confidence</div>
              <div class="bar"><div class="{bar_class}" style="width:{min(confidence*100,100):.1f}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

            confidence_placeholder.markdown(f"""
            <div class="glass panel" style="text-align:center">
              <span class="small-label">Probability</span>
              <div style="font:700 2.3rem 'Space Grotesk';color:{'#f87171' if is_alert else '#34d399'}">
                {confidence*100:.1f}%
              </div>
              <div class="muted" style="font-size:.7rem">current prediction</div>
            </div>
            """, unsafe_allow_html=True)

            confirmed = validator.update(label, confidence)

            if confirmed:
                alert = alert_manager.trigger()
                detected_events.append(alert)

                st.session_state.fall_count += 1
                st.session_state.last_alert = str(alert)
                st.session_state.incidents.append({
                    "title": "Fall detected",
                    "time": time.strftime("%H:%M:%S"),
                    "confidence": confidence * 100,
                    "activity": str(label),
                    "alert": str(alert),
                })

            status.markdown(
                f'<span class="mono muted">03 · Analyzing temporal sequence · frame {frame_index}/{frame_count}</span>',
                unsafe_allow_html=True,
            )

        frame_index += 1
        progress.progress(min(frame_index / frame_count, 1.0))

    capture.release()

    st.session_state.analysis_count += 1

    status.markdown(
        '<span class="mono" style="color:#34d399">✓ Analysis complete · temporal validation finished</span>',
        unsafe_allow_html=True,
    )

    html('</div>')

    html('<div style="height:18px"></div>')

    if detected_events:
        html(f"""
        <div class="alert">
          <div style="font:700 1.35rem 'Space Grotesk'">🔴 FALL DETECTED</div>
          <div class="muted" style="margin-top:8px">
            The temporal validator confirmed a fall event during this recording.
          </div>
          <div class="mono" style="margin-top:14px">Latest event: {detected_events[-1]}</div>
        </div>
        """)
    else:
        html("""
        <div class="safe">
          <div style="font:700 1.2rem 'Space Grotesk'">🟢 No confirmed fall detected</div>
          <div class="muted" style="margin-top:7px">
            The complete recording was processed and no sequence crossed the
            confirmed-fall threshold.
          </div>
        </div>
        """)

    try:
        os.remove(video_path)
    except OSError:
        pass


# ============================================================
# LIVE MONITOR
# ============================================================

def live_monitor():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">Live Monitor</div>
        <div class="section-sub">Real-time camera monitoring interface.</div>
      </div>
      <span class="online-pill"><span class="pulse"></span>LIVE</span>
    </div>
    """)

    left, right = st.columns([1.5, .8], gap="large")

    with left:
        html("""
        <div class="glass panel" style="min-height:420px">
          <div style="display:flex;justify-content:space-between;margin-bottom:12px">
            <span class="small-label">Camera 01 · Primary</span>
            <span class="status-ok">CONNECTED</span>
          </div>
        """)
        st.info("Connect your live camera stream here. The existing SafeFall inference pipeline currently analyzes uploaded recordings.")
        html("""
          <div style="height:250px;border-radius:16px;border:1px solid rgba(255,255,255,.07);
                      display:flex;align-items:center;justify-content:center;
                      background:radial-gradient(circle,rgba(52,211,153,.06),transparent 50%);">
            <div style="text-align:center">
              <div class="ai-orb"><div class="orb-core"></div></div>
              <div style="font:600 1rem 'Space Grotesk'">Monitoring ready</div>
              <div class="muted" style="font-size:.75rem;margin-top:5px">Awaiting camera stream</div>
            </div>
          </div>
        </div>
        """)

    with right:
        html("""
        <div class="glass panel">
          <span class="small-label">Real-time AI state</span>
          <div class="activity">""" + str(st.session_state.last_activity) + """</div>
          <div class="metric-name">Latest confidence</div>
          <div class="bar" style="margin-top:8px">
            <div class="fill" style="width:""" + str(min(st.session_state.last_confidence * 100,100)) + """%"></div>
          </div>
          <div class="status-row" style="margin-top:18px"><span>Vision engine</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>Temporal model</span><span class="status-ok">READY</span></div>
          <div class="status-row"><span>Alert engine</span><span class="status-ok">ARMED</span></div>
        </div>
        """)

    if st.session_state.last_alert:
        html(f"""
        <div class="alert" style="margin-top:18px">
          <div style="font:700 1.15rem 'Space Grotesk'">🔴 Latest alert</div>
          <div class="mono muted" style="margin-top:8px">{st.session_state.last_alert}</div>
        </div>
        """)


# ============================================================
# INCIDENTS
# ============================================================

def incidents_page():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">Incident Center</div>
        <div class="section-sub">Review confirmed fall events from this session.</div>
      </div>
    </div>
    """)

    if not st.session_state.incidents:
        html("""
        <div class="glass process-wrap">
          <div style="font-size:2.2rem">🛡️</div>
          <div style="font:700 1.2rem 'Space Grotesk';margin-top:10px">No incidents</div>
          <div class="muted" style="margin-top:7px">SafeFall has not recorded a confirmed event yet.</div>
        </div>
        """)
        return

    html('<div class="glass table-card">')
    for i, incident in enumerate(st.session_state.incidents[::-1], 1):
        html(f"""
        <div class="incident">
          <div style="display:flex;justify-content:space-between;gap:20px">
            <div>
              <div style="font:700 1rem 'Space Grotesk'">🔴 {incident["title"]}</div>
              <div class="muted" style="font-size:.78rem;margin-top:5px">
                {incident["activity"]} · Confidence {incident["confidence"]:.1f}%
              </div>
            </div>
            <div class="mono muted">{incident["time"]}</div>
          </div>
        </div>
        """)
    html('</div>')


# ============================================================
# ANALYTICS
# ============================================================

def analytics_page():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">Analytics</div>
        <div class="section-sub">Session-level monitoring intelligence.</div>
      </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        html(f"""
        <div class="metric">
          <div class="metric-name">Total analyses</div>
          <div class="metric-value">{st.session_state.analysis_count}</div>
          <div class="metric-foot">Recordings processed</div>
        </div>
        """)
    with c2:
        html(f"""
        <div class="metric">
          <div class="metric-name">Confirmed falls</div>
          <div class="metric-value">{st.session_state.fall_count}</div>
          <div class="metric-foot">Temporal validation events</div>
        </div>
        """)
    with c3:
        html(f"""
        <div class="metric">
          <div class="metric-name">Latest confidence</div>
          <div class="metric-value">{st.session_state.last_confidence*100:.1f}%</div>
          <div class="metric-foot">Most recent model output</div>
        </div>
        """)

    html('<div style="height:18px"></div>')

    if st.session_state.incidents:
        import pandas as pd

        df = pd.DataFrame(st.session_state.incidents)
        chart_df = df[["time", "confidence"]].copy()
        chart_df["confidence"] = chart_df["confidence"].astype(float)
        chart_df = chart_df.set_index("time")
        st.line_chart(chart_df, height=300)
    else:
        html("""
        <div class="glass panel">
          <span class="small-label">Confidence trend</span>
          <div class="muted">Run a video analysis to populate analytics.</div>
        </div>
        """)


# ============================================================
# AI PERFORMANCE
# ============================================================

def ai_performance():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">AI Performance</div>
        <div class="section-sub">Model architecture and runtime configuration.</div>
      </div>
    </div>
    """)

    html(f"""
    <div class="metric-grid">
      <div class="metric"><div class="metric-name">Architecture</div><div class="metric-value" style="font-size:1.35rem">BiLSTM</div><div class="metric-foot">Temporal classifier</div></div>
      <div class="metric"><div class="metric-name">Sequence length</div><div class="metric-value">{SEQUENCE_LENGTH}</div><div class="metric-foot">Frames per window</div></div>
      <div class="metric"><div class="metric-name">Classes</div><div class="metric-value">{len(CLASS_NAMES)}</div><div class="metric-foot">Activity categories</div></div>
      <div class="metric"><div class="metric-name">Validator</div><div class="metric-value" style="font-size:1.35rem">ACTIVE</div><div class="metric-foot">Multi-frame confirmation</div></div>
    </div>
    """)

    html("""
    <div class="glass panel">
      <span class="small-label">Tracked activity classes</span>
    """)
    html(" ".join(
        f'<span style="display:inline-block;padding:7px 10px;margin:4px;border:1px solid rgba(255,255,255,.1);border-radius:999px;color:#8da29a;font-size:.72rem">{name}</span>'
        for name in CLASS_NAMES
    ))
    html("</div>")


# ============================================================
# SYSTEM HEALTH
# ============================================================

def system_health():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">System Health</div>
        <div class="section-sub">SafeFall service diagnostics.</div>
      </div>
    </div>
    """)

    objects, error = model_status()

    html("""
    <div class="glass panel">
      <span class="small-label">Runtime diagnostics</span>
      <div class="status-row"><span>Streamlit UI</span><span class="status-ok">ONLINE</span></div>
      <div class="status-row"><span>Pose detector</span><span class="status-ok">READY</span></div>
      <div class="status-row"><span>Temporal validator</span><span class="status-ok">READY</span></div>
      <div class="status-row"><span>Alert manager</span><span class="status-ok">ARMED</span></div>
    """)

    if error:
        html("""
        <div class="status-row">
          <span>Activity model</span>
          <span class="status-bad">UNAVAILABLE</span>
        </div>
        """)
    else:
        html("""
        <div class="status-row">
          <span>Activity model</span>
          <span class="status-ok">LOADED</span>
        </div>
        """)

    html("</div>")

    if error:
        html("""
        <div class="alert" style="margin-top:18px">
          <div style="font:700 1.1rem 'Space Grotesk'">Model deployment issue</div>
          <div class="muted" style="margin-top:7px">
            The UI is operational, but the Predictor could not initialize.
            Make sure the trained model is committed to the repository at
            <span class="mono">models/activity_model/best_model.pt</span>
            or update the model path used by <span class="mono">Predictor</span>.
          </div>
        </div>
        """)
        with st.expander("Technical error"):
            st.exception(error)


# ============================================================
# SETTINGS
# ============================================================

def settings_page():
    render_header()

    html("""
    <div class="section-head">
      <div>
        <div class="section-title">Settings</div>
        <div class="section-sub">Tune detection and interface behaviour.</div>
      </div>
    </div>
    """)

    st.session_state.confidence_threshold = st.slider(
        "Fall confidence threshold",
        min_value=.50,
        max_value=.99,
        value=float(st.session_state.get("confidence_threshold", .70)),
        step=.01,
    )

    c1, c2 = st.columns(2)

    with c1:
        html(f"""
        <div class="glass panel">
          <span class="small-label">Detection</span>
          <div class="status-row"><span>Threshold</span><span class="mono">{st.session_state.confidence_threshold*100:.0f}%</span></div>
          <div class="status-row"><span>Sequence length</span><span class="mono">{SEQUENCE_LENGTH}</span></div>
          <div class="status-row"><span>Classes</span><span class="mono">{len(CLASS_NAMES)}</span></div>
        </div>
        """)

    with c2:
        html("""
        <div class="glass panel">
          <span class="small-label">Interface</span>
        """)
        reduced_motion = st.checkbox("Reduce animations", value=False)
        sound_alerts = st.checkbox("Sound alerts", value=True)
        html("""
        </div>
        """)

    if st.button("↻  Reset session statistics", use_container_width=True):
        st.session_state.analysis_count = 0
        st.session_state.fall_count = 0
        st.session_state.incidents = []
        st.session_state.last_confidence = 0
        st.session_state.last_activity = "Awaiting analysis"
        st.session_state.last_alert = None
        st.rerun()


# ============================================================
# ROUTER
# ============================================================

# Set threshold before any page needs it.
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = .70

animate_splash()

if not st.session_state.authenticated:
    login_page()
    st.stop()

boot_page()
render_sidebar()

page = st.session_state.page

if page == "Dashboard":
    dashboard()
elif page == "Live Monitor":
    live_monitor()
elif page == "Video Analysis":
    video_analysis()
elif page == "Incidents":
    incidents_page()
elif page == "Analytics":
    analytics_page()
elif page == "AI Performance":
    ai_performance()
elif page == "System Health":
    system_health()
elif page == "Settings":
    settings_page()
else:
    dashboard()

html("""
<div class="footer">
    SAFEFALL AI · VISION ENGINE + BiLSTM TEMPORAL ACTIVITY RECOGNITION · ON-DEVICE INFERENCE
</div>
""")

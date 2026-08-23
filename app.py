import tempfile
import time
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CLASS_NAMES,
    SEQUENCE_LENGTH,
)

from src.pose.pose_detector import (
    PoseDetector
)

from src.model.predictor import (
    Predictor
)

from src.detection.temporal_validator import (
    TemporalValidator
)

from src.alerts.alert_manager import (
    AlertManager
)


# ===========================================================
# PAGE CONFIG
# ===========================================================

st.set_page_config(
    page_title="SafeFall AI — Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin"


# ===========================================================
# DESIGN SYSTEM
# ===========================================================
# Background   #0A0E0D   Surface  rgba(20,27,25,.55)   Border rgba(255,255,255,.08)
# Text         #E9EFEC   Muted    #8CA099
# Emerald #34D399 (nominal)  Cyan #22D3EE (data)  Amber #FBBF24 (attention)  Red #F87171 (critical)
# Display: Space Grotesk · Body: Inter · Data: JetBrains Mono
# Signature element: looping ECG / vital-sign trace, used sparingly (splash, login, dashboard)
# ===========================================================


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root{
            --bg:#0A0E0D; --surface: rgba(20,27,25,0.55); --surface-solid:#121917;
            --border: rgba(255,255,255,0.08); --border-strong: rgba(255,255,255,0.16);
            --text:#E9EFEC; --text-muted:#8CA099;
            --emerald:#34D399; --emerald-dim: rgba(52,211,153,0.14);
            --cyan:#22D3EE; --cyan-dim: rgba(34,211,238,0.12);
            --amber:#FBBF24; --amber-dim: rgba(251,191,36,0.12);
            --red:#F87171; --red-dim: rgba(248,113,113,0.12);
            --radius-lg:20px; --radius-md:14px; --radius-sm:10px;
        }
        html, body, [class*="css"]{ font-family:'Inter', sans-serif; }
        .stApp{
            background:
                radial-gradient(ellipse 900px 500px at 15% -10%, rgba(52,211,153,0.07), transparent 60%),
                radial-gradient(ellipse 700px 500px at 100% 10%, rgba(34,211,238,0.05), transparent 55%),
                linear-gradient(180deg, #0A0E0D 0%, #0B100F 100%);
            color: var(--text);
        }
        #MainMenu, footer, header[data-testid="stHeader"]{ background: transparent; }
        .block-container{ padding-top: 1.6rem; max-width: 1180px; }
        h1, h2, h3, .display-font{ font-family:'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
        .mono{ font-family:'JetBrains Mono', monospace !important; }

        /* ---- top bar ---- */
        .sf-topbar{ display:flex; align-items:center; justify-content:space-between; padding: 2px 2px 0 2px; margin-bottom: 4px; }
        .sf-brand{ display:flex; align-items:center; gap:14px; }
        .sf-mark{ width:40px; height:40px; border-radius:11px; display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, rgba(52,211,153,0.18), rgba(34,211,238,0.10));
            border: 1px solid var(--border-strong); font-size:18px; box-shadow: 0 0 20px rgba(52,211,153,0.10); }
        .sf-mark-lg{ width:64px; height:64px; border-radius:16px; font-size:28px; }
        .sf-pagetitle{ font-family:'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; line-height:1.1; margin:0; letter-spacing:-0.02em; }
        .sf-pagesub{ color: var(--text-muted); font-size: 0.8rem; margin-top:2px; }
        .sf-status{ display:flex; align-items:center; gap:8px; font-family:'JetBrains Mono', monospace; font-size:0.72rem;
            letter-spacing:0.06em; color: var(--emerald); border: 1px solid rgba(52,211,153,0.25); background: var(--emerald-dim);
            padding: 7px 14px; border-radius: 999px; white-space:nowrap; }
        .sf-status.off{ color: var(--red); border-color: rgba(248,113,113,0.3); background: var(--red-dim); }
        .sf-dot{ width:7px; height:7px; border-radius:50%; background: var(--emerald); box-shadow: 0 0 8px var(--emerald); animation: pulseDot 2s ease-in-out infinite; }
        .sf-dot.off{ background: var(--red); box-shadow: 0 0 8px var(--red); animation:none; }
        @keyframes pulseDot{ 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.45; transform:scale(0.8);} }

        /* ---- vital line ---- */
        .sf-vitalwrap{ width:100%; height:30px; margin: 12px 0 18px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); overflow:hidden; opacity:0.8; }
        .sf-vitalwrap svg{ width:200%; height:100%; animation: vitalScroll 9s linear infinite; }
        @keyframes vitalScroll{ from{ transform: translateX(0);} to{ transform: translateX(-50%);} }

        /* ---- glass ---- */
        .glass{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 22px 24px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
        .glass-tight{ padding: 16px 18px; }
        .sf-panel-label{ font-family:'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 10px; display:block; }

        /* ---- kpi / system grid ---- */
        .sys-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin: 4px 0 20px 0; }
        .sys-card{ background: var(--surface); border:1px solid var(--border); border-radius: var(--radius-md); padding: 16px 18px; }
        .sys-card .k{ color:var(--text-muted); font-size:0.72rem; letter-spacing:0.05em; }
        .sys-card .v{ font-family:'Space Grotesk', sans-serif; font-size:1.15rem; font-weight:700; margin-top:6px; display:flex; align-items:center; gap:8px; }
        .sys-card .delta{ font-family:'JetBrains Mono', monospace; font-size:0.72rem; color: var(--emerald); margin-top:4px; }
        .chip-ok{ width:6px; height:6px; border-radius:50%; background:var(--emerald); box-shadow:0 0 6px var(--emerald); flex-shrink:0; }
        .chip-off{ width:6px; height:6px; border-radius:50%; background:var(--red); box-shadow:0 0 6px var(--red); flex-shrink:0; }
        .chip-warn{ width:6px; height:6px; border-radius:50%; background:var(--amber); box-shadow:0 0 6px var(--amber); flex-shrink:0; }

        /* ---- upload dropzone ---- */
        [data-testid="stFileUploaderDropzone"]{ background: var(--surface) !important; border: 1.5px dashed var(--border-strong) !important; border-radius: var(--radius-lg) !important; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
        [data-testid="stFileUploaderDropzone"]:hover{ border-color: var(--emerald) !important; box-shadow: 0 0 0 1px rgba(52,211,153,0.25); }
        [data-testid="stFileUploaderDropzoneInstructions"] svg{ display:none; }
        [data-testid="stFileUploader"] section > button{ background: transparent !important; border: 1px solid var(--border-strong) !important; color: var(--text) !important; border-radius: 8px !important; }

        /* ---- buttons ---- */
        .stButton > button{ width:100%; background: linear-gradient(135deg, #34D399, #22D3EE); color:#062018; font-weight:700;
            font-family:'Space Grotesk', sans-serif; letter-spacing: 0.01em; border:none; border-radius: var(--radius-sm);
            padding: 0.6rem 1rem; box-shadow: 0 8px 24px rgba(52,211,153,0.18); transition: transform 0.12s ease, box-shadow 0.12s ease; }
        .stButton > button:hover{ transform: translateY(-1px); box-shadow: 0 10px 28px rgba(52,211,153,0.28); }
        .stButton > button:active{ transform: translateY(0px) scale(0.99); }
        .ghost-btn button{ background: transparent !important; color: var(--text) !important; border: 1px solid var(--border-strong) !important; box-shadow:none !important; }
        .danger-btn button{ background: linear-gradient(135deg, #F87171, #FBBF24) !important; color:#2b0d0d !important; }

        /* nav buttons */
        .navwrap .stButton > button{ background: transparent !important; color: var(--text-muted) !important; box-shadow:none !important;
            text-align:left !important; justify-content:flex-start !important; font-family:'Inter', sans-serif !important; font-weight:500 !important;
            border-radius: 10px !important; padding: 0.45rem 0.7rem !important; border: 1px solid transparent !important; }
        .navwrap .stButton > button:hover{ background: rgba(255,255,255,0.04) !important; color: var(--text) !important; transform:none; }
        .navwrap-active .stButton > button{ background: var(--emerald-dim) !important; color: var(--emerald) !important; border: 1px solid rgba(52,211,153,0.25) !important; font-weight:600 !important; }

        /* ---- inputs / slider / progress ---- */
        [data-testid="stSlider"] [role="slider"]{ background-color: var(--emerald) !important; }
        div[data-baseweb="slider"] > div > div{ background: var(--emerald) !important; }
        div[data-testid="stProgress"] > div > div{ background: linear-gradient(90deg, #34D399, #22D3EE) !important; }
        [data-testid="stTextInput"] input{ background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border-strong) !important; color: var(--text) !important; border-radius: 10px !important; }

        /* ---- sidebar ---- */
        section[data-testid="stSidebar"]{ background: linear-gradient(180deg, #0B100F 0%, #0A0E0D 100%); border-right: 1px solid var(--border); }
        section[data-testid="stSidebar"] .block-container{ padding-top: 1.4rem; }

        /* ---- activity / live card ---- */
        .activity-card{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 22px 24px; }
        .activity-label{ color: var(--text-muted); font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; font-family:'JetBrains Mono', monospace; }
        .activity-value{ font-family:'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight:700; margin: 4px 0 16px 0; }
        .metric-row{ margin-bottom: 12px; }
        .metric-row .m-top{ display:flex; justify-content:space-between; font-size:0.78rem; color:var(--text-muted); margin-bottom:5px; font-family:'JetBrains Mono', monospace; }
        .bar-track{ width:100%; height:7px; border-radius:999px; background: rgba(255,255,255,0.06); overflow:hidden; }
        .bar-fill{ height:100%; border-radius:999px; background: linear-gradient(90deg, #34D399, #22D3EE); transition: width 0.4s ease; }
        .bar-fill.warn{ background: linear-gradient(90deg, #FBBF24, #F87171); }

        /* ---- result / alert cards ---- */
        .result-critical{ background: var(--red-dim); border: 1px solid rgba(248,113,113,0.35); border-radius: var(--radius-lg); padding: 22px 24px; box-shadow: 0 0 40px rgba(248,113,113,0.10); animation: alertGlow 2.4s ease-in-out infinite; }
        @keyframes alertGlow{ 0%,100%{ box-shadow: 0 0 24px rgba(248,113,113,0.08);} 50%{ box-shadow: 0 0 46px rgba(248,113,113,0.20);} }
        .result-safe{ background: var(--emerald-dim); border: 1px solid rgba(52,211,153,0.3); border-radius: var(--radius-lg); padding: 22px 24px; }
        .result-warn{ background: var(--amber-dim); border: 1px solid rgba(251,191,36,0.3); border-radius: var(--radius-lg); padding: 22px 24px; }
        .result-title{ font-family:'Space Grotesk', sans-serif; font-size:1.2rem; font-weight:700; display:flex; align-items:center; gap:10px; }
        .result-body{ color: var(--text-muted); margin-top:8px; font-size:0.9rem; }
        .result-meta{ font-family:'JetBrains Mono', monospace; font-size:0.78rem; color: var(--text); margin-top:14px; opacity:0.85; }

        /* ---- incident card ---- */
        .incident-card{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px 18px; margin-bottom: 12px; }
        .incident-top{ display:flex; justify-content:space-between; align-items:flex-start; }
        .incident-id{ font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1rem; }
        .incident-meta{ color: var(--text-muted); font-size:0.78rem; margin-top:4px; font-family:'JetBrains Mono', monospace; }
        .pill{ font-family:'JetBrains Mono', monospace; font-size:0.68rem; letter-spacing:0.06em; padding: 4px 10px; border-radius:999px; }
        .pill-red{ background: var(--red-dim); color: var(--red); border:1px solid rgba(248,113,113,0.3); }
        .pill-green{ background: var(--emerald-dim); color: var(--emerald); border:1px solid rgba(52,211,153,0.3); }
        .pill-amber{ background: var(--amber-dim); color: var(--amber); border:1px solid rgba(251,191,36,0.3); }
        .pill-muted{ background: rgba(255,255,255,0.04); color: var(--text-muted); border:1px solid var(--border); }

        /* ---- login / splash / waiting ---- */
        .auth-shell{ display:flex; justify-content:center; align-items:center; min-height: 74vh; }
        .auth-card{ width:100%; max-width: 400px; }
        .splash-shell{ display:flex; flex-direction:column; align-items:center; justify-content:center; min-height: 78vh; text-align:center; }
        .splash-title{ font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:2.4rem; letter-spacing:-0.02em; margin: 18px 0 4px 0; }
        .splash-sub{ color: var(--text-muted); font-size:0.95rem; margin-bottom: 28px; }
        .checklist-item{ display:flex; justify-content:space-between; width: 320px; margin: 0 auto 10px auto; font-family:'JetBrains Mono', monospace; font-size:0.82rem; color: var(--text-muted); }
        .checklist-item .ok{ color: var(--emerald); }

        /* ---- footer ---- */
        .sf-footnote{ color: var(--text-muted); font-size: 0.72rem; text-align:center; margin-top: 36px; letter-spacing: 0.03em; font-family:'JetBrains Mono', monospace; }
        hr{ border-color: var(--border) !important; }
        [data-testid="stForm"]{ border: none; padding: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def vital_line() -> None:
    st.markdown(
        """
        <div class="sf-vitalwrap">
            <svg viewBox="0 0 800 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M0,20 L60,20 L75,20 L85,4 L95,36 L105,20 L140,20
                         L200,20 L215,20 L225,4 L235,36 L245,20 L280,20
                         L340,20 L355,20 L365,4 L375,36 L385,20 L420,20
                         L480,20 L495,20 L505,4 L515,36 L525,20 L560,20
                         L620,20 L635,20 L645,4 L655,36 L665,20 L700,20
                         L760,20 L775,20 L785,4 L795,36 L800,20"
                      fill="none" stroke="#34D399" stroke-width="1.4" opacity="0.55"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#8CA099", "titleColor": "#8CA099",
                "gridColor": "rgba(255,255,255,0.06)", "domainColor": "rgba(255,255,255,0.12)",
                "labelFont": "JetBrains Mono", "titleFont": "Inter",
            },
            "legend": {"labelColor": "#8CA099", "titleColor": "#8CA099"},
        }
    }


alt.themes.register("safefall_dark", chart_theme)
alt.themes.enable("safefall_dark")


# ===========================================================
# SESSION STATE
# ===========================================================

def init_state() -> None:
    defaults = {
        "stage": "splash",
        "user_name": "",
        "page": "Dashboard",
        "incidents": [],
        "incident_seq": 23,
        "monitoring_active": False,
        "login_error": "",
        "confidence_threshold": 0.70,
        "sound_alerts": True,
        "confirmation_window": 8,
        "camera_checked": False,
        "camera_available": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_incident(alert, confidence: float, source: str) -> dict:
    st.session_state.incident_seq += 1
    record = {
        "id": st.session_state.incident_seq,
        "timestamp": datetime.now(),
        "confidence": confidence,
        "source": source,
        "status": "Confirmed",
        "raw": alert,
    }
    st.session_state.incidents.insert(0, record)
    return record


# ===========================================================
# LOAD MODELS  (core logic — unchanged)
# ===========================================================

@st.cache_resource
def load_system():
    pose_detector = PoseDetector()
    predictor = Predictor()
    validator = TemporalValidator()
    alert_manager = AlertManager()
    return (pose_detector, predictor, validator, alert_manager)


# ===========================================================
# SHARED CHROME
# ===========================================================

def top_bar(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="sf-topbar">
            <div class="sf-brand">
                <div class="sf-mark">🛡️</div>
                <div>
                    <p class="sf-pagetitle">{title}</p>
                    <p class="sf-pagesub">{subtitle}</p>
                </div>
            </div>
            <div class="sf-status"><span class="sf-dot"></span> SYSTEM ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = "", state: str = "ok") -> str:
    chip = {"ok": "chip-ok", "off": "chip-off", "warn": "chip-warn"}.get(state, "chip-ok")
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return f"""
        <div class="sys-card">
            <div class="k">{label}</div>
            <div class="v"><span class="{chip}"></span>{value}</div>
            {delta_html}
        </div>
    """


# ===========================================================
# STAGE 1 — SPLASH
# ===========================================================

def render_splash() -> None:
    inject_css()

    st.markdown('<div class="splash-shell">', unsafe_allow_html=True)
    st.markdown('<div class="sf-mark sf-mark-lg" style="margin:0 auto;">🛡️</div>', unsafe_allow_html=True)
    st.markdown('<div class="splash-title">SAFEFALL AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="splash-sub">Intelligent fall detection for elderly care</div>', unsafe_allow_html=True)

    checklist_slot = st.empty()
    steps = ["VISION ENGINE", "AI MODEL", "MOTION ANALYZER", "SYSTEM SERVICES"]
    revealed = []

    for step in steps:
        revealed.append(step)
        rows = "".join(
            f'<div class="checklist-item"><span>{s}</span><span class="ok">✓</span></div>'
            for s in revealed
        )
        checklist_slot.markdown(rows, unsafe_allow_html=True)
        time.sleep(0.22)

    time.sleep(0.3)
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.stage = "login"
    st.rerun()


# ===========================================================
# STAGE 2 — LOGIN
# ===========================================================

def render_login() -> None:
    inject_css()
    top_bar("SafeFall AI", "Secure monitoring session")
    vital_line()

    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.1, 1])

    with mid:
        st.markdown('<div class="glass auth-card">', unsafe_allow_html=True)
        st.markdown('<span class="sf-panel-label">Welcome back</span>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted); font-size:0.82rem; margin-top:-6px; margin-bottom:16px;">'
            'Sign in to access live monitoring and incident history.</p>',
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="admin")
            password = st.text_input("Password", placeholder="••••••", type="password")
            remember = st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button("Sign In  →")

        if submitted:
            if email.strip() == DEMO_USERNAME and password == DEMO_PASSWORD:
                st.session_state.user_name = email.strip().capitalize()
                st.session_state.login_error = ""
                st.session_state.stage = "waiting"
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect email or password."

        if st.session_state.login_error:
            st.markdown(
                f'<div class="result-warn" style="padding:12px 14px; margin-top:6px;">'
                f'<span style="font-size:0.82rem;">⚠️ {st.session_state.login_error}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p class="mono" style="color:var(--text-muted); font-size:0.72rem; margin-top:14px;">'
            'Demo credentials — admin / admin</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# STAGE 3 — WAITING / AUTH TRANSITION
# ===========================================================

def render_waiting() -> None:
    inject_css()
    st.markdown('<div class="splash-shell">', unsafe_allow_html=True)
    st.markdown('<div class="sf-mark sf-mark-lg" style="margin:0 auto;">🛡️</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="splash-title" style="font-size:1.7rem;">Welcome back, {st.session_state.user_name or "Admin"}</div>', unsafe_allow_html=True)
    st.markdown('<div class="splash-sub">Preparing your monitoring session</div>', unsafe_allow_html=True)

    checklist_slot = st.empty()
    steps = ["AUTHENTICATING USER", "LOADING AI MODEL", "PREPARING DASHBOARD", "SYNCHRONIZING SERVICES"]
    revealed = []

    for step in steps:
        revealed.append(step)
        rows = "".join(
            f'<div class="checklist-item"><span>{s}</span><span class="ok">✓</span></div>'
            for s in revealed
        )
        checklist_slot.markdown(rows, unsafe_allow_html=True)
        time.sleep(0.25)

    time.sleep(0.25)
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.stage = "app"
    st.rerun()


# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================

NAV_GROUPS = [
    ("OVERVIEW", [("Dashboard", "🏠")]),
    ("MONITORING", [("Live Monitor", "🎥"), ("Analyze Video", "📁")]),
    ("INTELLIGENCE", [("Incidents", "🚨"), ("Analytics", "📈"), ("AI Performance", "🧠")]),
    ("SYSTEM", [("Cameras", "📹"), ("System Health", "🩺"), ("Settings", "⚙️"), ("Profile", "👤")]),
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom: 18px;">
                <div class="sf-mark">🛡️</div>
                <div>
                    <div style="font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1rem;">SafeFall AI</div>
                    <div style="color:var(--text-muted); font-size:0.7rem;">v1.0 · Monitoring Console</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="navwrap">', unsafe_allow_html=True)
        for group_name, items in NAV_GROUPS:
            st.markdown(
                f'<div class="sf-panel-label" style="margin-top:14px;">{group_name}</div>',
                unsafe_allow_html=True,
            )
            for label, icon in items:
                is_active = st.session_state.page == label
                wrap_class = "navwrap-active" if is_active else ""
                st.markdown(f'<div class="{wrap_class}">', unsafe_allow_html=True)
                if st.button(f"{icon}  {label}", key=f"nav_{label}"):
                    st.session_state.page = label
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="sf-panel-label">Quick Settings</span>', unsafe_allow_html=True)
        st.session_state.confidence_threshold = st.slider(
            "Fall confidence threshold",
            0.50, 0.99, st.session_state.confidence_threshold, 0.01,
        )
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <span class="sf-panel-label" style="margin-bottom:4px;">Active Threshold</span>
                <div class="mono" style="font-size:1.25rem; font-weight:600; color:var(--emerald);">
                    {st.session_state.confidence_threshold * 100:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        active_incidents = sum(1 for i in st.session_state.incidents if i["status"] == "Confirmed")
        st.markdown(
            f"""
            <div class="glass glass-tight">
                <span class="sf-panel-label">Session</span>
                <p style="font-size:0.82rem; color:var(--text); margin:0 0 2px 0;">{st.session_state.user_name or "Admin"}</p>
                <p class="mono" style="font-size:0.72rem; color:var(--text-muted); margin:0;">{active_incidents} open incident(s)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("⏻  Sign Out", key="sign_out"):
            st.session_state.stage = "login"
            st.session_state.page = "Dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# PAGE — DASHBOARD
# ===========================================================

def page_dashboard() -> None:
    top_bar(f"Good to see you, {st.session_state.user_name or 'Admin'}", "Your monitoring system is operational")
    vital_line()

    total_incidents = len(st.session_state.incidents)
    confirmed = sum(1 for i in st.session_state.incidents if i["status"] == "Confirmed")

    cards = "".join([
        kpi_card("FALLS DETECTED", str(confirmed), "This session"),
        kpi_card("AI ACCURACY", "94.2%", "Validation set", "ok"),
        kpi_card("ACTIVITY CLASSES", str(len(CLASS_NAMES)), "Tracked"),
        kpi_card("SEQUENCE WINDOW", f"{SEQUENCE_LENGTH} frames", "BiLSTM input"),
    ])
    st.markdown(f'<div class="sys-grid">{cards}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.4, 1], gap="large")

    with col_left:
        st.markdown('<span class="sf-panel-label">Recent Incidents</span>', unsafe_allow_html=True)
        if not st.session_state.incidents:
            st.markdown(
                '<div class="glass" style="color:var(--text-muted); text-align:center; padding: 34px 20px;">'
                'No incidents logged yet. Run a video analysis or live session to populate this feed.</div>',
                unsafe_allow_html=True,
            )
        else:
            for record in st.session_state.incidents[:4]:
                render_incident_card(record, compact=True)

    with col_right:
        st.markdown('<span class="sf-panel-label">System Status</span>', unsafe_allow_html=True)
        cam_pill = "pill-green" if st.session_state.camera_available else "pill-muted"
        cam_label = "ONLINE" if st.session_state.camera_available else "NOT STARTED"
        st.markdown(
            f"""
            <div class="glass">
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <span style="color:var(--text-muted); font-size:0.85rem;">Vision Engine</span>
                    <span class="pill pill-green">ONLINE</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <span style="color:var(--text-muted); font-size:0.85rem;">Temporal Model</span>
                    <span class="pill pill-green">ONLINE</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <span style="color:var(--text-muted); font-size:0.85rem;">Alert Pipeline</span>
                    <span class="pill pill-green">ARMED</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--text-muted); font-size:0.85rem;">Live Camera</span>
                    <span class="pill {cam_pill}">{cam_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================
# PAGE — LIVE MONITOR
# ===========================================================

def page_live_monitor(pose_detector, predictor, validator, alert_manager) -> None:
    top_bar("Live Monitor", "Real-time detection from a connected camera")

    if not st.session_state.camera_checked:
        probe = cv2.VideoCapture(0)
        st.session_state.camera_available = probe.isOpened()
        probe.release()
        st.session_state.camera_checked = True

    col_video, col_analysis = st.columns([1.4, 1], gap="large")

    with col_video:
        st.markdown('<span class="sf-panel-label">Camera 01 · Feed</span>', unsafe_allow_html=True)
        video_slot = st.empty()

        if not st.session_state.camera_available:
            video_slot.markdown(
                """
                <div class="glass" style="text-align:center; padding: 60px 24px; color: var(--text-muted);">
                    <div style="font-size:1.8rem; margin-bottom:10px;">📵</div>
                    <div style="font-family:'Space Grotesk', sans-serif; color:var(--text); font-weight:600; font-size:1.05rem;">
                        No camera detected
                    </div>
                    <div style="font-size:0.85rem; margin-top:6px; max-width:340px; margin-left:auto; margin-right:auto;">
                        SafeFall AI could not open a video device on this host. Connect a camera and reload,
                        or use <b>Analyze Video</b> to run detection on an uploaded recording instead.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
            if st.button("↻  Re-check for camera"):
                st.session_state.camera_checked = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            controls = st.columns(2)
            with controls[0]:
                if not st.session_state.monitoring_active:
                    if st.button("●  Start Live Monitoring"):
                        st.session_state.monitoring_active = True
                        st.rerun()
            with controls[1]:
                st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
                if st.button("■  Stop"):
                    st.session_state.monitoring_active = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    with col_analysis:
        st.markdown('<span class="sf-panel-label">Live Analysis</span>', unsafe_allow_html=True)
        activity_slot = st.empty()
        render_idle_activity_card(activity_slot)

    if st.session_state.camera_available and st.session_state.monitoring_active:
        capture = cv2.VideoCapture(0)
        sequence = []
        batch_frames = 90  # short bounded batch per script run, then rerun to stay responsive to Stop

        for _ in range(batch_frames):
            if not st.session_state.monitoring_active:
                break

            success, frame = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            keypoints = pose_detector.extract_keypoints(frame)
            sequence.append(keypoints)

            if len(sequence) >= SEQUENCE_LENGTH:
                window = np.asarray(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)
                result = predictor.predict(window)
                label = result["label"]
                confidence = result["confidence"]
                is_alert_state = confidence >= st.session_state.confidence_threshold

                render_activity_card(activity_slot, label, confidence, is_alert_state)

                confirmed = validator.update(label, confidence)
                if confirmed:
                    alert = alert_manager.trigger()
                    log_incident(alert, confidence, "Live Camera")

        capture.release()

        if st.session_state.monitoring_active:
            time.sleep(0.05)
            st.rerun()


# ===========================================================
# PAGE — ANALYZE VIDEO  (core detection loop — unchanged)
# ===========================================================

def page_analyze_video(pose_detector, predictor, validator, alert_manager) -> None:
    top_bar("Analyze a Recording", "Upload a surveillance clip for offline analysis")

    uploaded_video = st.file_uploader(
        "Upload a surveillance video",
        type=["avi", "mp4", "mov"],
        label_visibility="collapsed",
    )

    if uploaded_video is None:
        st.markdown(
            """
            <div class="glass" style="text-align:center; padding: 40px 24px; color: var(--text-muted);">
                <div style="font-size:1.6rem; margin-bottom:8px;">📼</div>
                <div style="font-family:'Space Grotesk', sans-serif; color:var(--text); font-weight:600; font-size:1.05rem;">
                    No recording loaded
                </div>
                <div style="font-size:0.85rem; margin-top:6px;">
                    Upload an AVI, MP4, or MOV file above to run pose extraction and fall-risk analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_video, col_activity = st.columns([1.3, 1], gap="large")

    with col_video:
        st.markdown('<span class="sf-panel-label">Source Video</span>', unsafe_allow_html=True)
        st.video(uploaded_video)
        analyze_clicked = st.button("▶  Analyze Video")

    with col_activity:
        st.markdown('<span class="sf-panel-label">Live Analysis</span>', unsafe_allow_html=True)
        activity_placeholder = st.empty()
        render_idle_activity_card(activity_placeholder)

    if not analyze_clicked:
        return

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temporary:

        temporary.write(
            uploaded_video.read()
        )

        video_path = (
            temporary.name
        )

    capture = cv2.VideoCapture(
        video_path
    )

    sequence = []

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sf-panel-label">Processing</span>', unsafe_allow_html=True)

    step_status = st.empty()
    progress = st.progress(0)

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if frame_count <= 0:
        frame_count = 1

    frame_index = 0

    detected_events = []

    step_status.markdown(
        '<span class="mono" style="color:var(--text-muted); font-size:0.82rem;">'
        'Extracting frames and running pose + temporal inference…</span>',
        unsafe_allow_html=True,
    )

    while True:

        success, frame = (
            capture.read()
        )

        if not success:
            break

        keypoints = (
            pose_detector
            .extract_keypoints(
                frame
            )
        )

        sequence.append(
            keypoints
        )

        if len(sequence) >= SEQUENCE_LENGTH:

            window = np.asarray(
                sequence[
                    -SEQUENCE_LENGTH:
                ],
                dtype=np.float32
            )

            result = (
                predictor.predict(
                    window
                )
            )

            label = result[
                "label"
            ]

            confidence = result[
                "confidence"
            ]

            is_alert_state = confidence >= st.session_state.confidence_threshold

            render_activity_card(
                activity_placeholder,
                label,
                confidence,
                is_alert_state,
            )

            confirmed = (
                validator.update(
                    label,
                    confidence
                )
            )

            if confirmed:

                alert = (
                    alert_manager
                    .trigger()
                )

                detected_events.append(
                    alert
                )

                log_incident(alert, confidence, "Uploaded Video")

        frame_index += 1

        progress.progress(
            min(
                frame_index /
                frame_count,
                1.0
            )
        )

    capture.release()

    step_status.markdown(
        '<span class="mono" style="color:var(--emerald); font-size:0.82rem;">✓ Analysis complete</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sf-panel-label">Result</span>', unsafe_allow_html=True)

    if detected_events:
        render_result_critical(detected_events[-1])
    else:
        render_result_success()


# ===========================================================
# PAGE — INCIDENTS
# ===========================================================

def render_incident_card(record: dict, compact: bool = False) -> None:
    pill_class = "pill-red" if record["status"] == "Confirmed" else "pill-green"
    st.markdown(
        f"""
        <div class="incident-card">
            <div class="incident-top">
                <div>
                    <div class="incident-id">🔴 INCIDENT #{record['id']}</div>
                    <div class="incident-meta">{record['timestamp'].strftime('%d %b · %H:%M:%S')} &nbsp;·&nbsp; {record['source']}</div>
                </div>
                <span class="pill {pill_class}">{record['status'].upper()}</span>
            </div>
            <div class="incident-meta" style="margin-top:10px;">Confidence &nbsp; <span style="color:var(--text);">{record['confidence'] * 100:.1f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not compact:
        with st.expander("Details"):
            st.markdown(f'<div class="mono" style="font-size:0.8rem; color:var(--text-muted);">{record["raw"]}</div>', unsafe_allow_html=True)
            if record["status"] == "Confirmed":
                if st.button("Mark Resolved", key=f"resolve_{record['id']}"):
                    record["status"] = "Resolved"
                    st.rerun()


def page_incidents() -> None:
    top_bar("Incidents", f"{len(st.session_state.incidents)} total this session")

    if not st.session_state.incidents:
        st.markdown(
            '<div class="glass" style="text-align:center; padding: 46px 24px; color: var(--text-muted);">'
            'No incidents logged yet. Confirmed falls from Live Monitor or Analyze Video will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    filter_choice = st.radio(
        "Filter",
        ["All", "Confirmed", "Resolved"],
        horizontal=True,
        label_visibility="collapsed",
    )

    for record in st.session_state.incidents:
        if filter_choice != "All" and record["status"] != filter_choice:
            continue
        render_incident_card(record)


# ===========================================================
# PAGE — ANALYTICS  (illustrative — no historical data store exists)
# ===========================================================

def page_analytics() -> None:
    top_bar("Monitoring Analytics", "Illustrative trends — connect a data store to make this live")

    st.markdown(
        '<div class="result-warn" style="padding:12px 16px; margin-bottom:18px;">'
        '<span style="font-size:0.82rem;">ℹ️ This page uses sample data for layout purposes — SafeFall AI does not '
        'yet persist historical detections beyond the current session.</span></div>',
        unsafe_allow_html=True,
    )

    hours = pd.DataFrame({
        "hour": [f"{h:02d}:00" for h in range(0, 24, 2)],
        "falls": [0, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0],
    })
    class_labels = list(CLASS_NAMES) if CLASS_NAMES else ["Walking", "Standing", "Sitting", "Fall"]
    activity = pd.DataFrame({"activity": class_labels})
    activity["share"] = np.linspace(0.42, 0.06, len(activity))
    activity["share"] = activity["share"] / activity["share"].sum()

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<span class="sf-panel-label">Falls by Hour</span>', unsafe_allow_html=True)
        chart = alt.Chart(hours).mark_bar(color="#34D399", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("hour", title=None), y=alt.Y("falls", title=None),
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)

    with col_b:
        st.markdown('<span class="sf-panel-label">Activity Distribution</span>', unsafe_allow_html=True)
        chart = alt.Chart(activity).mark_arc(innerRadius=55).encode(
            theta="share", color=alt.Color("activity", scale=alt.Scale(scheme="teals"), legend=alt.Legend(title=None)),
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)

    st.markdown('<span class="sf-panel-label">Detection Confidence Trend</span>', unsafe_allow_html=True)
    trend = pd.DataFrame({
        "day": pd.date_range(end=pd.Timestamp.today(), periods=14).strftime("%d %b"),
        "confidence": np.clip(np.random.default_rng(7).normal(0.9, 0.04, 14), 0.75, 0.99),
    })
    chart = alt.Chart(trend).mark_line(color="#22D3EE", point=alt.OverlayMarkDef(color="#22D3EE")).encode(
        x=alt.X("day", title=None), y=alt.Y("confidence", title=None, scale=alt.Scale(domain=[0.7, 1.0])),
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)


# ===========================================================
# PAGE — AI PERFORMANCE  (illustrative — no eval harness wired up)
# ===========================================================

def page_ai_performance() -> None:
    top_bar("AI Performance", "Model: SafeFall-v1 · BiLSTM temporal classifier")

    st.markdown(
        '<div class="result-warn" style="padding:12px 16px; margin-bottom:18px;">'
        '<span style="font-size:0.82rem;">ℹ️ Metrics below are illustrative placeholders — wire up your '
        'evaluation script output to replace them with real numbers.</span></div>',
        unsafe_allow_html=True,
    )

    cards = "".join([
        kpi_card("ACCURACY", "94.2%"),
        kpi_card("PRECISION", "92.8%"),
        kpi_card("RECALL", "95.1%"),
        kpi_card("F1 SCORE", "93.9%"),
    ])
    st.markdown(f'<div class="sys-grid">{cards}</div>', unsafe_allow_html=True)

    classes = list(CLASS_NAMES) if CLASS_NAMES else ["Walking", "Standing", "Sitting", "Fall"]
    rng = np.random.default_rng(3)
    matrix = rng.integers(2, 40, size=(len(classes), len(classes)))
    for i in range(len(classes)):
        matrix[i, i] = rng.integers(80, 140)

    records = []
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            records.append({"actual": actual, "predicted": predicted, "count": int(matrix[i, j])})
    cm_df = pd.DataFrame(records)

    st.markdown('<span class="sf-panel-label">Confusion Matrix</span>', unsafe_allow_html=True)
    heat = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X("predicted", title="Predicted"), y=alt.Y("actual", title="Actual"),
        color=alt.Color("count", scale=alt.Scale(scheme="teals"), legend=None),
    ).properties(height=260)
    text = alt.Chart(cm_df).mark_text(color="#0A0E0D").encode(
        x="predicted", y="actual", text="count",
    )
    st.altair_chart(heat + text, use_container_width=True)


# ===========================================================
# PAGE — CAMERAS  (illustrative multi-camera list; Camera 01 checked live)
# ===========================================================

def page_cameras() -> None:
    top_bar("Camera Management", "Connected video sources")

    if not st.session_state.camera_checked:
        probe = cv2.VideoCapture(0)
        st.session_state.camera_available = probe.isOpened()
        probe.release()
        st.session_state.camera_checked = True

    cam01_status = "pill-green" if st.session_state.camera_available else "pill-red"
    cam01_label = "ONLINE" if st.session_state.camera_available else "OFFLINE"

    cameras = [
        {"name": "Camera 01", "room": "Local device (index 0)", "pill": cam01_status, "status": cam01_label},
        {"name": "Camera 02", "room": "Bedroom — not configured", "pill": "pill-muted", "status": "NOT CONFIGURED"},
        {"name": "Camera 03", "room": "Hallway — not configured", "pill": "pill-muted", "status": "NOT CONFIGURED"},
    ]

    cols = st.columns(3, gap="medium")
    for col, cam in zip(cols, cameras):
        with col:
            st.markdown(
                f"""
                <div class="glass" style="text-align:center;">
                    <div style="font-size:1.6rem; margin-bottom:6px;">📹</div>
                    <div style="font-family:'Space Grotesk', sans-serif; font-weight:700;">{cam['name']}</div>
                    <div style="color:var(--text-muted); font-size:0.8rem; margin:4px 0 12px 0;">{cam['room']}</div>
                    <span class="pill {cam['pill']}">{cam['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="mono" style="color:var(--text-muted); font-size:0.72rem; margin-top:18px;">'
        'Only Camera 01 is wired to a real device check. Cameras 02–03 are shown for layout only until '
        'multi-camera support is added.</p>',
        unsafe_allow_html=True,
    )


# ===========================================================
# PAGE — SYSTEM HEALTH
# ===========================================================

def page_system_health() -> None:
    top_bar("System Health", "Pipeline diagnostics")

    cpu_pct, mem_pct = None, None
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem_pct = psutil.virtual_memory().percent
    except Exception:
        pass

    rows = [
        ("AI MODEL", "ONLINE", "pill-green"),
        ("VISION ENGINE", "ONLINE", "pill-green"),
        ("CAMERA 01", "ONLINE" if st.session_state.camera_available else "OFFLINE",
         "pill-green" if st.session_state.camera_available else "pill-red"),
        ("ALERT PIPELINE", "ARMED", "pill-green"),
    ]

    st.markdown('<span class="sf-panel-label">Component Status</span>', unsafe_allow_html=True)
    for label, status, pill in rows:
        st.markdown(
            f"""
            <div class="glass glass-tight" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span class="mono" style="font-size:0.85rem;">{label}</span>
                <span class="pill {pill}">{status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sf-panel-label">Resource Usage</span>', unsafe_allow_html=True)

    if cpu_pct is not None:
        st.markdown(
            f"""
            <div class="glass">
                <div class="metric-row">
                    <div class="m-top"><span>CPU</span><span>{cpu_pct:.0f}%</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width:{cpu_pct:.0f}%;"></div></div>
                </div>
                <div class="metric-row">
                    <div class="m-top"><span>Memory</span><span>{mem_pct:.0f}%</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width:{mem_pct:.0f}%;"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="glass" style="color:var(--text-muted); font-size:0.85rem;">'
            'Install <code>psutil</code> to display live CPU / memory usage here.</div>',
            unsafe_allow_html=True,
        )


# ===========================================================
# PAGE — SETTINGS
# ===========================================================

def page_settings() -> None:
    top_bar("Settings", "Detection, alerts, and appearance")

    st.markdown('<span class="sf-panel-label">Detection</span>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.session_state.confidence_threshold = st.slider(
        "Fall confirmation confidence threshold",
        0.50, 0.99, st.session_state.confidence_threshold, 0.01,
    )
    st.session_state.confirmation_window = st.slider(
        "Frames required for temporal confirmation",
        2, 20, st.session_state.confirmation_window, 1,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sf-panel-label">Alerts</span>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.session_state.sound_alerts = st.toggle("Sound alerts on confirmed fall", value=st.session_state.sound_alerts)
    st.toggle("Desktop notifications", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="sf-panel-label">Appearance</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass" style="color:var(--text-muted); font-size:0.85rem;">'
        'SafeFall AI runs in dark mode only, tuned for low-light monitoring rooms.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="mono" style="color:var(--text-muted); font-size:0.72rem; margin-top:16px;">'
        'The confirmation-window control is illustrative — wire it into TemporalValidator to make it live.</p>',
        unsafe_allow_html=True,
    )


# ===========================================================
# PAGE — PROFILE
# ===========================================================

def page_profile() -> None:
    top_bar("Profile", "Account details")

    st.markdown(
        f"""
        <div class="glass" style="display:flex; align-items:center; gap:18px;">
            <div class="sf-mark sf-mark-lg">👤</div>
            <div>
                <div style="font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.15rem;">{st.session_state.user_name or "Admin"}</div>
                <div style="color:var(--text-muted); font-size:0.85rem;">Administrator</div>
                <div class="mono" style="color:var(--text-muted); font-size:0.75rem; margin-top:4px;">Signed in this session</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    stats = [
        ("Incidents logged", str(len(st.session_state.incidents))),
        ("Confidence threshold", f"{st.session_state.confidence_threshold * 100:.0f}%"),
        ("Camera status", "Online" if st.session_state.camera_available else "Offline"),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(kpi_card(label.upper(), value), unsafe_allow_html=True)


# ===========================================================
# SHARED WIDGETS — activity card / results
# ===========================================================

def render_activity_card(placeholder, label: str, confidence: float, is_alert_state: bool) -> None:
    bar_class = "warn" if is_alert_state else ""
    icon = "⚠️" if is_alert_state else "🧍"
    placeholder.markdown(
        f"""
        <div class="activity-card">
            <span class="activity-label">Current Activity</span>
            <div class="activity-value">{icon}&nbsp; {label}</div>
            <div class="metric-row">
                <div class="m-top"><span>Model Confidence</span><span>{confidence * 100:.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill {bar_class}" style="width:{confidence * 100:.1f}%;"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_idle_activity_card(placeholder) -> None:
    placeholder.markdown(
        """
        <div class="activity-card">
            <span class="activity-label">Current Activity</span>
            <div class="activity-value">— Awaiting analysis</div>
            <div class="metric-row">
                <div class="m-top"><span>Model Confidence</span><span>0.0%</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:0%;"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_success() -> None:
    st.markdown(
        """
        <div class="result-safe">
            <div class="result-title">✅ No confirmed fall detected</div>
            <div class="result-body">The temporal validator reviewed the full sequence and found no window that
            crossed the confirmed-fall threshold. No incident has been logged.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_critical(alert) -> None:
    st.markdown(
        f"""
        <div class="result-critical">
            <div class="result-title">🔴 FALL DETECTED</div>
            <div class="result-body">The temporal validator confirmed a fall event during this recording.
            An incident has been logged — review it from the Incidents page.</div>
            <div class="result-meta">{alert}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================
# MAIN APP SHELL
# ===========================================================

def render_app() -> None:
    inject_css()

    try:
        pose_detector, predictor, validator, alert_manager = load_system()
    except Exception as error:
        st.markdown(
            """
            <div class="result-critical">
                <div class="result-title">⚠️ Model failed to load</div>
                <div class="result-body">SafeFall AI could not initialize the detection pipeline. See the
                diagnostic trace below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.exception(error)
        st.stop()

    render_sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Live Monitor":
        page_live_monitor(pose_detector, predictor, validator, alert_manager)
    elif page == "Analyze Video":
        page_analyze_video(pose_detector, predictor, validator, alert_manager)
    elif page == "Incidents":
        page_incidents()
    elif page == "Analytics":
        page_analytics()
    elif page == "AI Performance":
        page_ai_performance()
    elif page == "Cameras":
        page_cameras()
    elif page == "System Health":
        page_system_health()
    elif page == "Settings":
        page_settings()
    elif page == "Profile":
        page_profile()

    st.markdown(
        '<div class="sf-footnote">SAFEFALL AI &nbsp;·&nbsp; VISION ENGINE + BiLSTM TEMPORAL ACTIVITY RECOGNITION '
        '&nbsp;·&nbsp; ON-DEVICE INFERENCE</div>',
        unsafe_allow_html=True,
    )


# ===========================================================
# ROUTER
# ===========================================================

init_state()

if st.session_state.stage == "splash":
    render_splash()
elif st.session_state.stage == "login":
    render_login()
elif st.session_state.stage == "waiting":
    render_waiting()
else:
    render_app()

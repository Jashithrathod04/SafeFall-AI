import tempfile
import time
from pathlib import Path

import cv2
@@ -27,34 +28,452 @@
)


# ===========================================================
# PAGE CONFIG
# ===========================================================

st.set_page_config(
    page_title="SafeFall AI",
    page_title="SafeFall AI — Fall Detection",
    page_icon="🛡️",
    layout="wide"
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
# ===========================================================
# DESIGN SYSTEM — TOKENS
# ===========================================================
# Palette:
#   Background   #0A0E0D  (deep graphite, faint green undertone)
#   Surface      rgba(20, 27, 25, 0.55)  (glass panel)
#   Border       rgba(255, 255, 255, 0.08)
#   Text primary #E9EFEC
#   Text muted   #8CA099
#   Accent AI    #34D399  (emerald — "system nominal")
#   Accent info  #22D3EE  (cyan — data / analysis)
#   Accent warn  #FBBF24  (amber — attention)
#   Accent crit  #F87171  (red — fall / emergency)
#
# Type:
#   Display  — Space Grotesk  (headings, brand)
#   Body     — Inter          (copy, labels)
#   Data     — JetBrains Mono (metrics, timestamps, confidence)
#
# Signature element: a continuous animated vital-sign / ECG line —
# ties the "monitoring a person's safety" premise to the literal
# visual language of patient monitoring, used under the masthead
# and as an idle-state motif.
# ===========================================================


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root{
            --bg:#0A0E0D;
            --surface: rgba(20, 27, 25, 0.55);
            --surface-solid:#121917;
            --border: rgba(255,255,255,0.08);
            --border-strong: rgba(255,255,255,0.16);
            --text:#E9EFEC;
            --text-muted:#8CA099;
            --emerald:#34D399;
            --emerald-dim: rgba(52,211,153,0.14);
            --cyan:#22D3EE;
            --amber:#FBBF24;
            --red:#F87171;
            --red-dim: rgba(248,113,113,0.12);
            --radius-lg:20px;
            --radius-md:14px;
            --radius-sm:10px;
        }

        html, body, [class*="css"]{
            font-family:'Inter', sans-serif;
        }

        /* ---------- app background ---------- */
        .stApp{
            background:
                radial-gradient(ellipse 900px 500px at 15% -10%, rgba(52,211,153,0.07), transparent 60%),
                radial-gradient(ellipse 700px 500px at 100% 10%, rgba(34,211,238,0.05), transparent 55%),
                linear-gradient(180deg, #0A0E0D 0%, #0B100F 100%);
            color: var(--text);
        }

        #MainMenu, footer, header[data-testid="stHeader"]{ background: transparent; }
        .block-container{ padding-top: 2.2rem; max-width: 1180px; }

        /* ---------- typography ---------- */
        h1, h2, h3, .display-font{
            font-family:'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }
        .mono{ font-family:'JetBrains Mono', monospace !important; }

        /* ---------- masthead ---------- */
        .sf-masthead{
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding: 4px 2px 0 2px;
            margin-bottom: 6px;
        }
        .sf-brand{
            display:flex;
            align-items:center;
            gap:14px;
        }
        .sf-mark{
            width:44px; height:44px;
            border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, rgba(52,211,153,0.18), rgba(34,211,238,0.10));
            border: 1px solid var(--border-strong);
            font-size:20px;
            box-shadow: 0 0 24px rgba(52,211,153,0.12);
        }
        .sf-title{
            font-family:'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            line-height:1.1;
            margin:0;
            letter-spacing:-0.02em;
        }
        .sf-subtitle{
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top:2px;
            letter-spacing: 0.02em;
        }
        .sf-status{
            display:flex; align-items:center; gap:8px;
            font-family:'JetBrains Mono', monospace;
            font-size:0.72rem;
            letter-spacing:0.06em;
            color: var(--emerald);
            border: 1px solid rgba(52,211,153,0.25);
            background: var(--emerald-dim);
            padding: 7px 14px;
            border-radius: 999px;
        }
        .sf-dot{
            width:7px; height:7px; border-radius:50%;
            background: var(--emerald);
            box-shadow: 0 0 8px var(--emerald);
            animation: pulseDot 2s ease-in-out infinite;
        }
        @keyframes pulseDot{
            0%, 100%{ opacity:1; transform:scale(1); }
            50%{ opacity:0.45; transform:scale(0.8); }
        }

        /* ---------- vital line (signature element) ---------- */
        .sf-vitalwrap{
            width:100%; height:34px; margin: 14px 0 22px 0;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            overflow:hidden;
            opacity:0.8;
        }
        .sf-vitalwrap svg{ width:200%; height:100%; animation: vitalScroll 9s linear infinite; }
        @keyframes vitalScroll{
            from{ transform: translateX(0); }
            to{ transform: translateX(-50%); }
        }

        /* ---------- glass panels ---------- */
        .glass{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 22px 24px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }
        .glass-tight{ padding: 16px 18px; }

        .sf-panel-label{
            font-family:'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 10px;
            display:block;
        }

        /* ---------- system status grid ---------- */
        .sys-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin: 4px 0 26px 0; }
        .sys-card{
            background: var(--surface);
            border:1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px 18px;
        }
        .sys-card .k{ color:var(--text-muted); font-size:0.72rem; letter-spacing:0.05em; }
        .sys-card .v{ font-family:'Space Grotesk', sans-serif; font-size:1.15rem; font-weight:700; margin-top:6px; display:flex; align-items:center; gap:8px; }
        .chip-ok{ width:6px; height:6px; border-radius:50%; background:var(--emerald); box-shadow:0 0 6px var(--emerald); }

        /* ---------- upload dropzone ---------- */
        [data-testid="stFileUploaderDropzone"]{
            background: var(--surface) !important;
            border: 1.5px dashed var(--border-strong) !important;
            border-radius: var(--radius-lg) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover{
            border-color: var(--emerald) !important;
            box-shadow: 0 0 0 1px rgba(52,211,153,0.25);
        }
        [data-testid="stFileUploaderDropzoneInstructions"] svg{ display:none; }
        [data-testid="stFileUploader"] section > button{
            background: transparent !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }

        /* ---------- buttons ---------- */
        .stButton > button{
            width:100%;
            background: linear-gradient(135deg, #34D399, #22D3EE);
            color:#062018;
            font-weight:700;
            font-family:'Space Grotesk', sans-serif;
            letter-spacing: 0.01em;
            border:none;
            border-radius: var(--radius-sm);
            padding: 0.65rem 1rem;
            box-shadow: 0 8px 24px rgba(52,211,153,0.18);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover{
            transform: translateY(-1px);
            box-shadow: 0 10px 28px rgba(52,211,153,0.28);
        }
        .stButton > button:active{ transform: translateY(0px) scale(0.99); }

        /* ---------- slider ---------- */
        [data-testid="stSlider"] [role="slider"]{ background-color: var(--emerald) !important; }
        div[data-baseweb="slider"] > div > div{ background: var(--emerald) !important; }

        /* ---------- progress bar ---------- */
        div[data-testid="stProgress"] > div > div{
            background: linear-gradient(90deg, #34D399, #22D3EE) !important;
        }

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, #0B100F 0%, #0A0E0D 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .block-container{ padding-top: 1.6rem; }

        /* ---------- live analysis card ---------- */
        .activity-card{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 22px 24px;
        }
        .activity-label{
            color: var(--text-muted);
            font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase;
            font-family:'JetBrains Mono', monospace;
        }
        .activity-value{
            font-family:'Space Grotesk', sans-serif;
            font-size: 1.9rem; font-weight:700; margin: 4px 0 16px 0;
        }
        .metric-row{ margin-bottom: 12px; }
        .metric-row .m-top{ display:flex; justify-content:space-between; font-size:0.78rem; color:var(--text-muted); margin-bottom:5px; font-family:'JetBrains Mono', monospace; }
        .bar-track{ width:100%; height:7px; border-radius:999px; background: rgba(255,255,255,0.06); overflow:hidden; }
        .bar-fill{ height:100%; border-radius:999px; background: linear-gradient(90deg, #34D399, #22D3EE); transition: width 0.4s ease; }
        .bar-fill.warn{ background: linear-gradient(90deg, #FBBF24, #F87171); }

        /* ---------- result cards ---------- */
        .result-critical{
            background: var(--red-dim);
            border: 1px solid rgba(248,113,113,0.35);
            border-radius: var(--radius-lg);
            padding: 24px 26px;
            box-shadow: 0 0 40px rgba(248,113,113,0.10);
            animation: alertGlow 2.4s ease-in-out infinite;
        }
        @keyframes alertGlow{
            0%, 100%{ box-shadow: 0 0 24px rgba(248,113,113,0.08); }
            50%{ box-shadow: 0 0 46px rgba(248,113,113,0.20); }
        }
        .result-safe{
            background: var(--emerald-dim);
            border: 1px solid rgba(52,211,153,0.3);
            border-radius: var(--radius-lg);
            padding: 24px 26px;
        }
        .result-title{ font-family:'Space Grotesk', sans-serif; font-size:1.25rem; font-weight:700; display:flex; align-items:center; gap:10px; }
        .result-body{ color: var(--text-muted); margin-top:8px; font-size:0.92rem; }
        .result-meta{ font-family:'JetBrains Mono', monospace; font-size:0.8rem; color: var(--text); margin-top:14px; opacity:0.85; }

        /* ---------- footer note ---------- */
        .sf-footnote{
            color: var(--text-muted);
            font-size: 0.72rem;
            text-align:center;
            margin-top: 40px;
            letter-spacing: 0.03em;
            font-family:'JetBrains Mono', monospace;
        }

        hr{ border-color: var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title(
    "🛡️ SafeFall AI"
)

st.subheader(
    "AI-Powered Elderly Fall Detection System"
)
def render_vital_line() -> None:
    """Signature ambient element — a looping ECG-style trace."""
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


def render_masthead() -> None:
    st.markdown(
        """
        <div class="sf-masthead">
            <div class="sf-brand">
                <div class="sf-mark">🛡️</div>
                <div>
                    <p class="sf-title">SafeFall AI</p>
                    <p class="sf-subtitle">Computer-vision fall detection for elderly care</p>
                </div>
            </div>
            <div class="sf-status"><span class="sf-dot"></span> SYSTEM ONLINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_vital_line()


def render_system_grid() -> None:
    st.markdown(
        f"""
        <div class="sys-grid">
            <div class="sys-card">
                <div class="k">VISION ENGINE</div>
                <div class="v"><span class="chip-ok"></span>Pose Detector</div>
            </div>
            <div class="sys-card">
                <div class="k">TEMPORAL MODEL</div>
                <div class="v"><span class="chip-ok"></span>BiLSTM · seq {SEQUENCE_LENGTH}</div>
            </div>
            <div class="sys-card">
                <div class="k">ACTIVITY CLASSES</div>
                <div class="v"><span class="chip-ok"></span>{len(CLASS_NAMES)} tracked</div>
            </div>
            <div class="sys-card">
                <div class="k">ALERT PIPELINE</div>
                <div class="v"><span class="chip-ok"></span>Armed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

st.write(
    "Computer vision + YOLOv8 Pose + "
    "BiLSTM temporal activity recognition."
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
            Review the details below and escalate according to your monitoring protocol.</div>
            <div class="result-meta">{alert}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

# ===========================================================
# RENDER — STATIC CHROME
# ===========================================================

inject_css()
render_masthead()
render_system_grid()


# ===========================================================
# LOAD MODELS  (unchanged core logic)
# ===========================================================

@st.cache_resource
def load_system():
@@ -86,56 +505,112 @@ def load_system():

except Exception as error:

    st.error(
        "The AI model could not be loaded."
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


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
# ===========================================================
# SIDEBAR — SETTINGS  (unchanged variable: confidence_threshold)
# ===========================================================

st.sidebar.header(
    "Monitoring Settings"
)
with st.sidebar:

confidence_threshold = st.sidebar.slider(
    "Fall confidence threshold",
    0.50,
    0.99,
    0.70,
    0.01
)
    st.markdown('<span class="sf-panel-label">Monitoring Settings</span>', unsafe_allow_html=True)

st.sidebar.info(
    "The system uses temporal confirmation "
    "to reduce false alarms."
)
    confidence_threshold = st.slider(
        "Fall confidence threshold",
        0.50,
        0.99,
        0.70,
        0.01,
    )

    st.markdown(
        f"""
        <div class="glass glass-tight" style="margin-top:6px;">
            <span class="sf-panel-label" style="margin-bottom:6px;">Active Threshold</span>
            <div class="mono" style="font-size:1.4rem; font-weight:600; color:var(--emerald);">
                {confidence_threshold * 100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass glass-tight">
            <span class="sf-panel-label">How this works</span>
            <p style="color:var(--text-muted); font-size:0.84rem; line-height:1.5; margin:0;">
            SafeFall AI uses temporal confirmation across multiple frames before raising an alert —
            a single uncertain frame will not trigger a false alarm.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass glass-tight">
            <span class="sf-panel-label">Pipeline</span>
            <p class="mono" style="color:var(--text-muted); font-size:0.78rem; line-height:1.9; margin:0;">
            01&nbsp; Pose extraction<br/>
            02&nbsp; Sequence buffering<br/>
            03&nbsp; BiLSTM inference<br/>
            04&nbsp; Temporal validation<br/>
            05&nbsp; Alert dispatch
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------

# ===========================================================
# VIDEO INPUT
# ---------------------------------------------------------
# ===========================================================

st.markdown('<span class="sf-panel-label">Analyze a Recording</span>', unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Upload a surveillance video",
    type=["avi", "mp4", "mov"]
    type=["avi", "mp4", "mov"],
    label_visibility="collapsed",
)


if uploaded_video is not None:

    st.video(
        uploaded_video
    )
    col_video, col_activity = st.columns([1.3, 1], gap="large")

    if st.button(
        "▶ Analyze Video"
    ):
    with col_video:
        st.markdown('<span class="sf-panel-label">Source Video</span>', unsafe_allow_html=True)
        st.video(uploaded_video)
        analyze_clicked = st.button("▶  Analyze Video")

    with col_activity:
        st.markdown('<span class="sf-panel-label">Live Analysis</span>', unsafe_allow_html=True)
        activity_placeholder = st.empty()
        render_idle_activity_card(activity_placeholder)

    if analyze_clicked:

        with tempfile.NamedTemporaryFile(
            delete=False,
@@ -156,10 +631,10 @@ def load_system():

        sequence = []

        prediction_placeholder = (
            st.empty()
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="sf-panel-label">Processing</span>', unsafe_allow_html=True)

        step_status = st.empty()
        progress = st.progress(0)

        frame_count = int(
@@ -175,6 +650,12 @@ def load_system():

        detected_events = []

        step_status.markdown(
            '<span class="mono" style="color:var(--text-muted); font-size:0.82rem;">'
            'Extracting frames and running pose + temporal inference…</span>',
            unsafe_allow_html=True,
        )

        while True:

            success, frame = (
@@ -218,10 +699,13 @@ def load_system():
                    "confidence"
                ]

                prediction_placeholder.metric(
                    "Current Activity",
                is_alert_state = confidence >= confidence_threshold

                render_activity_card(
                    activity_placeholder,
                    label,
                    f"{confidence * 100:.1f}% confidence"
                    confidence,
                    is_alert_state,
                )

                confirmed = (
@@ -254,20 +738,41 @@ def load_system():

        capture.release()

        st.divider()
        step_status.markdown(
            '<span class="mono" style="color:var(--emerald); font-size:0.82rem;">✓ Analysis complete</span>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="sf-panel-label">Result</span>', unsafe_allow_html=True)

        if detected_events:

            st.error(
                "🚨 FALL DETECTED"
            )

            st.write(
                detected_events[-1]
            )
            render_result_critical(detected_events[-1])

        else:

            st.success(
                "No confirmed fall detected."
            )
            render_result_success()

else:
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


st.markdown(
    '<div class="sf-footnote">SAFEFALL AI &nbsp;·&nbsp; VISION ENGINE + BiLSTM TEMPORAL ACTIVITY RECOGNITION '
    '&nbsp;·&nbsp; ON-DEVICE INFERENCE</div>',
    unsafe_allow_html=True,
)

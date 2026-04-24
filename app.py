"""
app.py — AI Study Companion  (Streamlit front-end)
Rule-based attention detection using MediaPipe Face Mesh + OpenCV.

Run:
    streamlit run app.py
"""

import os, time, threading, queue, datetime
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils import (
    extract_landmarks, compute_ear,
    estimate_head_pose, annotate_frame,
    AttentionClassifier, SessionStats, save_session_summary,
    format_duration, STATE_FOCUSED, STATE_DISTRACTED,
    STATE_SLEEPING, STATE_NO_FACE, STATE_COLORS,
)

# ─── Optional alert sound (winsound = stdlib on Windows) ─────────────────
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False   # non-Windows OS

# ══════════════════════════════════════════════════════════════════════════
#  Page config & CSS
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Study Companion",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; }

/* ── Global dark background ── */
.stApp { background: #0b1326; color: #dae2fd; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Page header ── */
.app-header {
    text-align: left;
    padding: 1.6rem 0 0.8rem;
    background: rgba(15, 23, 42, 0.95);
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.4rem;
}
.app-title {
    font-size: 3.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    color: #8c909f;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* ── Dashboard Cards ── */
.glass-card {
    background: #171f33;
    border: 1px solid #2d3449;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: #4d8eff;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
}

/* ── Stat tiles ── */
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.stat-tile {
    flex: 1;
    min-width: 80px;
    background: #131b2e;
    border: 1px solid #2d3449;
    border-radius: 12px;
    padding: 0.8rem 0.8rem;
    text-align: left;
}
.stat-label { 
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem; 
    color: #8c909f; 
    text-transform: uppercase; 
    letter-spacing: .05em; 
    font-weight: 600;
}
.stat-value { 
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; 
    font-weight: 600; 
    color: #dae2fd;
    margin-top: 0.2rem; 
    line-height: 1;
}
.stat-pct   { font-size: 0.75rem; color: #8c909f; margin-top: 0.2rem; }

/* ── State badge ── */
.state-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-size: 1.1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: .03em;
    text-transform: uppercase;
}

/* ── Progress bars ── */
.prog-wrap { margin: 0.5rem 0 0.4rem; }
.prog-label { font-size: 0.75rem; color: #c2c6d6; font-weight: 600; display: flex; justify-content: space-between; }
.prog-bar-bg { background: #1e293b; border-radius: 999px; height: 6px; margin-top: 4px; }
.prog-bar-fill { border-radius: 999px; height: 6px; }

/* ── Alert banner ── */
.alert-banner {
    background: rgba(255, 180, 171, 0.1);
    border: 1px solid rgba(255, 180, 171, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #ffb4ab;
    margin-bottom: 0.6rem;
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}

/* ── Break banner ── */
.break-banner {
    background: rgba(74, 225, 118, 0.1);
    border: 1px solid rgba(74, 225, 118, 0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #4ae176;
    margin-bottom: 0.6rem;
}

/* ── Event log ── */
.event-row { display: flex; gap: 0.6rem; align-items: center;
             font-size: 0.75rem; padding: 0.4rem 0;
             border-bottom: 1px solid #1e293b; }
.event-time { color: #8c909f; min-width: 72px; font-family: 'Space Grotesk', sans-serif;}

/* ── Buttons ── */
.stButton>button {
    width: 100%;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.2rem !important;
    border: 1px solid #2d3449 !important;
    background: #171f33 !important;
    color: #dae2fd !important;
    transition: all .2s ease !important;
}
.stButton>button:hover { 
    background: #1e293b !important;
    border-color: #4d8eff !important;
    box-shadow: 0 0 10px rgba(77, 142, 255, 0.2) !important;
}

/* Primary Button Override */
.stButton>button[data-baseweb="button"]:has(div:contains("Start")) {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
}
.stButton>button[data-baseweb="button"]:has(div:contains("Start")):hover {
    background: #2563eb !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
}

/* ── Webcam placeholder ── */
.cam-placeholder {
    background: #060e20;
    border: 1px solid #2d3449;
    border-radius: 16px;
    height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8c909f;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: inset 0 0 40px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "running":       False,
        "session_stats": None,
        "classifier":    None,
        "alerts":        [],
        "last_state":    STATE_NO_FACE,
        "ear":           0.0,
        "pitch":         0.0,
        "yaw":           0.0,
        "roll":          0.0,
        "alert_played":  False,
        "frame_count":   0,
        "chat_history":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════
#  Helper: render HTML stat tile
# ══════════════════════════════════════════════════════════════════════════
STATE_BADGE_COLORS = {
    STATE_FOCUSED:    ("#4ae176", "#003111"),
    STATE_DISTRACTED: ("#ec6a06", "#4a1c00"),
    STATE_SLEEPING:   ("#ffb4ab", "#690005"),
    STATE_NO_FACE:    ("#8c909f", "#1e293b"),
}
STATE_ICONS = {
    STATE_FOCUSED:    "•",
    STATE_DISTRACTED: "•",
    STATE_SLEEPING:   "•",
    STATE_NO_FACE:    "•",
}

def render_state_badge(state: str) -> str:
    fg, bg = STATE_BADGE_COLORS.get(state, ("#9ca3af", "#1f2937"))
    icon   = STATE_ICONS.get(state, "?")
    return (
        f'<span class="state-badge" '
        f'style="color:{fg};background:{bg};border:1px solid {fg}44;">'
        f'{icon} {state}</span>'
    )

def render_progress(label: str, pct: float, color: str) -> str:
    w = max(0, min(100, pct))
    return (
        f'<div class="prog-wrap">'
        f'<div class="prog-label"><span>{label}</span><span>{pct:.1f}%</span></div>'
        f'<div class="prog-bar-bg"><div class="prog-bar-fill" '
        f'style="width:{w}%;background:{color};"></div></div>'
        f'</div>'
    )

def render_stat_tile(label: str, value: str, pct: str = "", color: str = "#e8eaf0") -> str:
    return (
        f'<div class="stat-tile">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color};">{value}</div>'
        f'{"<div class=stat-pct>" + pct + "</div>" if pct else ""}'
        f'</div>'
    )

# ══════════════════════════════════════════════════════════════════════════
#  Optional: play alert in background thread
# ══════════════════════════════════════════════════════════════════════════
def _play_alert():
    """Play a short beep using the built-in winsound module (Windows only)."""
    if SOUND_AVAILABLE:
        try:
            # 880 Hz for 400 ms
            winsound.Beep(880, 400)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════
#  Sidebar: ML Data Collection Mode
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 ML Data Collection")
    st.info("Record facial features (EAR, Pitch, Yaw, Roll) to train a Deep Learning model later.")
    
    st.session_state["collect_data"] = st.checkbox("Enable Data Collection", value=False)
    
    st.session_state["ml_label"] = st.selectbox(
        "Target Label to Record:", 
        [STATE_FOCUSED, STATE_DISTRACTED, STATE_SLEEPING]
    )
    
    if st.session_state["collect_data"]:
        st.warning(f"When session starts, features will be appended to `ml_dataset.csv` as **{st.session_state['ml_label']}**.")
        
        try:
            sz = os.path.getsize("ml_dataset.csv")
            st.success(f"Dataset size: {sz / 1024:.1f} KB")
        except FileNotFoundError:
            st.caption("Dataset file will be created automatically.")

# ══════════════════════════════════════════════════════════════════════════
#  Page header
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <p class="app-title">AI Study Companion</p>
  <p class="app-sub">Real-time attention tracking</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  Tabs & Layout
# ══════════════════════════════════════════════════════════════════════════
tab_live, tab_coach = st.tabs(["Live Tracker", "AI Coach"])

col_cam, col_dash = tab_live.columns([3, 2], gap="medium")

# ── LEFT: camera feed ─────────────────────────────────────────────────────
with col_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    cam_placeholder   = st.empty()
    state_placeholder = st.empty()
    alert_placeholder = st.empty()

    cam_placeholder.markdown(
        '<div class="cam-placeholder">📷 Camera will appear here when session starts</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── controls row ──
    c1, c2, c3 = st.columns(3)
    with c1:
        start_btn = st.button("▶ Start Session", type="primary",
                              disabled=st.session_state["running"])
    with c2:
        stop_btn  = st.button("⏹ Stop Session",
                              disabled=not st.session_state["running"])
    with c3:
        save_btn  = st.button("💾 Save Summary",
                              disabled=st.session_state["session_stats"] is None)

# ── RIGHT: dashboard ──────────────────────────────────────────────────────
with col_dash:
    # Current state card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Current State")
    current_state_ph = st.empty()
    current_state_ph.markdown(
        render_state_badge(STATE_NO_FACE), unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats tiles
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Session Stats")
    stats_ph = st.empty()
    stats_ph.markdown(
        '<div class="stat-row">'
        + render_stat_tile("Session Time", "00:00:00")
        + render_stat_tile("Focused",      "00:00:00", color="#00c875")
        + render_stat_tile("Distracted",   "00:00:00", color="#f97316")
        + render_stat_tile("Sleeping",     "00:00:00", color="#e53e3e")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Progress bars
    prog_ph = st.empty()
    prog_ph.markdown(
        render_progress("Focus",       0.0, "#4ae176")
        + render_progress("Distracted", 0.0, "#ec6a06")
        + render_progress("Sleeping",   0.0, "#ffb4ab"),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Pose metrics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Head-Pose Metrics")
    pose_ph = st.empty()
    pose_ph.markdown(
        '<div class="stat-row">'
        + render_stat_tile("EAR",   "—")
        + render_stat_tile("Pitch", "—")
        + render_stat_tile("Yaw",   "—")
        + render_stat_tile("Roll",  "—")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Event log
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### State Change Log")
    log_ph = st.empty()
    log_ph.markdown("<p style='color:#4b5563;font-size:0.8rem;'>No events yet.</p>",
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab_coach:
    st.markdown("### Post-Session AI Coach")
    
    stats = st.session_state.get("session_stats")
    if stats is None or stats.total_sec() < 5:
        st.info("Start and finish a study session first! I will analyze your performance here once you are done.")
    else:
        # Display chat history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat input
        if prompt := st.chat_input("Ask me for study tips or to analyze your session..."):
            
            # Show user message
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate rule-based response
            summary = stats.summary_dict()
            f_pct = summary["focus_pct"]
            d_pct = summary["distract_pct"]
            s_pct = summary["sleep_pct"]
            
            p_lower = prompt.lower()
            if "summary" in p_lower or "analyze" in p_lower or "how did i do" in p_lower:
                if f_pct >= 80:
                    response = f"🌟 **Excellent Session!** You were focused **{f_pct}%** of the time. You kept distractions to a minimum. Keep up this amazing momentum!"
                elif d_pct > 30:
                    response = f"👀 **Distraction Alert!** You were distracted for **{d_pct}%** of the session. Try putting your phone in another room or blocking distracting websites."
                elif s_pct > 10:
                    response = f"😴 **Drowsiness Detected.** You spent **{s_pct}%** of the session sleepy. It might be time to take a quick 20-minute power nap or grab some coffee!"
                else:
                    response = f"📊 **Session Overview:** You were focused for **{f_pct}%**, distracted for **{d_pct}%**, and sleepy for **{s_pct}%**. A solid effort, but there is still room to improve your deep work streaks."
            elif "tip" in p_lower or "improve" in p_lower or "help" in p_lower:
                if d_pct > f_pct:
                    response = "💡 **Study Tip:** Since you get distracted easily, try the **Pomodoro Technique**: 25 minutes of intense focus followed by a 5-minute break. Don't touch your phone until the break!"
                elif s_pct > 0:
                    response = "💡 **Study Tip:** To fight drowsiness, ensure your study room is brightly lit, sit up straight, and stay hydrated. Avoid studying in bed at all costs."
                else:
                    response = "💡 **Study Tip:** You are doing great! To reach the next level of focus, try listening to binaural beats or lo-fi music without lyrics to block out background noise."
            else:
                response = "I'm your rule-based AI Coach! Try asking me to **'analyze my session'** or give you a **'study tip'** based on your recent performance."
                
            # Show AI response
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# ══════════════════════════════════════════════════════════════════════════
#  Button handlers
# ══════════════════════════════════════════════════════════════════════════
if start_btn:
    st.session_state["running"]       = True
    st.session_state["session_stats"] = SessionStats()
    st.session_state["classifier"]    = AttentionClassifier()
    st.session_state["alerts"]        = []
    st.session_state["frame_count"]   = 0
    st.rerun()

if stop_btn:
    st.session_state["running"] = False
    st.rerun()

if save_btn and st.session_state["session_stats"]:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_session_summary(st.session_state["session_stats"],
                                f"session_{ts}.csv")
    st.success(f"✅ Session saved to `{path}`")

# ══════════════════════════════════════════════════════════════════════════
#  Main capture loop  (runs only when session is active)
# ══════════════════════════════════════════════════════════════════════════
if st.session_state["running"]:

    stats      = st.session_state["session_stats"]
    classifier = st.session_state["classifier"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO)

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        
        last_ts = 0

        while st.session_state["running"]:

            ret, frame = cap.read()
            if not ret:
                break

            st.session_state["frame_count"] += 1
            h, w = frame.shape[:2]

            # ── MediaPipe inference ────────────────────────────────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            current_ms = int(time.time() * 1000)
            if current_ms <= last_ts:
                current_ms = last_ts + 1
            last_ts = current_ms
            
            result = face_landmarker.detect_for_video(mp_image, current_ms)

            ear   = 0.0
            pitch = yaw = roll = 0.0
            state = STATE_NO_FACE

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                lm_arr  = extract_landmarks(face_lm, w, h)

                ear             = compute_ear(lm_arr)
                pitch, yaw, roll = estimate_head_pose(lm_arr, w, h)
                state           = classifier.classify(ear, pitch, yaw, roll)
                
                # ── ML Data Collection ─────────────────────────────────
                if st.session_state.get("collect_data"):
                    file_exists = os.path.isfile("ml_dataset.csv")
                    with open("ml_dataset.csv", "a") as f:
                        if not file_exists:
                            f.write("timestamp_ms,ear,pitch,yaw,roll,label\n")
                        label = st.session_state["ml_label"]
                        f.write(f"{current_ms},{ear:.5f},{pitch:.5f},{yaw:.5f},{roll:.5f},{label}\n")

            else:
                classifier.reset()

            # ── Accumulate session stats ───────────────────────────
            new_alerts = stats.update(state)

            # ── Alert handling ─────────────────────────────────────
            for a in new_alerts:
                if a not in st.session_state["alerts"]:
                    st.session_state["alerts"].append(a)
                    if not st.session_state.get("alert_played"):
                        threading.Thread(target=_play_alert, daemon=True).start()
                        st.session_state["alert_played"] = True

            # clear old alerts after 15 s
            now = time.time()
            st.session_state["alerts"] = [
                a for a in st.session_state["alerts"]
                if "distracted" not in a.lower()
                   or state == STATE_DISTRACTED
            ]

            # ── Annotate frame ─────────────────────────────────────
            frame = annotate_frame(frame, state, ear, pitch, yaw, roll)
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ── Update UI ──────────────────────────────────────────
            cam_placeholder.image(img, channels="RGB", use_container_width=True)

            # state badge
            current_state_ph.markdown(
                render_state_badge(state), unsafe_allow_html=True
            )

            # alerts
            alert_html = ""
            for a in st.session_state["alerts"]:
                cls = "break-banner" if "☕" in a else "alert-banner"
                alert_html += f'<div class="{cls}">{a}</div>'
            if alert_html:
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # stats tiles
            s = stats.summary_dict()
            stats_ph.markdown(
                '<div class="stat-row">'
                + render_stat_tile("Session Time",
                                   format_duration(s["total_sec"]))
                + render_stat_tile("Focused",
                                   format_duration(s["focused_sec"]),
                                   f'{s["focus_pct"]:.1f}%', "#00c875")
                + render_stat_tile("Distracted",
                                   format_duration(s["distracted_sec"]),
                                   f'{s["distract_pct"]:.1f}%', "#f97316")
                + render_stat_tile("Sleeping",
                                   format_duration(s["sleeping_sec"]),
                                   f'{s["sleep_pct"]:.1f}%', "#e53e3e")
                + "</div>",
                unsafe_allow_html=True,
            )

            # progress bars
            prog_ph.markdown(
                render_progress("Focus",        s["focus_pct"],    "#00c875")
                + render_progress("Distracted", s["distract_pct"], "#f97316")
                + render_progress("Sleeping",   s["sleep_pct"],    "#e53e3e"),
                unsafe_allow_html=True,
            )

            # pose metrics
            pose_ph.markdown(
                '<div class="stat-row">'
                + render_stat_tile("EAR",   f"{ear:.3f}")
                + render_stat_tile("Pitch", f"{pitch:+.1f}°")
                + render_stat_tile("Yaw",   f"{yaw:+.1f}°")
                + render_stat_tile("Roll",  f"{roll:+.1f}°")
                + "</div>",
                unsafe_allow_html=True,
            )

            # event log (last 10)
            events = s["events"][-10:][::-1]
            if events:
                rows = ""
                for ts, st_ev in events:
                    t_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    ic    = STATE_ICONS.get(st_ev, "?")
                    fg, _ = STATE_BADGE_COLORS.get(st_ev, ("#9ca3af", "#111"))
                    rows += (
                        f'<div class="event-row">'
                        f'<span class="event-time">⏱ {t_str}</span>'
                        f'<span style="color:{fg};">{ic} {st_ev}</span>'
                        f'</div>'
                    )
                log_ph.markdown(rows, unsafe_allow_html=True)

            # throttle to ~20 fps to reduce CPU load
            time.sleep(0.05)

    cap.release()
    st.session_state["running"] = False
    st.rerun()

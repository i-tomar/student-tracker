import os, time, threading, queue, datetime
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Import your custom logic from utils.py
from utils import (
    extract_landmarks, compute_ear,
    estimate_head_pose, annotate_frame,
    AttentionClassifier, SessionStats, save_session_summary,
    format_duration, STATE_FOCUSED, STATE_DISTRACTED,
    STATE_SLEEPING, STATE_NO_FACE, STATE_COLORS
)

# --- Alert sound setup (Works locally on Windows only) ---
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False 

# ==========================================
# Page Config & Styles
# ==========================================
st.set_page_config(page_title="AI Study Companion", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1326; color: #dae2fd; }
#MainMenu, footer { visibility: hidden; }
.app-header { text-align: left; padding: 1.6rem 0 0.8rem; background: rgba(15, 23, 42, 0.95); border-bottom: 1px solid #1e293b; margin-bottom: 1.4rem; }
.app-title { font-size: 3.6rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0; }
.app-sub { font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; color: #8c909f; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-top: 0.25rem; }
.glass-card { background: #171f33; border: 1px solid #2d3449; border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.stat-tile { flex: 1; min-width: 80px; background: #131b2e; border: 1px solid #2d3449; border-radius: 12px; padding: 0.8rem 0.8rem; }
.stat-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.65rem; color: #8c909f; text-transform: uppercase; font-weight: 600; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 600; color: #dae2fd; margin-top: 0.2rem; }
.state-badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 999px; font-size: 1.1rem; font-weight: 600; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; }
.prog-wrap { margin: 0.5rem 0 0.4rem; }
.prog-label { font-size: 0.75rem; color: #c2c6d6; font-weight: 600; display: flex; justify-content: space-between; }
.prog-bar-bg { background: #1e293b; border-radius: 999px; height: 6px; margin-top: 4px; }
.prog-bar-fill { border-radius: 999px; height: 6px; }
.alert-banner { background: rgba(255, 180, 171, 0.1); border: 1px solid rgba(255, 180, 171, 0.2); border-radius: 12px; padding: 0.8rem 1rem; color: #ffb4ab; margin-bottom: 0.6rem; animation: pulse 2s infinite; }
.break-banner { background: rgba(74, 225, 118, 0.1); border: 1px solid rgba(74, 225, 118, 0.2); border-radius: 12px; padding: 0.8rem 1rem; color: #4ae176; margin-bottom: 0.6rem; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Data Bridge Setup
# ==========================================
# This Queue passes data from the Camera Thread to the UI Thread
result_queue = queue.Queue(maxsize=10)

def _init_state():
    if "session_stats" not in st.session_state:
        st.session_state["session_stats"] = SessionStats()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "latest_summary" not in st.session_state:
        st.session_state["latest_summary"] = None

_init_state()

# --- UI Helpers ---
STATE_BADGE_COLORS = {
    STATE_FOCUSED: ("#4ae176", "#003111"), STATE_DISTRACTED: ("#ec6a06", "#4a1c00"),
    STATE_SLEEPING: ("#ffb4ab", "#690005"), STATE_NO_FACE: ("#8c909f", "#1e293b"),
}

def render_state_badge(state: str) -> str:
    fg, bg = STATE_BADGE_COLORS.get(state, ("#9ca3af", "#1f2937"))
    return f'<span class="state-badge" style="color:{fg};background:{bg};border:1px solid {fg}44;">â€¢ {state}</span>'

def render_stat_tile(label: str, value: str, pct: str = "", color: str = "#e8eaf0") -> str:
    return f'<div class="stat-tile"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color};">{value}</div>{f"<div style=font-size:0.7rem;color:#8c909f>{pct}</div>" if pct else ""}</div>'

def render_progress(label: str, pct: float, color: str) -> str:
    return f'<div class="prog-wrap"><div class="prog-label"><span>{label}</span><span>{pct:.1f}%</span></div><div class="prog-bar-bg"><div style="width:{pct}%;background:{color};height:6px;border-radius:999px;"></div></div></div>'

# ==========================================
# Video Processor (The Brain)
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.classifier = AttentionClassifier()
        self.stats = SessionStats() # Keep stats local to the camera thread
        
        # Initialize MediaPipe
        base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_ts = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Process Frame
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        curr_ms = int(time.time() * 1000)
        if curr_ms <= self.last_ts: curr_ms = self.last_ts + 1
        self.last_ts = curr_ms
        
        result = self.detector.detect_for_video(mp_image, curr_ms)
        
        state = STATE_NO_FACE
        ear = 0.0
        pitch = yaw = roll = 0.0
        
        if result.face_landmarks:
            face_lm = result.face_landmarks[0]
            lm_arr = extract_landmarks(face_lm, w, h)
            ear = compute_ear(lm_arr)
            pitch, yaw, roll = estimate_head_pose(lm_arr, w, h)
            state = self.classifier.classify(ear, pitch, yaw, roll)
        else:
            self.classifier.reset()

        # Update stats
        self.stats.update(state)
        
        # Send data to dashboard
        if not result_queue.full():
            result_queue.put({
                "state": state, "ear": ear, 
                "pose": (pitch, yaw, roll), 
                "summary": self.stats.summary_dict()
            })

        # Visual feedback on camera
        img = annotate_frame(img, state, ear, pitch, yaw, roll)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Main Layout
# ==========================================
st.markdown('<div class="app-header"><p class="app-title">AI Study Companion</p><p class="app-sub">Real-time attention tracking</p></div>', unsafe_allow_html=True)

tab_live, tab_coach = st.tabs(["Live Tracker", "AI Coach"])
col_cam, col_dash = tab_live.columns([3, 2], gap="medium")

with col_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    # The Webcam Component
    ctx = webrtc_streamer(
        key="student-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_dash:
    # 1. State Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Current State")
    state_ph = st.empty()
    state_ph.markdown(render_state_badge(STATE_NO_FACE), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Stats Dashboard
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Session Stats")
    stats_ph = st.empty()
    prog_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Head Pose
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Head-Pose Metrics")
    pose_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# Live Dashboard Update Loop
# ==========================================
# This part "catches" the data from the camera and fills the placeholders above
while ctx.state.playing:
    try:
        data = result_queue.get(timeout=1.0)
        
        # Update Dashboard Elements
        state_ph.markdown(render_state_badge(data["state"]), unsafe_allow_html=True)
        
        s = data["summary"]
        stats_ph.markdown(f'''
            <div class="stat-row">
                {render_stat_tile("Session", format_duration(s["total_sec"]))}
                {render_stat_tile("Focus", format_duration(s["focused_sec"]), f"{s['focus_pct']:.0f}%", "#4ae176")}
                {render_stat_tile("Distract", format_duration(s["distracted_sec"]), f"{s['distract_pct']:.0f}%", "#ec6a06")}
            </div>
        ''', unsafe_allow_html=True)

        prog_ph.markdown(
            render_progress("Focus Intensity", s["focus_pct"], "#4ae176") +
            render_progress("Distraction Level", s["distract_pct"], "#ec6a06"),
            unsafe_allow_html=True
        )

        p, y, r = data["pose"]
        pose_ph.markdown(f'''
            <div class="stat-row">
                {render_stat_tile("EAR", f"{data['ear']:.2f}")}
                {render_stat_tile("Pitch", f"{p:+.0f}Â°")}
                {render_stat_tile("Yaw", f"{y:+.0f}Â°")}
            </div>
        ''', unsafe_allow_html=True)
        
        # Save to session state so AI Coach can see it later
        st.session_state["latest_summary"] = s

    except queue.Empty:
        continue

# ==========================================
# AI Coach Tab
# ==========================================
with tab_coach:
    st.markdown("### Post-Session AI Coach")
    summary = st.session_state["latest_summary"]
    
    if not summary or summary["total_sec"] < 5:
        st.info("ðŸ“ Start a study session in the 'Live Tracker' tab first!")
    else:
        # Display session highlights
        st.write(f"Based on your last session of **{format_duration(summary['total_sec'])}**:")
        
        # Simple rule-based AI Coach response
        f_pct = summary["focus_pct"]
        if f_pct > 80:
            st.success("ðŸŒŸ **Focus Master:** You maintained high attention. Try the Pomodoro technique to stay fresh.")
        elif summary["distract_pct"] > 30:
            st.warning("ðŸš§ **Distraction Detected:** You looked away frequently. Check for environment noise or phone notifications.")
        elif summary["sleep_pct"] > 10:
            st.error("ðŸ˜´ **Low Energy:** You seem sleepy. Take a 15-minute power nap or stand up and stretch.")

        # Chat interface
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask for study advice..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Simple Assistant logic
            response = "I noticed you were focused for " + str(int(f_pct)) + "%. Try setting a smaller goal next time!"
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

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

# Import your project-specific logic from utils.py
from utils import (
    extract_landmarks, compute_ear,
    estimate_head_pose, annotate_frame,
    AttentionClassifier, SessionStats, save_session_summary,
    format_duration, STATE_FOCUSED, STATE_DISTRACTED,
    STATE_SLEEPING, STATE_NO_FACE, STATE_COLORS, STATE_ICONS
)

# ==========================================
# Page Configuration & CSS
# ==========================================
st.set_page_config(page_title="AI Study Companion", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1326; color: #dae2fd; }
#MainMenu, footer { visibility: hidden; }
.app-header { text-align: left; padding: 1.6rem 0 0.8rem; background: rgba(15, 23, 42, 0.95); border-bottom: 1px solid #1e293b; margin-bottom: 1.4rem; }
.app-title { font-size: 3.2rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0; }
.app-sub { font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; color: #8c909f; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-top: 0.25rem; }
.glass-card { background: #171f33; border: 1px solid #2d3449; border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.stat-tile { flex: 1; min-width: 80px; background: #131b2e; border: 1px solid #2d3449; border-radius: 12px; padding: 0.8rem 0.8rem; }
.stat-label { font-family: 'Space Grotesk', sans-serif; font-size: 0.65rem; color: #8c909f; text-transform: uppercase; font-weight: 600; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 600; color: #dae2fd; margin-top: 0.2rem; }
.state-badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 999px; font-size: 1.1rem; font-weight: 600; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; }
.prog-wrap { margin: 0.5rem 0 0.4rem; }
.prog-label { font-size: 0.75rem; color: #c2c6d6; font-weight: 600; display: flex; justify-content: space-between; }
.prog-bar-bg { background: #1e293b; border-radius: 999px; height: 6px; margin-top: 4px; }
.alert-banner { background: rgba(255, 180, 171, 0.1); border: 1px solid rgba(255, 180, 171, 0.2); border-radius: 12px; padding: 0.8rem 1rem; color: #ffb4ab; margin-bottom: 0.6rem; }
.break-banner { background: rgba(74, 225, 118, 0.1); border: 1px solid rgba(74, 225, 118, 0.2); border-radius: 12px; padding: 0.8rem 1rem; color: #4ae176; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Global Data Bridge
# ==========================================
result_queue = queue.Queue(maxsize=10)

def _init_state():
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
    if "latest_summary" not in st.session_state: st.session_state["latest_summary"] = None
    if "alerts" not in st.session_state: st.session_state["alerts"] = []

_init_state()

# --- UI Helpers ---
STATE_BADGE_COLORS = {
    STATE_FOCUSED: ("#4ae176", "#003111"), STATE_DISTRACTED: ("#ec6a06", "#4a1c00"),
    STATE_SLEEPING: ("#ffb4ab", "#690005"), STATE_NO_FACE: ("#8c909f", "#1e293b"),
}

def render_state_badge(state: str) -> str:
    fg, bg = STATE_BADGE_COLORS.get(state, ("#9ca3af", "#1f2937"))
    return f'<span class="state-badge" style="color:{fg};background:{bg};border:1px solid {fg}44;">{STATE_ICONS.get(state, "â€¢")} {state}</span>'

def render_stat_tile(label: str, value: str, pct: str = "", color: str = "#e8eaf0") -> str:
    return f'<div class="stat-tile"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color};">{value}</div>{f"<div style=font-size:0.7rem;color:#8c909f>{pct}</div>" if pct else ""}</div>'

def render_progress(label: str, pct: float, color: str) -> str:
    return f'<div class="prog-wrap"><div class="prog-label"><span>{label}</span><span>{pct:.1f}%</span></div><div class="prog-bar-bg"><div style="width:{pct}%;background:{color};height:6px;border-radius:999px;"></div></div></div>'

# ==========================================
# Video Processor (AI Brain)
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.classifier = AttentionClassifier()
        self.stats = SessionStats()
        self.detector = None
        self.last_ts = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            
            # Lazy load MediaPipe to prevent startup networking crashes
            if self.detector is None:
                base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1, min_face_detection_confidence=0.5
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)

            # Inference
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            curr_ms = int(time.time() * 1000)
            if curr_ms <= self.last_ts: curr_ms = self.last_ts + 1
            self.last_ts = curr_ms
            
            result = self.detector.detect_for_video(mp_image, curr_ms)
            
            state, ear, pitch, yaw, roll = STATE_NO_FACE, 0.0, 0.0, 0.0, 0.0
            
            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                lm_arr = extract_landmarks(face_lm, w, h)
                ear = compute_ear(lm_arr)
                pitch, yaw, roll = estimate_head_pose(lm_arr, w, h)
                state = self.classifier.classify(ear, pitch, yaw, roll)
            else:
                self.classifier.reset()

            # Update stats & alerts
            new_alerts = self.stats.update(state)
            
            # Send data to dashboard thread
            if not result_queue.full():
                result_queue.put({
                    "state": state, "ear": ear, "pose": (pitch, yaw, roll),
                    "summary": self.stats.summary_dict(), "alerts": new_alerts
                })

            img = annotate_frame(img, state, ear, pitch, yaw, roll)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame # Fallback to raw frame if there's a networking blip

# ==========================================
# UI Main Body
# ==========================================
st.markdown('<div class="app-header"><p class="app-title">AI Study Companion</p><p class="app-sub">Real-time student activity tracker</p></div>', unsafe_allow_html=True)

tab_live, tab_coach = st.tabs(["ðŸ“¹ Live Tracker", "ðŸ§  AI Coach"])
col_cam, col_dash = tab_live.columns([3, 2], gap="medium")

with col_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="activity-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    alert_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with col_dash:
    # State Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Current State")
    state_ph = st.empty()
    state_ph.markdown(render_state_badge(STATE_NO_FACE), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Session Stats")
    stats_ph = st.empty()
    prog_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Pose Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Head-Pose Metrics")
    pose_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# Real-Time UI Update Loop
# ==========================================
while ctx.state.playing:
    try:
        data = result_queue.get(timeout=1.0)
        
        # 1. Update Badge
        state_ph.markdown(render_state_badge(data["state"]), unsafe_allow_html=True)
        
        # 2. Update Alerts
        if data["alerts"]:
            for a in data["alerts"]:
                if a not in st.session_state["alerts"]: st.session_state["alerts"].append(a)
        
        alert_html = ""
        for a in st.session_state["alerts"][-2:]: # Show last 2 alerts
            cls = "break-banner" if "â˜•" in a else "alert-banner"
            alert_html += f'<div class="{cls}">{a}</div>'
        alert_ph.markdown(alert_html, unsafe_allow_html=True)

        # 3. Update Dashboard Stats
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
        
        st.session_state["latest_summary"] = s # Save for AI Coach

    except queue.Empty:
        continue

# ==========================================
# AI Coach Tab
# ==========================================
with tab_coach:
    st.markdown("### Post-Session AI Coach")
    summary = st.session_state["latest_summary"]
    
    if not summary or summary["total_sec"] < 5:
        st.info("ðŸ“ Start a session to see your AI insights here!")
    else:
        st.write(f"Session Analysis (**{format_duration(summary['total_sec'])}**):")
        f_pct = summary["focus_pct"]
        if f_pct > 80: st.success("ðŸŒŸ Excellent focus! You are in the flow state.")
        elif summary["distract_pct"] > 30: st.warning("ðŸš§ High distractions. Try a quieter environment.")
        elif summary["sleep_pct"] > 10: st.error("ðŸ˜´ You look tired. Take a break!")

        # Chat interface
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask for advice..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            response = f"I see you focused for {int(f_pct)}% of the time. That's a great baseline!"
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"): st.markdown(response)

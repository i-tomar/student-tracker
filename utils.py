"""
utils.py — Core logic for AI Study Companion
Handles: landmark extraction, EAR calculation, head-pose estimation,
         state classification, time tracking, alert generation.
No deep-learning model is used — everything is threshold-based.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import tensorflow as tf
import pickle

# MediaPipe Tasks API will be initialized in app.py

# ─────────────────────────── Landmark indices ──────────────────────────────
# MediaPipe 468-landmark Face Mesh indices
# Left eye  (from user's POV camera)
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 6-point EAR landmarks (P1..P6 in the classic Soukupová formula)
LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]   # top-left, top-right, bottom-right, bottom-left + corners
RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]

# Nose tip, chin, left/right eye corners, left/right mouth corners (for solvePnP)
FACE_3D_MODEL_IDXS = [1, 152, 226, 446, 57, 287]   # nose, chin, L-eye, R-eye, L-mouth, R-mouth

# 3-D reference face model (generic, unit-metres scale)
FACE_3D_MODEL_PTS = np.array([
    [0.0,    0.0,    0.0   ],   # nose tip
    [0.0,   -330.0, -65.0  ],   # chin
    [-225.0,  170.0,-135.0 ],   # left eye corner
    [ 225.0,  170.0,-135.0 ],   # right eye corner
    [-150.0, -150.0,-125.0 ],   # left mouth corner
    [ 150.0, -150.0,-125.0 ],   # right mouth corner
], dtype=np.float64)

# ─────────────────────────── Thresholds ────────────────────────────────────
EAR_THRESHOLD          = 0.22   # below → eye closed
EAR_CONSEC_FRAMES      = 15     # consecutive closed frames → drowsy
PITCH_THRESHOLD        = 15.0   # degrees forward/backward tilt
YAW_THRESHOLD          = 20.0   # degrees left/right rotation → distracted
ROLL_THRESHOLD         = 25.0   # degrees tilt
DISTRACT_ALERT_SEC     = 10.0   # alert after N s of continuous distraction
FOCUS_BREAK_SEC        = 25 * 60  # suggest break after 25 min focus (Pomodoro)
DROWSY_CONSEC_SEC      = 2.0    # seconds of closed eyes → sleeping state

# ─────────────────────────── States ────────────────────────────────────────
STATE_FOCUSED    = "Focused"
STATE_DISTRACTED = "Distracted"
STATE_SLEEPING   = "Sleeping"
STATE_NO_FACE    = "No Face"


# ════════════════════════════════════════════════════════════════════════════
#  Data classes
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SessionStats:
    """Accumulated time per state + event log."""
    start_time:       float = field(default_factory=time.time)
    focused_sec:      float = 0.0
    distracted_sec:   float = 0.0
    sleeping_sec:     float = 0.0
    no_face_sec:      float = 0.0
    current_state:    str   = STATE_NO_FACE
    state_start_time: float = field(default_factory=time.time)
    events:           list  = field(default_factory=list)  # [(timestamp, state)]
    distract_start:   Optional[float] = None   # when continuous distraction started
    focus_start:      Optional[float] = None   # when continuous focus started

    # ── derived helpers ──────────────────────────────────────────────
    def total_sec(self) -> float:
        return time.time() - self.start_time

    def percent(self, sec: float) -> float:
        total = self.total_sec()
        return (sec / total * 100) if total > 0 else 0.0

    def update(self, new_state: str) -> List[str]:
        """Called every frame with the newly classified state.
        Returns list of alert strings (may be empty)."""
        now   = time.time()
        dt    = now - self.state_start_time

        # accumulate time for previous state
        _acc = {
            STATE_FOCUSED:    "focused_sec",
            STATE_DISTRACTED: "distracted_sec",
            STATE_SLEEPING:   "sleeping_sec",
            STATE_NO_FACE:    "no_face_sec",
        }
        attr = _acc.get(self.current_state)
        if attr:
            setattr(self, attr, getattr(self, attr) + dt)

        alerts: List[str] = []

        # state changed → log event
        if new_state != self.current_state:
            self.events.append((now, new_state))
            # reset streak timers
            if new_state == STATE_DISTRACTED:
                self.distract_start = now
                self.focus_start    = None
            elif new_state == STATE_FOCUSED:
                self.focus_start    = now
                self.distract_start = None
            else:
                self.distract_start = None
                self.focus_start    = None

        # check streak alerts
        if self.distract_start and (now - self.distract_start) >= DISTRACT_ALERT_SEC:
            alerts.append(f"⚠️ You've been distracted for {int(now - self.distract_start)}s! Refocus!")

        if self.focus_start and (now - self.focus_start) >= FOCUS_BREAK_SEC:
            alerts.append("☕ You've been focused for 25 minutes. Take a short break!")

        self.current_state    = new_state
        self.state_start_time = now
        return alerts

    def summary_dict(self) -> dict:
        total = self.total_sec()
        return {
            "total_sec":       round(total, 1),
            "focused_sec":     round(self.focused_sec, 1),
            "distracted_sec":  round(self.distracted_sec, 1),
            "sleeping_sec":    round(self.sleeping_sec, 1),
            "focus_pct":       round(self.percent(self.focused_sec), 1),
            "distract_pct":    round(self.percent(self.distracted_sec), 1),
            "sleep_pct":       round(self.percent(self.sleeping_sec), 1),
            "events":          self.events,
        }


# ════════════════════════════════════════════════════════════════════════════
#  Landmark extraction helpers
# ════════════════════════════════════════════════════════════════════════════
def extract_landmarks(face_landmarks_list, img_w: int, img_h: int) -> np.ndarray:
    """Return (468, 3) array of pixel-space x,y and normalised z coords."""
    pts = []
    for lm in face_landmarks_list:
        pts.append([lm.x * img_w, lm.y * img_h, lm.z])
    return np.array(pts, dtype=np.float64)


def get_2d_pts(landmarks: np.ndarray, idxs: List[int]) -> np.ndarray:
    """Slice landmark array to specific indices, returning (N,2) float."""
    return landmarks[idxs, :2].astype(np.float64)


# ════════════════════════════════════════════════════════════════════════════
#  EAR calculation
# ════════════════════════════════════════════════════════════════════════════
def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Classic 6-point EAR (Soukupová & Čech 2016).
    eye_pts: shape (6, 2)  → [left_corner, top1, top2, right_corner, bot1, bot2]
    """
    # Vertical distances
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    # Horizontal distance
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_ear(landmarks: np.ndarray) -> float:
    """Average EAR across both eyes."""
    left_pts  = get_2d_pts(landmarks, LEFT_EAR_PTS)
    right_pts = get_2d_pts(landmarks, RIGHT_EAR_PTS)
    return (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0


# ════════════════════════════════════════════════════════════════════════════
#  Head-pose estimation (solvePnP)
# ════════════════════════════════════════════════════════════════════════════
def estimate_head_pose(landmarks: np.ndarray,
                       img_w: int,
                       img_h: int) -> Tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees using solvePnP.
    Pitch  > 0 → looking down, < 0 → up
    Yaw    > 0 → left,         < 0 → right  (mirror convention)
    Roll   > 0 → tilting right, < 0 → left
    """
    img_pts = get_2d_pts(landmarks, FACE_3D_MODEL_IDXS)

    focal_length = img_w
    cam_matrix   = np.array([
        [focal_length, 0,            img_w / 2],
        [0,            focal_length, img_h / 2],
        [0,            0,            1         ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL_PTS, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(-rot_mat[2, 1], rot_mat[2, 2]))
        yaw   = np.degrees(np.arctan2( rot_mat[2, 0], sy))
        roll  = np.degrees(np.arctan2(-rot_mat[1, 0], rot_mat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
        yaw   = np.degrees(np.arctan2( rot_mat[2, 0], sy))
        roll  = 0.0

    # Normalize angles to [-90, 90] to fix axis flipping (e.g., pitch = -177 -> 3)
    for angle in (pitch, yaw, roll):
        pass # We will reassign them individually
        
    def normalize_angle(a):
        if a > 90: return a - 180
        elif a < -90: return a + 180
        return a
        
    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)

    return float(pitch), float(yaw), float(roll)


# ════════════════════════════════════════════════════════════════════════════
#  State classification
# ════════════════════════════════════════════════════════════════════════════
class AttentionClassifier:
    """Classifies user state using either the Deep Learning model or threshold rules."""

    def __init__(self, use_ml: bool = True):
        self._closed_counter = 0          # consecutive frames with eyes closed
        self._eye_closed_since: Optional[float] = None
        
        self.use_ml = use_ml
        if self.use_ml:
            try:
                self.model = tf.keras.models.load_model("study_companion.keras")
                with open("scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
                # Warmup inference
                _ = self.model(np.zeros((1, 4)), training=False)
            except Exception as e:
                print(f"ML Model load failed ({e}), falling back to rules.")
                self.use_ml = False

    def classify(self, ear: float, pitch: float, yaw: float, roll: float) -> str:
        """Return one of the STATE_* constants."""
        
        if self.use_ml:
            X = np.array([[ear, pitch, yaw, roll]])
            X_scaled = self.scaler.transform(X)
            # model(x) is significantly faster than model.predict(x) for single samples
            preds = self.model(X_scaled, training=False).numpy()[0]
            class_idx = np.argmax(preds)
            
            if class_idx == 0: return STATE_FOCUSED
            elif class_idx == 1: return STATE_DISTRACTED
            else: return STATE_SLEEPING

        # ── Rule-based Fallback ──────────────────────────────────────────────
        if ear < EAR_THRESHOLD:
            self._closed_counter += 1
            if self._eye_closed_since is None:
                self._eye_closed_since = time.time()
        else:
            self._closed_counter   = 0
            self._eye_closed_since = None

        if (self._closed_counter >= EAR_CONSEC_FRAMES or
                (self._eye_closed_since is not None and
                 time.time() - self._eye_closed_since >= DROWSY_CONSEC_SEC)):
            return STATE_SLEEPING

        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD or abs(roll) > ROLL_THRESHOLD:
            return STATE_DISTRACTED

        return STATE_FOCUSED

    def reset(self):
        self._closed_counter   = 0
        self._eye_closed_since = None


# ════════════════════════════════════════════════════════════════════════════
#  Frame annotation helpers
# ════════════════════════════════════════════════════════════════════════════
STATE_COLORS = {
    STATE_FOCUSED:    (0, 200, 80),    # green
    STATE_DISTRACTED: (0, 140, 255),   # orange
    STATE_SLEEPING:   (60, 60, 220),   # red-ish (BGR)
    STATE_NO_FACE:    (150, 150, 150), # grey
}

def annotate_frame(frame: np.ndarray,
                   state: str,
                   ear: float,
                   pitch: float,
                   yaw: float,
                   roll: float) -> np.ndarray:
    """Draw state label, EAR, and pose angles on the frame."""
    color = STATE_COLORS.get(state, (200, 200, 200))
    h, w  = frame.shape[:2]

    # semi-transparent status bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 55), (w, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"State : {state}",
                (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"EAR:{ear:.2f}  P:{pitch:+.1f}  Y:{yaw:+.1f}  R:{roll:+.1f}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

    # coloured border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
    return frame


# ════════════════════════════════════════════════════════════════════════════
#  Session summary CSV export
# ════════════════════════════════════════════════════════════════════════════
def save_session_summary(stats: SessionStats, path: str = "session_summary.csv"):
    """Write session stats + event log to a CSV file."""
    import csv, datetime
    summary = stats.summary_dict()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Session Start", datetime.datetime.fromtimestamp(stats.start_time).isoformat()])
        w.writerow(["Total Duration (s)", summary["total_sec"]])
        w.writerow(["Focused (s)",       summary["focused_sec"]])
        w.writerow(["Distracted (s)",    summary["distracted_sec"]])
        w.writerow(["Sleeping (s)",      summary["sleeping_sec"]])
        w.writerow(["Focus %",           summary["focus_pct"]])
        w.writerow(["Distracted %",      summary["distract_pct"]])
        w.writerow(["Sleeping %",        summary["sleep_pct"]])
        w.writerow([])
        w.writerow(["Timestamp", "State"])
        for ts, st in summary["events"]:
            w.writerow([datetime.datetime.fromtimestamp(ts).isoformat(), st])
    return path


def format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

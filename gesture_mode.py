"""
gesture_mode.py — Mode 1: MediaPipe Hand Gesture Control

Tracks hand landmarks via webcam and maps finger bend angles
to servo positions in real time.

Uses the MediaPipe Tasks API (0.10.x+) with HandLandmarker.
Requires: hand_landmarker.task model file in the project directory.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    FINGER_NAMES, FINGER_JOINTS, NUM_SERVOS,
    SMOOTHING_ALPHA, INVERT_SERVOS,
    STRAIGHT_ANGLE, CURLED_ANGLE,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    SERVO_CENTER,
)

# Path to the MediaPipe hand landmarker model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# MediaPipe hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm
]


# ── Math helpers ────────────────────────────────────────────────────────────

def _vec(a, b):
    """3D vector from landmark a to landmark b (each has x, y, z)."""
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)


def _angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return np.degrees(np.arccos(cos_a))


def _map_angle_to_servo(angle_deg, straight=None, curled=None, invert=False):
    """Map a measured finger bend angle to a 0-180 servo value."""
    straight = straight if straight is not None else STRAIGHT_ANGLE
    curled = curled if curled is not None else CURLED_ANGLE

    lo, hi = min(straight, curled), max(straight, curled)
    clamped = np.clip(angle_deg, lo, hi)

    if lo == hi:
        out = 90.0
    else:
        out = np.interp(clamped, [lo, hi], [0.0, 180.0])

    if invert:
        out = 180.0 - out
    return int(np.clip(round(out), 0, 180))


def _draw_landmarks(frame, landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark dots
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


# ── Gesture Mode Class ─────────────────────────────────────────────────────

class GestureMode:
    """
    Real-time hand tracking → servo control using MediaPipe Tasks API.

    Usage:
        mode = GestureMode()
        mode.start(servo_controller)   # blocking loop
    """

    def __init__(self):
        self.running = False
        self.cap = None
        self.landmarker = None
        self.invert = INVERT_SERVOS
        self.prev_angles = np.full(NUM_SERVOS, SERVO_CENTER, dtype=float)

        # Calibration storage
        self.calib_straight = None
        self.calib_curled = None
        self.calib_step = 0  # 0=none, 1=got straight

        # Latest detection result (updated by callback)
        self._latest_result = None
        self._frame_timestamp_ms = 0

    # ── public API ──────────────────────────────────────────────────────

    def start(self, servo_controller):
        """
        Open the camera and run the tracking loop.
        Returns when the user presses 'q' or 'm', or stop() is called.
        """
        self.running = True
        self._open_camera()
        self._init_landmarker()

        print("[GESTURE] Mode started — press 'q' to quit, 'c' to calibrate, 'i' to invert")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame for MediaPipe
            self._frame_timestamp_ms += 33  # ~30fps
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            # Run detection (async — result comes via callback)
            self.landmarker.detect_async(mp_image, self._frame_timestamp_ms)

            # Process latest result
            finger_angles = self._process_result(frame)
            servo_targets = self._angles_to_servos(finger_angles)

            # Exponential moving average smoothing
            smoothed = (SMOOTHING_ALPHA * np.array(servo_targets, dtype=float)
                        + (1.0 - SMOOTHING_ALPHA) * self.prev_angles)
            self.prev_angles = smoothed
            final = [int(round(a)) for a in smoothed.tolist()]

            # Send to Arduino
            servo_controller.send_angles(final)

            # HUD overlay
            self._draw_hud(frame, final)
            cv2.imshow("Gernoid - Gesture Mode", frame)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("m"):
                self.running = False
                break
            elif key == ord("i"):
                self.invert = not self.invert
                print(f"[GESTURE] Inversion: {self.invert}")
            elif key == ord("c"):
                self._calibrate_step(finger_angles)

        self._cleanup()

    def stop(self):
        """Signal the loop to stop (can be called from another thread)."""
        self.running = False

    # ── internal ─────────────────────────────────────────────────────────

    def _open_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def _result_callback(self, result, output_image, timestamp_ms):
        """Called by MediaPipe when a new detection result is ready."""
        self._latest_result = result

    def _init_landmarker(self):
        """Initialize the MediaPipe HandLandmarker with the Tasks API."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {MODEL_PATH}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task', "
                "'hand_landmarker.task')\""
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=MP_MAX_HANDS,
            min_hand_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            result_callback=self._result_callback,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _process_result(self, frame):
        """Extract finger bend angles from the latest detection result."""
        angles = [0.0] * NUM_SERVOS

        result = self._latest_result
        if result is None or not result.hand_landmarks:
            return angles

        landmarks = result.hand_landmarks[0]  # First hand
        h, w = frame.shape[:2]

        # Draw landmarks on frame
        _draw_landmarks(frame, landmarks, w, h)

        # Compute bend angle for each finger
        for i, fname in enumerate(FINGER_NAMES):
            mcp_idx, pip_idx, tip_idx = FINGER_JOINTS[fname]
            v1 = _vec(landmarks[mcp_idx], landmarks[pip_idx])
            v2 = _vec(landmarks[pip_idx], landmarks[tip_idx])
            ang = _angle_between(v1, v2)
            angles[i] = ang

            # Label on frame
            px = int(landmarks[pip_idx].x * w)
            py = int(landmarks[pip_idx].y * h)
            cv2.putText(frame, f"{fname[:2]}:{int(ang)}",
                        (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)

        return angles

    def _angles_to_servos(self, finger_angles):
        """Convert finger bend angles to servo targets using calibration."""
        if self.calib_straight and self.calib_curled:
            return [
                _map_angle_to_servo(
                    finger_angles[i],
                    straight=self.calib_straight[i],
                    curled=self.calib_curled[i],
                    invert=self.invert,
                )
                for i in range(NUM_SERVOS)
            ]
        else:
            return [
                _map_angle_to_servo(finger_angles[i], invert=self.invert)
                for i in range(NUM_SERVOS)
            ]

    def _calibrate_step(self, current_angles):
        if self.calib_step == 0:
            self.calib_straight = list(current_angles)
            self.calib_step = 1
            print(f"[CALIB] Straight captured: {self.calib_straight}")
            print("[CALIB] Now curl your fingers and press 'c' again.")
        else:
            self.calib_curled = list(current_angles)
            self.calib_step = 0
            print(f"[CALIB] Curled captured: {self.calib_curled}")
            print("[CALIB] Calibration done!")

    def _draw_hud(self, frame, angles):
        cv2.putText(frame, "MODE: GESTURE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "Servos: " + ",".join(map(str, angles)), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Invert: {self.invert}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "[m] switch  [c] calibrate  [q] quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        self._latest_result = None
        self._frame_timestamp_ms = 0
        print("[GESTURE] Mode stopped")

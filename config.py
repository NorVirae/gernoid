"""
config.py — Central configuration for Gernoid robotic arm.

All shared constants and settings live here. Both gesture mode
and AI mode import from this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Serial / Arduino
# =============================================================================
SERIAL_PORT = "COM3"          # Change to your Arduino's port
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.1          # seconds

# =============================================================================
# Camera
# =============================================================================
CAMERA_INDEX = 0              # 0 = default webcam
CAMERA_WIDTH = 1980
CAMERA_HEIGHT = 1080

# =============================================================================
# Servo Tuning
# =============================================================================
NUM_SERVOS = 5
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

SMOOTHING_ALPHA = 0.4         # 1.0 = no smoothing, lower = smoother
INVERT_SERVOS = True          # Flip servo direction globally
SERVO_CENTER = 90             # Neutral/rest position

# Default finger joint angle ranges (degrees)
STRAIGHT_ANGLE = 150.0        # Typical angle when finger is straight
CURLED_ANGLE = 30.0           # Typical angle when finger is curled

# =============================================================================
# Groq / LLM (AI Mode)
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Fallback vision model

# How often AI mode captures + queries the LLM (seconds)
AI_CAPTURE_INTERVAL = 2.0

# =============================================================================
# Grip Presets — fallback servo angles if LLM doesn't give perfect values
# =============================================================================
GRIP_PRESETS = {
    "open":      [0,   0,   0,   0,   0],     # All fingers open
    "closed":    [180, 180, 180, 180, 180],    # Full fist
    "pinch":     [120, 120, 0,   0,   0],      # Thumb + index pinch
    "power":     [150, 160, 160, 160, 160],    # Power grip (grab a bottle)
    "precision": [100, 110, 110, 0,   0],      # Precision grip (3 fingers)
    "point":     [0,   0,   180, 180, 180],    # Index pointing
    "safe":      [45,  45,  45,  45,  45],     # Safe neutral position
}

# =============================================================================
# MediaPipe Hand Tracking
# =============================================================================
MP_MAX_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# Mediapipe joint indices: (MCP, PIP, TIP) for each finger
FINGER_JOINTS = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}

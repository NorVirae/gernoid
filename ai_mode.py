"""
ai_mode.py — Mode 2: AI/LLM Object Recognition Control

Captures frames from the webcam, sends them to Groq's vision model,
and uses the LLM's response to decide whether (and how) to grip
an object with the robotic hand.

Requires: GROQ_API_KEY in your .env file
"""

import cv2
import json
import time
import base64
import numpy as np

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    GROQ_API_KEY, GROQ_MODEL, GROQ_VISION_MODEL,
    AI_CAPTURE_INTERVAL, GRIP_PRESETS, NUM_SERVOS,
    SMOOTHING_ALPHA, SERVO_CENTER,
)

# ── Groq client setup ──────────────────────────────────────────────────────

_groq_client = None


def _get_groq():
    """Lazy-init the Groq client."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            print("[AI] WARNING: No GROQ_API_KEY set. AI mode will use fallback presets only.")
            return None
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
            print("[AI] Groq client initialized")
        except ImportError:
            print("[AI] 'groq' package not installed. Run: pip install groq")
            return None
        except Exception as e:
            print(f"[AI] Failed to init Groq: {e}")
            return None
    return _groq_client


# ── Prompt template ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a robotic hand controller. You control a 5-finger robotic hand with servos.
Each servo ranges from 0 (fully open) to 180 (fully closed).
The fingers are: thumb, index, middle, ring, pinky.

When you see an image, you must:
1. Identify the object in the image.
2. Decide if the robotic hand should grip it.
3. If yes, determine the best grip type and return servo angles.
4. If no (dangerous, too large, or nothing present), return a safe open position.

ALWAYS respond with ONLY valid JSON in this exact format:
{
  "object": "description of what you see",
  "should_grip": true or false,
  "reason": "brief explanation",
  "grip_type": "power|pinch|precision|open|closed|safe",
  "angles": [thumb, index, middle, ring, pinky]
}

Examples of when NOT to grip: sharp objects, hot surfaces, nothing visible, objects too large.
Examples of good grips: bottles (power), small pills (pinch), pens (precision).
"""

USER_PROMPT = "Look at this image from the robot's camera. What do you see? Should the hand grip it?"


# ── AI Mode Class ───────────────────────────────────────────────────────────

class AIMode:
    """
    Camera → LLM → servo angles.

    Usage:
        mode = AIMode()
        mode.start(servo_controller)   # blocking loop
    """

    def __init__(self):
        self.running = False
        self.cap = None
        self.prev_angles = np.full(NUM_SERVOS, SERVO_CENTER, dtype=float)
        self.last_decision = {
            "object": "nothing",
            "should_grip": False,
            "reason": "waiting...",
            "grip_type": "safe",
            "angles": GRIP_PRESETS["safe"],
        }
        self.last_capture_time = 0.0

    # ── public API ──────────────────────────────────────────────────────

    def start(self, servo_controller):
        """
        Open camera and run the AI analysis loop.
        Returns when user presses 'q' or 'm'.
        """
        self.running = True
        self._open_camera()
        print("[AI] Mode started — press 'q' to quit, 'm' to switch, 'g' to force grip query")
        print(f"[AI] Auto-capture every {AI_CAPTURE_INTERVAL}s")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            now = time.time()

            # Periodic capture + LLM query
            if now - self.last_capture_time >= AI_CAPTURE_INTERVAL:
                self.last_capture_time = now
                self._query_llm(frame)

            # Apply current grip decision
            target = self.last_decision.get("angles", GRIP_PRESETS["safe"])
            smoothed = (SMOOTHING_ALPHA * np.array(target, dtype=float)
                        + (1.0 - SMOOTHING_ALPHA) * self.prev_angles)
            self.prev_angles = smoothed
            final = [int(round(a)) for a in smoothed.tolist()]

            servo_controller.send_angles(final)

            # HUD overlay
            self._draw_hud(frame, final)
            cv2.imshow("Gernoid — AI Mode", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("m"):
                self.running = False
                break
            elif key == ord("g"):
                # Force an immediate grip query
                self._query_llm(frame)
            elif key == ord("o"):
                # Quick open hand
                self.last_decision["angles"] = GRIP_PRESETS["open"]
                self.last_decision["grip_type"] = "open"
                print("[AI] Manual override: OPEN")

        self._cleanup()

    def start_dry_run(self, image_path=None):
        """
        Dry-run mode for testing without hardware.
        Uses a test image (or blank frame) and prints what would be sent.
        """
        print("[AI] DRY RUN — no servos will be moved")

        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"[AI] Could not read image: {image_path}")
                return
        else:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "DRY RUN TEST FRAME", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        self._query_llm(frame)
        print(f"\n[DRY RUN RESULT]")
        print(f"  Object:     {self.last_decision['object']}")
        print(f"  Should grip: {self.last_decision['should_grip']}")
        print(f"  Reason:     {self.last_decision['reason']}")
        print(f"  Grip type:  {self.last_decision['grip_type']}")
        print(f"  Angles:     {self.last_decision['angles']}")

    def stop(self):
        self.running = False

    # ── internal ─────────────────────────────────────────────────────────

    def _open_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def _frame_to_base64(self, frame):
        """Encode an OpenCV frame as a base64 JPEG string."""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode("utf-8")

    def _query_llm(self, frame):
        """Send the frame to Groq vision model and parse the grip response."""
        client = _get_groq()
        if client is None:
            # No API key — use a simple fallback
            self._fallback_decision()
            return

        b64_image = self._frame_to_base64(frame)

        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()
            self._parse_response(raw)

        except Exception as e:
            print(f"[AI] LLM query failed: {e}")
            self._fallback_decision()

    def _parse_response(self, raw_text):
        """Parse the LLM JSON response into a grip decision."""
        try:
            # Try to extract JSON from the response (LLM sometimes adds text around it)
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_text[start:end])
            else:
                raise ValueError("No JSON found in response")

            # Validate angles
            angles = data.get("angles", GRIP_PRESETS["safe"])
            if not isinstance(angles, list) or len(angles) != NUM_SERVOS:
                angles = GRIP_PRESETS.get(data.get("grip_type", "safe"), GRIP_PRESETS["safe"])

            angles = [max(0, min(180, int(a))) for a in angles]

            self.last_decision = {
                "object": data.get("object", "unknown"),
                "should_grip": data.get("should_grip", False),
                "reason": data.get("reason", ""),
                "grip_type": data.get("grip_type", "safe"),
                "angles": angles,
            }

            grip_str = "GRIP" if self.last_decision["should_grip"] else "NO GRIP"
            print(f"[AI] {grip_str}: {self.last_decision['object']} "
                  f"({self.last_decision['grip_type']}) → {angles}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[AI] Failed to parse LLM response: {e}")
            print(f"[AI] Raw: {raw_text[:200]}")
            self._fallback_decision()

    def _fallback_decision(self):
        """Use preset grip when LLM is unavailable."""
        self.last_decision = {
            "object": "unknown (fallback)",
            "should_grip": False,
            "reason": "LLM unavailable — using safe position",
            "grip_type": "safe",
            "angles": GRIP_PRESETS["safe"],
        }
        print("[AI] Using fallback: safe position")

    def _draw_hud(self, frame, angles):
        """Draw AI mode info on the camera frame."""
        d = self.last_decision
        color = (0, 255, 0) if d["should_grip"] else (0, 0, 255)

        cv2.putText(frame, "MODE: AI RECOGNITION", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(frame, f"Object: {d['object'][:40]}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        grip_label = "GRIP" if d["should_grip"] else "NO GRIP"
        cv2.putText(frame, f"{grip_label} ({d['grip_type']})", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Reason: {d['reason'][:50]}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "Servos: " + ",".join(map(str, angles)), (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "[m] switch  [g] force query  [o] open  [q] quit",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        print("[AI] Mode stopped")


# ── CLI dry-run entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    mode = AIMode()
    mode.start_dry_run(image_path=img)

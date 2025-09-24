"""
mp_to_servo.py - FIXED VERSION

Requirements:
  pip install mediapipe opencv-python numpy pyserial

Usage:
  - Connect Arduino to a serial port (e.g. COM3 or /dev/ttyUSB0)
  - Edit SERIAL_PORT and SERIAL_BAUD if necessary
  - Run: python mp_to_servo.py
  - Press 'c' to run a quick calibration (first keep hand straight, press 'c', then curl and press 'c' again)
  - Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Optional: send over serial to Arduino
SERIAL_ENABLED = True
SERIAL_PORT = "COM3"     # <-- change to your Arduino port, e.g. "/dev/ttyUSB0"
SERIAL_BAUD = 115200

# Smoothing factor (alpha for EMA). 1.0 = no smoothing, lower = smoother/slower.
SMOOTHING_ALPHA = 0.4

# FIXED: Set to True to invert servo behavior
INVERT_SERVOS = True

# Default calibration angles (measured bend angle -> servo)
# If you don't calibrate: assume 0 deg (straight) -> servo 0, 180 deg (folded) -> servo 180.
CALIB = {
    "straight": None,  # will be set to list of 5 angles or left None
    "curled": None
}

# finger order we'll report: thumb, index, middle, ring, pinky
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Mediapipe indices for joints
# For each finger we use (MCP_index, PIP_index, TIP_index)
FINGER_JOINTS = {
    "thumb": (2, 3, 4),      # thumb_mcp, thumb_ip, thumb_tip
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}

# -----------------------------------------------------------------------------
# serial setup (delayed import so script can still run without pyserial installed)
ser = None
if SERIAL_ENABLED:
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        # small pause to allow Arduino to reset
        time.sleep(1.0)
        print(f"[INFO] Serial opened on {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        print(f"[WARN] Could not open serial {SERIAL_PORT}: {e}\nFalling back to console-only output.")
        ser = None

# -----------------------------------------------------------------------------
def vector(a, b):
    """Return 3D vector b - a given two mediapipe landmarks (each has x,y,z)."""
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)

def angle_between(v1, v2):
    """Compute angle in degrees between vectors v1 and v2."""
    # avoid division by zero
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    # numeric safety
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return np.degrees(np.arccos(cosang))

def map_angle_to_servo(angle_deg, straight_deg=0.0, curled_deg=180.0, invert=False):
    """Map an observed bend angle (0..180) to servo angle 0..180 using calibration."""
    # if calibration values None -> use defaults
    if straight_deg is None: straight_deg = 0.0
    if curled_deg is None: curled_deg = 180.0
    
    # FIXED: Corrected the angle interpretation
    # Typically, smaller joint angles mean more bent fingers
    # So we want: smaller angle -> smaller servo value (more closed)
    #            larger angle -> larger servo value (more open)
    
    # Clamp the angle
    angle_clamped = np.clip(angle_deg, min(straight_deg, curled_deg), max(straight_deg, curled_deg))
    
    # Map linearly - but with corrected interpretation
    if straight_deg == curled_deg:
        out = 90.0  # neutral position if no range
    else:
        # Map so that straight (larger angle) -> high servo value
        # and curled (smaller angle) -> low servo value
        out = np.interp(angle_clamped, [min(straight_deg, curled_deg), max(straight_deg, curled_deg)], [0.0, 180.0])
    
    if invert:
        out = 180.0 - out
    return int(np.clip(round(out), 0, 180))

# Alternative mapping function if the above doesn't work
def alternative_map_angle_to_servo(angle_deg, straight_deg=150.0, curled_deg=30.0, invert=False):
    """Alternative mapping with typical finger joint angle ranges."""
    # Typical finger joint angles:
    # - Straight finger: ~150-180 degrees
    # - Bent finger: ~30-60 degrees
    
    if straight_deg is None: straight_deg = 150.0
    if curled_deg is None: curled_deg = 30.0
    
    angle_clamped = np.clip(angle_deg, min(straight_deg, curled_deg), max(straight_deg, curled_deg))
    
    # Map: bent finger (small angle) -> low servo (closed)
    #      straight finger (large angle) -> high servo (open)
    out = np.interp(angle_clamped, [curled_deg, straight_deg], [0.0, 180.0])
    
    if invert:
        out = 180.0 - out
    return int(np.clip(round(out), 0, 180))

# -----------------------------------------------------------------------------
def send_angles_over_serial(angles):
    """Angles is an iterable of ints [thumb,index,middle,ring,pinky]. We send CSV newline-terminated."""
    global ser
    line = "{},{},{},{},{}\n".format(*angles)
    if ser:
        try:
            ser.write(line.encode("ascii"))
            print(line.strip())  # Remove newline for cleaner console output
        except Exception as e:
            print(f"[WARN] Serial write failed: {e}")
    else:
        print("[SERIAL-FALLBACK] " + line.strip())

# -----------------------------------------------------------------------------
def calibrate_angles(detected_angles):
    """Simple calibration: detected_angles should be list of 5 floats (deg).
       Press 'c' twice: first to capture straight, second to capture curled.
    """
    print("[CALIB] Captured:", detected_angles)
    return detected_angles

# -----------------------------------------------------------------------------
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # previous smoothed servo angles
    prev_angles = np.array([90, 90, 90, 90, 90], dtype=float)  # start centered

    print("[INFO] Press 'c' to calibrate, 'q' to quit.")
    print("[INFO] Press 'i' to toggle inversion on/off")
    calibrating_step = 0  # 0=no, 1=got straight, 2=done
    
    # Add inversion toggle
    current_invert = INVERT_SERVOS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # default angle values if no hand found
        finger_angles_deg = [0.0] * 5

        if result.multi_hand_landmarks:
            # use first detected hand
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # compute bend angle for each finger
            for i, fname in enumerate(FINGER_NAMES):
                mcp_idx, pip_idx, tip_idx = FINGER_JOINTS[fname]
                a = hand_landmarks.landmark[mcp_idx]
                b = hand_landmarks.landmark[pip_idx]
                c = hand_landmarks.landmark[tip_idx]
                v1 = vector(a, b)   # proximal
                v2 = vector(b, c)   # distal
                ang = angle_between(v1, v2)   # degrees
                finger_angles_deg[i] = ang

                # annotate on frame with both angle and servo value
                x_px = int(b.x * frame.shape[1])
                y_px = int(b.y * frame.shape[0])
                cv2.putText(frame, f"{fname[:2]}:{int(ang)}",
                            (x_px+4, y_px-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # map angles to servo using calibration if provided
        if CALIB["straight"] is not None and CALIB["curled"] is not None:
            servo_targets = [
                map_angle_to_servo(finger_angles_deg[i],
                                   straight_deg=CALIB["straight"][i],
                                   curled_deg=CALIB["curled"][i],
                                   invert=current_invert)
                for i in range(5)
            ]
        else:
            # Use alternative mapping with typical finger ranges
            servo_targets = [
                alternative_map_angle_to_servo(finger_angles_deg[i], 
                                             straight_deg=150.0, 
                                             curled_deg=30.0, 
                                             invert=current_invert)
                for i in range(5)
            ]

        # smoothing
        servo_targets = np.array(servo_targets, dtype=float)
        smoothed = (SMOOTHING_ALPHA * servo_targets) + ((1.0 - SMOOTHING_ALPHA) * prev_angles)
        prev_angles = smoothed

        # final ints to send
        final_angles = [int(round(a)) for a in smoothed.tolist()]

        # send to Arduino (or print)
        send_angles_over_serial(final_angles)

        # HUD show values
        cv2.putText(frame, "Servos: " + ",".join(map(str, final_angles)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Invert: {current_invert}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Hand -> Servo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            # Toggle inversion
            current_invert = not current_invert
            print(f"[INFO] Inversion toggled to: {current_invert}")
        elif key == ord("c"):
            # calibration capture
            if calibrating_step == 0:
                # capture straight
                CALIB["straight"] = finger_angles_deg.copy()
                print("[CALIB] Straight captured:", CALIB["straight"])
                calibrating_step = 1
                print("Now curl your hand (close fingers) and press 'c' again to capture curled.")
            elif calibrating_step == 1:
                CALIB["curled"] = finger_angles_deg.copy()
                print("[CALIB] Curled captured:", CALIB["curled"])
                calibrating_step = 0
                print("[CALIB] Done. Using captured calibration values.")
                
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main()
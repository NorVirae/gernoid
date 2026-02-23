"""
main_controller.py — Gernoid Dual-Mode Controller

Main entry point. Switches between:
  Mode 1: Gesture control (MediaPipe hand tracking)
  Mode 2: AI control (Groq LLM object recognition)

Run:  python main_controller.py
"""

import sys
from servo_comms import ServoController
from gesture_mode import GestureMode
from ai_mode import AIMode
from config import GRIP_PRESETS


BANNER = """
╔══════════════════════════════════════╗
║         GERNOID CONTROL v2.0        ║
║   Dual-Mode Robotic Hand Control    ║
╠══════════════════════════════════════╣
║  [1] Gesture Mode (MediaPipe)       ║
║  [2] AI Mode (Groq + Camera)        ║
║  [q] Quit                           ║
╚══════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # Connect to Arduino
    servo = ServoController()
    if not servo.connect():
        print("[MAIN] Could not connect to Arduino. Running in dry-run mode.")
        print("[MAIN] Servo commands will be printed to console.\n")

    # Move to safe position
    servo.send_angles(GRIP_PRESETS["safe"])

    gesture = GestureMode()
    ai = AIMode()

    current_mode = None

    while True:
        print("\n── Select Mode ─────────────────────────")
        print("  [1] Gesture Mode  (hand tracking → servos)")
        print("  [2] AI Mode       (camera → LLM → grip)")
        print("  [q] Quit")
        print("─────────────────────────────────────────")

        choice = input(">> ").strip().lower()

        if choice in ("1", "gesture", "g"):
            current_mode = "gesture"
            print("\n[MAIN] Starting Gesture Mode...")
            print("[MAIN] Press 'm' inside the window to return here.\n")
            gesture = GestureMode()  # fresh instance
            gesture.start(servo)

        elif choice in ("2", "ai", "a"):
            current_mode = "ai"
            print("\n[MAIN] Starting AI Mode...")
            print("[MAIN] Press 'm' inside the window to return here.\n")
            ai = AIMode()  # fresh instance
            ai.start(servo)

        elif choice in ("q", "quit", "exit"):
            break

        else:
            print("[MAIN] Invalid choice. Type 1, 2, or q.")

    # Cleanup
    servo.send_angles(GRIP_PRESETS["open"])
    servo.close()
    print("\n[MAIN] Gernoid shut down. Goodbye!")


if __name__ == "__main__":
    main()

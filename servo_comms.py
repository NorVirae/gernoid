"""
servo_comms.py — Shared serial communication with the Arduino.

Both gesture mode and AI mode use this module to send servo angles
to the Arduino over USB serial.

Protocol: "thumb,index,middle,ring,pinky\n"  (CSV, 0-180 each)
"""

import time
import serial
from config import SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, NUM_SERVOS


class ServoController:
    """Manages the serial connection to the Arduino + PCA9685 servo driver."""

    def __init__(self, port=None, baud=None):
        self.port = port or SERIAL_PORT
        self.baud = baud or SERIAL_BAUD
        self.ser = None
        self.connected = False

    def connect(self):
        """Open the serial connection. Returns True on success."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=SERIAL_TIMEOUT)
            time.sleep(1.0)  # Wait for Arduino to reset after serial open
            self.connected = True
            print(f"[SERVO] Connected on {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"[SERVO] Connection failed on {self.port}: {e}")
            self.connected = False
            return False

    def send_angles(self, angles):
        """
        Send a list of 5 servo angles (0-180) to the Arduino.
        angles: list/tuple of 5 ints.
        Returns True on success.
        """
        if len(angles) != NUM_SERVOS:
            print(f"[SERVO] Expected {NUM_SERVOS} angles, got {len(angles)}")
            return False

        # Clamp all values to 0-180
        clamped = [max(0, min(180, int(a))) for a in angles]
        line = ",".join(str(a) for a in clamped) + "\n"

        if self.ser and self.connected:
            try:
                self.ser.write(line.encode("ascii"))
                return True
            except Exception as e:
                print(f"[SERVO] Write failed: {e}")
                self.connected = False
                return False
        else:
            # Fallback: print to console so you can see what would be sent
            print(f"[SERVO-DRY] {line.strip()}")
            return True

    def send_command(self, command):
        """
        Send a raw command string to the Arduino (e.g. 'TEST', 'INV,1,0,0,1,0').
        """
        if self.ser and self.connected:
            try:
                self.ser.write((command.strip() + "\n").encode("ascii"))
                return True
            except Exception as e:
                print(f"[SERVO] Command failed: {e}")
                return False
        else:
            print(f"[SERVO-DRY] CMD: {command.strip()}")
            return True

    def read_response(self, timeout=0.5):
        """Read a line of response from Arduino (if any)."""
        if self.ser and self.connected:
            try:
                self.ser.timeout = timeout
                line = self.ser.readline().decode("ascii", errors="replace").strip()
                return line if line else None
            except Exception:
                return None
        return None

    def close(self):
        """Close the serial connection."""
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.connected = False
        print("[SERVO] Connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

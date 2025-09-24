# debug_capture_fixed.py
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()  # OK to use like this; or use context manager

while True:
    success, frame = cap.read()
    if not success:
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)

    if result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # draw on the original BGR 'frame' so drawn colors show correctly
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # iterate each landmark (0..20) inside the hand
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                x_px = int(lm.x * frame.shape[1])
                y_px = int(lm.y * frame.shape[0])
                print(f"hand {hand_idx} lm {lm_idx}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.6f}")
                cv2.putText(frame, str(lm_idx), (x_px + 2, y_px - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)

    # show the BGR frame (with drawn landmarks and labels)
    cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

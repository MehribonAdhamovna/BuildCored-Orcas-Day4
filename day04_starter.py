import cv2
import mediapipe as mp
import time
import sys

# --- SETUP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap = cv2.VideoCapture(1)
if not cap.isOpened(): sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmarks for EAR
LEFT_EYE = [159, 145, 160, 144, 161, 153, 33, 133]
RIGHT_EYE = [386, 374, 387, 373, 388, 380, 362, 263]

def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id):
    v = sum([abs(landmarks[t].y - landmarks[b].y) for t, b in zip(top_ids, bottom_ids)]) / len(top_ids)
    h = abs(landmarks[left_id].x - landmarks[right_id].x)
    return v / h if h != 0 else 0.0

# --- CONFIG ---
EAR_THRESHOLD = 0.22
BLINK_WINDOW = 2.0  # Time to get 3 blinks
MIN_FRAMES = 3      # Minimum frames closed to count as a blink (Debounce)

STATE_IDLE = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED = "LOCKED"

# --- VARS ---
state = STATE_IDLE
blink_count = 0
start_time = 0
eye_closed_frames = 0
is_winking = False
wink_start = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    curr_ear = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        l_ear = get_ear(landmarks, [159, 160, 161], [145, 144, 153], 33, 133)
        r_ear = get_ear(landmarks, [386, 387, 388], [374, 373, 380], 362, 263)
        curr_ear = (l_ear + r_ear) / 2

        # Draw Eye Dots (Requirement: Visual EAR landmarks)
        for idx in LEFT_EYE + RIGHT_EYE:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 1, (0, 255, 0), -1)

        # 1. UNLOCK LOGIC (Wink)
        if state == STATE_LOCKED:
            if (l_ear < EAR_THRESHOLD and r_ear > 0.25) or (r_ear < EAR_THRESHOLD and l_ear > 0.25):
                if not is_winking:
                    is_winking, wink_start = True, time.time()
                if time.time() - wink_start > 1.5:
                    state, blink_count = STATE_IDLE, 0
            else: is_winking = False

        # 2. LOCK LOGIC (State Machine)
        else:
            if curr_ear < EAR_THRESHOLD:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= MIN_FRAMES:
                    if state == STATE_IDLE:
                        state, blink_count, start_time = STATE_COUNTING, 1, time.time()
                    elif state == STATE_COUNTING:
                        blink_count += 1
                        if blink_count >= 3: state = STATE_LOCKED
                eye_closed_frames = 0

            if state == STATE_COUNTING and (time.time() - start_time > BLINK_WINDOW):
                state, blink_count = STATE_IDLE, 0

    # --- UI RENDERING (Matches Checklist) ---
    if state == STATE_LOCKED:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, "LOCKED", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        if is_winking: cv2.putText(frame, "Unlocking...", (w//2-80, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
    # Requirements: EAR value, Threshold, and State displayed
    cv2.putText(frame, f"EAR: {curr_ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"State: {state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if state == STATE_COUNTING:
        # Requirement: Show progress and time limit
        rem = max(0, BLINK_WINDOW - (time.time() - start_time))
        cv2.putText(frame, f"Blinks: {blink_count}/3", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Window: {rem:.1f}s", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("BlinkLock Submission", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('u'): state, blink_count = STATE_IDLE, 0

cap.release()
cv2.destroyAllWindows()
print("\nBlinkLock ended. See you tomorrow for Day 05!")
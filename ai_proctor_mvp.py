# ai_proctor_mvp.py
import cv2
import mediapipe as mp
import time
import json
from datetime import datetime

# --- Configuration ---
FACE_ABSENCE_TIMEOUT = 8.0       # seconds before termination if no face
LOOKAWAY_THRESHOLD = 3           # counts before flagging
LOOKAWAY_MIN_ANGLE = 25          # degrees yaw or pitch considered "looking away"
DETECTION_FPS = 10               # how many frames per second logic checks (approx)
LOG_PATH = "proctor_log.json"

# --- Utilities ---
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def log_event(logs, event_type, detail):
    entry = {"ts": now_iso(), "type": event_type, "detail": detail}
    logs.append(entry)
    print(entry)

# --- Head pose estimation helpers using MediaPipe face landmarks ---
# Using landmarks to estimate approximate yaw (left/right) and pitch (up/down)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def estimate_head_angles(landmarks, image_w, image_h):
    # landmarks: list of normalized landmarks from mediapipe
    # This is a quick heuristic: compare nose tip to eye centers to estimate yaw/pitch.
    # For robust results, use solvePnP with 3D model points; this is a simplified heuristic.
    def to_xy(lm):
        return (lm.x * image_w, lm.y * image_h)

    # key landmark indices from mediapipe face mesh:
    # nose tip ~ 1, left eye inner ~ 33, right eye inner ~ 263 (approx)
    try:
        nose = to_xy(landmarks[1])
        left_eye = to_xy(landmarks[33])
        right_eye = to_xy(landmarks[263])
    except Exception:
        return 0.0, 0.0

    # yaw: difference between nose and midpoint of eyes horizontally
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    dx = nose[0] - eye_mid_x
    # normalized by distance between eyes to get degrees-like scale
    eye_dist = abs(left_eye[0] - right_eye[0]) + 1e-6
    yaw_score = (dx / eye_dist) * 50.0   # 50 is scale factor => few tens of degrees

    # pitch: vertical difference between nose and eye midpoint
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    dy = nose[1] - eye_mid_y
    pitch_score = (dy / eye_dist) * 50.0

    return yaw_score, pitch_score

# --- Main monitoring loop ---
def run_monitoring():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=2,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    logs = []
    last_face_seen_time = time.time()
    lookaway_counter = 0
    flagged_lookaway = False
    flagged_multiperson = False
    exam_terminated = False
    frame_count = 0
    fps_interval = 1.0 / DETECTION_FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            now = time.time()
            # Resize for speed if needed
            h, w = frame.shape[:2]
            # convert to RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use face detection to count faces quickly
            det_results = face_detection.process(rgb)
            faces = det_results.detections if det_results.detections else []
            num_faces = len(faces)

            # Face presence logic
            if num_faces == 0:
                # no face seen
                if (now - last_face_seen_time) >= FACE_ABSENCE_TIMEOUT:
                    # terminate exam
                    log_event(logs, "terminate_exam", {"reason": "face_absent", "timeout_s": FACE_ABSENCE_TIMEOUT})
                    exam_terminated = True
                    break
            else:
                last_face_seen_time = now

            # multiple person check
            if num_faces > 1:
                # display warning and log once per detection
                if not flagged_multiperson:
                    log_event(logs, "warning_multiperson", {"num_faces": num_faces})
                    log_event(logs, "flag_multiperson", {"num_faces": num_faces})
                    flagged_multiperson = True
                # overlay text
                cv2.putText(frame, "WARNING: Additional person detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                flagged_multiperson = False

            # For head direction, use face mesh landmarks on the primary face
            # We'll do head checks at reduced fps to save CPU
            if frame_count % max(1, int(30/DETECTION_FPS)) == 0:
                mesh_results = face_mesh.process(rgb)
                if mesh_results.multi_face_landmarks:
                    # use first face
                    landmarks = mesh_results.multi_face_landmarks[0].landmark
                    yaw, pitch = estimate_head_angles(landmarks, w, h)
                    # determine if looking away
                    if abs(yaw) > LOOKAWAY_MIN_ANGLE or abs(pitch) > LOOKAWAY_MIN_ANGLE:
                        # count a look-away occurrence
                        lookaway_counter += 1
                        log_event(logs, "warning_lookaway", {"count": lookaway_counter, "yaw": float(yaw), "pitch": float(pitch)})
                        # overlay warning
                        cv2.putText(frame, f"Please face the screen ({lookaway_counter}/{LOOKAWAY_THRESHOLD})", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        # flag if exceeded threshold
                        if lookaway_counter > LOOKAWAY_THRESHOLD and not flagged_lookaway:
                            log_event(logs, "flag_lookaway", {"count": lookaway_counter})
                            flagged_lookaway = True
                    else:
                        # no lookaway currently - optional: reduce counter over time? (not implemented)
                        pass
                else:
                    # no landmarks -> likely face absent (face presence logic handles termination)
                    pass

            # draw small webcam preview info on the frame (for your slide simulation)
            cv2.putText(frame, f"Faces: {num_faces}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # show frame
            cv2.imshow("AI Proctoring - Press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # end loop
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # write logs to file
    with open(LOG_PATH, "w") as f:
        json.dump({"events": logs, "terminated": exam_terminated, "final_lookaway_count": lookaway_counter}, f, indent=2)
    print(f"Log saved to {LOG_PATH}")
    return LOG_PATH

if __name__ == "__main__":
    run_monitoring()

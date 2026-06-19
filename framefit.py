import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.components.containers import landmark as landmark_module
import os

# ── Landmark index constants (same as old mp.solutions.pose) ──────────────
NOSE          = 0
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW    = 13; RIGHT_ELBOW    = 14
LEFT_WRIST    = 15; RIGHT_WRIST    = 16
LEFT_HIP      = 23; RIGHT_HIP      = 24
LEFT_KNEE     = 25; RIGHT_KNEE     = 26
LEFT_ANKLE    = 27; RIGHT_ANKLE    = 28

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")


# ── Drawing helpers ────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28)
]

def draw_pose(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], (0, 255, 255), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (255, 0, 255), -1)
    return image


# ── Angle calculator ───────────────────────────────────────────────────────
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return round(360 - angle if angle > 180 else angle, 2)

def lm(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]


# ── Exercise logic ─────────────────────────────────────────────────────────
def bicep_curl(landmarks, stage, counter):
    angle = calc_angle(lm(landmarks, LEFT_SHOULDER), lm(landmarks, LEFT_ELBOW), lm(landmarks, LEFT_WRIST))
    feedback, ftype = "", "warn"
    if angle > 160: stage = "down"
    if angle < 42 and stage == 'down':
        stage = "up"; counter += 1; feedback = "Great curl!"; ftype = "good"
    elif 40 <= angle <= 160:
        feedback = "Control your movement"; ftype = "warn"
    return angle, stage, counter, feedback, ftype

def pushups(landmarks, stage, counter):
    angle = calc_angle(lm(landmarks, LEFT_SHOULDER), lm(landmarks, LEFT_ELBOW), lm(landmarks, LEFT_WRIST))
    feedback, ftype = "", "warn"
    if angle > 160: stage = "up"
    if angle < 90 and stage == 'up':
        stage = "down"; counter += 1; feedback = "Nice push!"; ftype = "good"
    elif 90 <= angle <= 160:
        feedback = "Go a bit lower"; ftype = "warn"
    return angle, stage, counter, feedback, ftype

def shoulder_press(landmarks, stage, counter):
    left_angle  = calc_angle(lm(landmarks, LEFT_ELBOW),  lm(landmarks, LEFT_SHOULDER),  lm(landmarks, LEFT_HIP))
    right_angle = calc_angle(lm(landmarks, RIGHT_ELBOW), lm(landmarks, RIGHT_SHOULDER), lm(landmarks, RIGHT_HIP))
    avg_angle = (left_angle + right_angle) / 2
    feedback, ftype = "", "warn"
    if left_angle < 80 and right_angle < 80: stage = "down"
    if left_angle > 150 and right_angle > 150 and stage == 'down':
        stage = "up"; counter += 1; feedback = "Excellent!"; ftype = "good"
    elif 70 <= avg_angle <= 150: feedback = "Push both arms evenly!"; ftype = "warn"
    elif avg_angle < 70: feedback = "Lower arms fully first."; ftype = "warn"
    else: feedback = "Keep going!"; ftype = "warn"
    return avg_angle, stage, counter, feedback, ftype

def squats(landmarks, stage, counter):
    angle = calc_angle(lm(landmarks, LEFT_HIP), lm(landmarks, LEFT_KNEE), lm(landmarks, LEFT_ANKLE))
    feedback, ftype = "", "warn"
    if angle > 160: stage = "up"
    if angle < 90 and stage == 'up':
        stage = "down"; counter += 1; feedback = "Perfect squat!"; ftype = "good"
    elif 90 <= angle <= 160:
        feedback = "Go lower!"; ftype = "warn"
    return angle, stage, counter, feedback, ftype

def lateral_raises(landmarks, stage, counter):
    angle = calc_angle(lm(landmarks, LEFT_HIP), lm(landmarks, LEFT_SHOULDER), lm(landmarks, LEFT_ELBOW))
    feedback, ftype = "", "warn"
    if angle < 40: stage = "down"
    if angle > 80 and stage == 'down':
        stage = "up"; counter += 1; feedback = "Nice raise!"; ftype = "good"
    elif 40 <= angle <= 80:
        feedback = "Lift arms higher"; ftype = "warn"
    return angle, stage, counter, feedback, ftype


# ── HUD overlay ────────────────────────────────────────────────────────────
# BGR colour palette for feedback types
FEEDBACK_COLORS = {
    "good":  (0, 220, 100),   # green  — rep completed / great form
    "warn":  (0, 200, 255),   # amber  — guidance / correction needed
    "error": (60,  60, 235),  # red    — no pose / missing landmarks
}

def draw_info_panel(image, counter, stage, feedback, exercise, ftype="warn"):
    overlay = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(overlay, (10, 10), (w - 10, 160), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    cv2.putText(image, f"{exercise.upper()} TRACKER", (25, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Reps: {counter}", (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Stage: {stage.upper() if stage else '-'}", (250, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 0), 2, cv2.LINE_AA)
    color = FEEDBACK_COLORS.get(ftype, FEEDBACK_COLORS["warn"])
    cv2.putText(image, feedback, (25, 135),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
    return image


# ── Main runner ────────────────────────────────────────────────────────────
def run_framefit(exercise_choice):
    analyzers = {
        'bicep curl':    bicep_curl,
        'pushup':        pushups,
        'shoulder press':shoulder_press,
        'squat':         squats,
        'lateral raise': lateral_raises,
    }
    analyzer = analyzers[exercise_choice.lower()]

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Could not open camera."); return

    counter, stage, feedback = 0, None, ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                try:
                    angle, stage, counter, feedback, ftype = analyzer(landmarks, stage, counter)
                    cv2.putText(frame, f"{int(angle)}", (50, 420),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                except Exception as e:
                    feedback = "Ensure full body is visible."
                    ftype = "error"

                draw_pose(frame, landmarks)
            else:
                feedback = "No pose detected — step back a bit."
                ftype = "error"

            frame = draw_info_panel(frame, counter, stage, feedback, exercise_choice, ftype)
            cv2.imshow(f'FrameFit | {exercise_choice.title()}', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════╗")
    print("║   FrameFit Exercise Tracker   ║")
    print("╚══════════════════════════════╝")
    print("\nChoose an exercise:")
    print("  1. Bicep Curl")
    print("  2. Pushup")
    print("  3. Shoulder Press")
    print("  4. Squat")
    print("  5. Lateral Raise\n")

    choice = input("Enter choice (1-5): ").strip()
    exercises = {
        "1": "bicep curl",
        "2": "pushup",
        "3": "shoulder press",
        "4": "squat",
        "5": "lateral raise",
    }

    if choice in exercises:
        print(f"\n▶ Starting {exercises[choice].title()} tracker... Press Q to quit.\n")
        run_framefit(exercises[choice])
    else:
        print("Invalid choice. Please restart and enter 1–5.")

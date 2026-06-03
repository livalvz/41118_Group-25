import csv
import os
import cv2
import mediapipe as mp
import numpy as np
 
DATA_PATH = "gesture_landmarks.csv"
 
# Only collect static hand shapes.
# Dynamic gestures such as swipe_left, swipe_right, and swipe_up
# are detected in gesture_recognition_node.py by tracking wrist movement.
GESTURE_KEYS = {
    ord("1"): "one_finger",
    ord("2"): "two_fingers",
    ord("3"): "three_fingers",
    ord("o"): "open_palm",
    ord("f"): "closed_fist",
}
 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
 
def normalise_landmarks(hand_landmarks):
    """
    Converts 21 MediaPipe landmarks into a normalised 63-value feature vector.
    The wrist is used as the origin, then all values are scaled by hand size.
    """
 
    points = []
 
    for landmark in hand_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])
 
    points = np.array(points)
 
    wrist = points[0]
    points = points - wrist
 
    scale = np.linalg.norm(points[9])
 
    if scale == 0:
        scale = 1
 
    points = points / scale
 
    return points.flatten().tolist()
 
 
def create_csv_if_needed():
    if not os.path.exists(DATA_PATH):
        header = ["label"]
 
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
 
        with open(DATA_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
 
 
def save_sample(label, features):
    with open(DATA_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([label] + features)
 
 
def main():
    create_csv_if_needed()
 
    cap = cv2.VideoCapture(0)
 
    sample_counts = {label: 0 for label in GESTURE_KEYS.values()}
 
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
 
        while True:
            ret, frame = cap.read()
 
            if not ret:
                print("Could not read from camera")
                break
 
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
            results = hands.process(rgb)
 
            current_features = None
 
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
 
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
 
                current_features = normalise_landmarks(hand_landmarks)
 
            cv2.putText(
                frame,
                "Press: 1=blue, 2=red, 3=direction mode, o=stop, f=home, q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
 
            cv2.putText(
                frame,
                "Swipes are NOT collected here. They are detected live by wrist movement.",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )
 
            y = 85
            for label, count in sample_counts.items():
                cv2.putText(
                    frame,
                    f"{label}: {count}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )
                y += 25
 
            cv2.imshow("Collect Static Gesture Landmarks", frame)
 
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("q"):
                break
 
            if key in GESTURE_KEYS:
                if current_features is not None:
                    label = GESTURE_KEYS[key]
                    save_sample(label, current_features)
                    sample_counts[label] += 1
                    print(f"Saved sample for: {label}")
                else:
                    print("No hand detected, sample not saved")
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
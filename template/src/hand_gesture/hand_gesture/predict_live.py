# predict_live.py
import cv2
import joblib
import mediapipe as mp
import numpy as np

from collections import deque, Counter


MODEL_PATH = "models/gesture_classifier.pkl"

GESTURE_TO_COMMAND = {
    "open_palm": "STOP",
    "fist": "PAUSE",
    "thumbs_up": "START",
    "pointing": "SELECT",
    "peace": "RESET",
}

CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 8


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalise_landmarks(hand_landmarks):
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

    return points.flatten().reshape(1, -1)


def get_smoothed_prediction(prediction_history):
    if not prediction_history:
        return "no_gesture"

    counts = Counter(prediction_history)
    return counts.most_common(1)[0][0]


def main():
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]

    prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)

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

            gesture = "no_gesture"
            command = "NO_COMMAND"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                features = normalise_landmarks(hand_landmarks)

                probabilities = model.predict_proba(features)[0]
                best_index = np.argmax(probabilities)

                raw_gesture = model.classes_[best_index]
                confidence = probabilities[best_index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(raw_gesture)
                    gesture = get_smoothed_prediction(prediction_history)
                    command = GESTURE_TO_COMMAND.get(gesture, "NO_COMMAND")
                else:
                    prediction_history.append("uncertain")
                    gesture = "uncertain"
                    command = "NO_COMMAND"

            else:
                prediction_history.clear()

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Command: {command}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press q to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Live Gesture Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
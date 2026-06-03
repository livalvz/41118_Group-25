#!/usr/bin/env python3
 
# Gesture recognition node
# Static gestures are recognised using MediaPipe landmarks + Random Forest.
# Direction commands are controlled by holding a fist and moving the hand
# into screen zones:
#   fist + left side  -> TURN_LEFT
#   fist + right side -> TURN_RIGHT
#   fist + top side   -> GO_FORWARD
 
from collections import Counter, deque
import os
 
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
 
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
 
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
 
 
MODEL_PATH = "models/gesture_classifier.pkl"
 
# Static gestures from the trained Random Forest classifier
GESTURE_TO_COMMAND = {
    "one_finger": "GO_TO_BLUE_BLOCK",
    "two_fingers": "GO_TO_RED_BLOCK",
    "open_palm": "STOP",
    "closed_fist": "RETURN_HOME",
    "fist": "RETURN_HOME",
    "three_fingers": "STOP",
}
 
THREE_FINGERS_ZONE_TO_COMMAND = {
    "three_fingers_left_zone": "TURN_LEFT",
    "three_fingers_right_zone": "TURN_RIGHT",
    "three_fingers_top_zone": "GO_FORWARD",
    "three_fingers_centre_zone": "STOP",
}
 
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 8
 
# Screen zone boundaries.
# MediaPipe x and y are normalised from 0.0 to 1.0.
# x = 0 is left side of image, x = 1 is right side.
# y = 0 is top of image, y = 1 is bottom.
LEFT_ZONE_X = 0.35
RIGHT_ZONE_X = 0.65
TOP_ZONE_Y = 0.35
 
 
class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__("gesture_recognition_node")
 
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.declare_parameter("smoothing_window", SMOOTHING_WINDOW)
        self.declare_parameter("publish_rate_hz", 15.0)
 
        self.declare_parameter("show_window", True)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("camera_fps", 15)
 
        self.camera_index = self.get_parameter("camera_index").value
        self.model_path = self.get_parameter("model_path").value
 
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
 
        smoothing_window = int(self.get_parameter("smoothing_window").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
 
        self.show_window = bool(self.get_parameter("show_window").value)
        self.frame_width = int(self.get_parameter("frame_width").value)
        self.frame_height = int(self.get_parameter("frame_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
 
        self.bridge = CvBridge()
 
        # History for smoothing static classifier predictions
        self.prediction_history = deque(maxlen=smoothing_window)
 
        # Debug values for overlay
        self.hand_centre_x = 0.0
        self.hand_centre_y = 0.0
        self.zone_name = "no_zone"
 
        self.gesture_pub = self.create_publisher(
            String,
            "/gesture/name",
            10
        )
 
        self.confidence_pub = self.create_publisher(
            Float32,
            "/gesture/confidence",
            10
        )
 
        self.command_pub = self.create_publisher(
            String,
            "/gesture/command",
            10
        )
 
        self.debug_image_pub = self.create_publisher(
            Image,
            "/gesture/debug_image",
            qos_profile_sensor_data
        )
 
        self.model, self.feature_names = self.load_model(self.model_path)
 
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
 
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
 
        self.cap = cv2.VideoCapture(self.camera_index)
 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
 
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Could not open camera index {self.camera_index}"
            )
            raise RuntimeError("Could not open webcam")
 
        timer_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(timer_period, self.process_frame)
 
        self.get_logger().info("Gesture recognition node started")
        self.get_logger().info(f"Using model: {self.model_path}")
        self.get_logger().info(
            f"Camera: {self.frame_width}x{self.frame_height} at {self.camera_fps} FPS"
        )
        self.get_logger().info(f"OpenCV preview window enabled: {self.show_window}")
        self.get_logger().info("Static gestures: one_finger, two_fingers, open_palm, closed_fist")
        self.get_logger().info("Fist zones: left=TURN_LEFT, right=TURN_RIGHT, top=GO_FORWARD")
 
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not find model at {model_path}. "
                "Run train_classifier.py first."
            )
 
        model_data = joblib.load(model_path)
 
        if isinstance(model_data, dict):
            model = model_data["model"]
            feature_names = model_data.get("feature_names", None)
        else:
            model = model_data
            feature_names = None
 
        return model, feature_names
 
    def normalise_landmarks(self, hand_landmarks):
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
        features = points.flatten().reshape(1, -1)
 
        if self.feature_names is not None:
            features = pd.DataFrame(features, columns=self.feature_names)
 
        return features
 
    def get_smoothed_prediction(self):
        if not self.prediction_history:
            return "no_gesture"
 
        counts = Counter(self.prediction_history)
        return counts.most_common(1)[0][0]
 
    def get_hand_centre(self, hand_landmarks):
        """
        Calculates the centre of the whole hand using all 21 MediaPipe landmarks.
        Returns normalised x and y values between 0.0 and 1.0.
        """
 
        xs = []
        ys = []
 
        for landmark in hand_landmarks.landmark:
            xs.append(landmark.x)
            ys.append(landmark.y)
 
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
 
        centre_x = (min_x + max_x) / 2.0
        centre_y = (min_y + max_y) / 2.0
 
        return centre_x, centre_y
 
    def get_three_fingers_zone_command(self, hand_landmarks):
        """
        If the detected static gesture is three_fingers, this function checks where
        the hand is on the screen.
 
        Priority:
        1. Top zone    -> GO_FORWARD
        2. Left zone   -> TURN_LEFT
        3. Right zone  -> TURN_RIGHT
        4. Centre zone -> STOP
        """
 
        centre_x, centre_y = self.get_hand_centre(hand_landmarks)
 
        self.hand_centre_x = centre_x
        self.hand_centre_y = centre_y
 
        if centre_y < TOP_ZONE_Y:
            self.zone_name = "three_fingers_top_zone"
            return (
                "three_fingers_top_zone",
                THREE_FINGERS_ZONE_TO_COMMAND["three_fingers_top_zone"]
            )
 
        if centre_x < LEFT_ZONE_X:
            self.zone_name = "three_fingers_left_zone"
            return (
                "three_fingers_left_zone",
                THREE_FINGERS_ZONE_TO_COMMAND["three_fingers_left_zone"]
            )
 
        if centre_x > RIGHT_ZONE_X:
            self.zone_name = "three_fingers_right_zone"
            return (
                "three_fingers_right_zone",
                THREE_FINGERS_ZONE_TO_COMMAND["three_fingers_right_zone"]
            )
 
        self.zone_name = "three_fingers_centre_zone"
        return (
            "three_fingers_centre_zone",
            THREE_FINGERS_ZONE_TO_COMMAND["three_fingers_centre_zone"]
        )
 
    def draw_zone_guides(self, frame):
        """
        Draws visual guide lines showing the left, right, and top control zones.
        """
 
        height, width = frame.shape[:2]
 
        left_x = int(width * LEFT_ZONE_X)
        right_x = int(width * RIGHT_ZONE_X)
        top_y = int(height * TOP_ZONE_Y)
 
        # Vertical left boundary
        cv2.line(
            frame,
            (left_x, 0),
            (left_x, height),
            (255, 255, 255),
            1
        )
 
        # Vertical right boundary
        cv2.line(
            frame,
            (right_x, 0),
            (right_x, height),
            (255, 255, 255),
            1
        )
 
        # Horizontal top boundary
        cv2.line(
            frame,
            (0, top_y),
            (width, top_y),
            (255, 255, 255),
            1
        )
 
        cv2.putText(
            frame,
            "3 FINGERS + LEFT = TURN_LEFT",
            (10, top_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
 
        cv2.putText(
            frame,
            "3 FINGERS + TOP = GO_FORWARD",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
 
        cv2.putText(
            frame,
            "3 FINGERS + RIGHT = TURN_RIGHT",
            (right_x + 10, top_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )
 
        return frame
 
    def publish_text_outputs(self, gesture, confidence, command):
        gesture_msg = String()
        gesture_msg.data = gesture
 
        confidence_msg = Float32()
        confidence_msg.data = float(confidence)
 
        command_msg = String()
        command_msg.data = command
 
        self.gesture_pub.publish(gesture_msg)
        self.confidence_pub.publish(confidence_msg)
        self.command_pub.publish(command_msg)
 
    def draw_overlay(self, frame, gesture, confidence, command):
        frame = self.draw_zone_guides(frame)
 
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )
 
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
 
        cv2.putText(
            frame,
            f"Command: {command}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
 
        cv2.putText(
            frame,
            f"Zone: {self.zone_name}",
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
 
        cv2.putText(
            frame,
            f"Hand centre: x={self.hand_centre_x:.2f}, y={self.hand_centre_y:.2f}",
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
 
        cv2.putText(
            frame,
            "Press q in this window to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
 
        return frame
 
    def process_frame(self):
        ret, frame = self.cap.read()
 
        if not ret:
            self.get_logger().warn("Could not read frame from webcam")
            return
 
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        results = self.hands.process(rgb)
 
        gesture = "no_gesture"
        command = "NO_COMMAND"
        confidence = 0.0
        self.zone_name = "no_zone"
        self.hand_centre_x = 0.0
        self.hand_centre_y = 0.0
 
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
 
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
 
            features = self.normalise_landmarks(hand_landmarks)
 
            probabilities = self.model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
 
            raw_gesture = self.model.classes_[best_index]
            confidence = float(probabilities[best_index])
 
            if confidence >= self.confidence_threshold:
                self.prediction_history.append(raw_gesture)
                gesture = self.get_smoothed_prediction()
                command = GESTURE_TO_COMMAND.get(gesture, "NO_COMMAND")
            else:
                self.prediction_history.append("uncertain")
                gesture = "uncertain"
                command = "NO_COMMAND"
 
            # Three fingers + screen zone override.
            # This replaces unreliable swipe detection.
            if gesture == "three_fingers":
                zone_gesture, zone_command = self.get_three_fingers_zone_command(
                    hand_landmarks
                )
                gesture = zone_gesture
                command = zone_command
 
        else:
            self.prediction_history.clear()
 
        frame = self.draw_overlay(
            frame,
            gesture,
            confidence,
            command
        )
 
        if self.show_window:
            cv2.imshow("Gesture Recognition Debug View", frame)
 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Quit requested from OpenCV window")
                rclpy.shutdown()
                return
 
        self.publish_text_outputs(gesture, confidence, command)
 
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "gesture_camera"
 
        self.debug_image_pub.publish(image_msg)
 
    def destroy_node(self):
        if hasattr(self, "cap"):
            self.cap.release()
 
        if hasattr(self, "hands"):
            self.hands.close()
 
        cv2.destroyAllWindows()
 
        super().destroy_node()
 
 
def main(args=None):
    rclpy.init(args=args)
 
    node = GestureRecognitionNode()
 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()
 
 
if __name__ == "__main__":
    main()
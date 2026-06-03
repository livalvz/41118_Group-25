import os
from ament_index_python.packages import get_package_share_directory
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import threading
import numpy as np
import base64
import re
 
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
 
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
 
 
# Flask
package_share = get_package_share_directory('gui')
template_dir = os.path.join(package_share, 'templates')
static_dir = os.path.join(package_share, 'static')
 
app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)
 
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)
 
 
# State
latest_status = "Waiting for ROS2..."
latest_command = "NONE"
latest_confidence = 0.0
 
latest_gesture_frame = np.zeros((720, 960, 3), dtype=np.uint8)
latest_car_frame = np.zeros((720, 960, 3), dtype=np.uint8)
 
bridge = CvBridge()
 
 
# Image encoder
def encode_image(frame):
    success, buffer = cv2.imencode('.jpg', frame)
 
    if not success:
        return ""
 
    return base64.b64encode(buffer).decode('utf-8')
 
 
# ros node
class GestureBridge(Node):
    def __init__(self):
        super().__init__('gesture_bridge')
 
        # Gesture text topics
        self.create_subscription(
            String,
            '/gesture/name',
            self.name_cb,
            10
        )
 
        self.create_subscription(
            Float32,
            '/gesture/confidence',
            self.conf_cb,
            10
        )
 
        self.create_subscription(
            String,
            '/gesture/command',
            self.cmd_cb,
            10
        )
 
        # Gesture image topic for bottom-right GUI view
        self.create_subscription(
            Image,
            '/gesture/debug_image',
            self.img_cb,
            qos_profile_sensor_data
        )
 
        # Driving status topics
        self.create_subscription(
            String,
            '/driving/status',
            self.driving_status_cb,
            10
        )
 
        self.create_subscription(
            Float32,
            '/driving/reward',
            self.reward_cb,
            10
        )
 
        # Driving image topic for big left GUI view
        self.create_subscription(
            Image,
            '/driving/debug_image',
            self.car_img_cb,
            10
        )
 
        print("ROS2 GUI Bridge Node Started")
 
        socketio.emit("status_update", {
            "status": "ROS2 NODE CONNECTED"
        })
 
    # gesture callbacks
    def name_cb(self, msg):
        global latest_status
        latest_status = msg.data
 
        socketio.emit("gesture_update", {
            "gesture": latest_status,
            "command": latest_command
        })
 
    def cmd_cb(self, msg):
        global latest_command
        latest_command = msg.data
 
        print("CMD RECEIVED:", msg.data)
 
        socketio.emit("gesture_update", {
            "gesture": latest_status,
            "command": latest_command
        })
 
    def conf_cb(self, msg):
        global latest_confidence
        latest_confidence = float(msg.data)
 
        socketio.emit("confidence_update", {
            "confidence": round(latest_confidence, 2)
        })
 
    def img_cb(self, msg):
        global latest_gesture_frame
 
        try:
            frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            latest_gesture_frame = frame
 
            socketio.emit("image_update", {
                "image": encode_image(frame)
            })
 
        except Exception as e:
            print("Gesture image error:", e)
 
    # driving callbacks
    def driving_status_cb(self, msg):
        socketio.emit("driving_status_update", {
            "status": msg.data
        })
 
        socketio.emit("status_update", {
            "status": msg.data
        })
 
    def reward_cb(self, msg):
        socketio.emit("reward_update", {
            "reward": round(float(msg.data), 3)
        })
 
    def car_img_cb(self, msg):
        global latest_car_frame
 
        try:
            frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            latest_car_frame = frame
 
            socketio.emit("car_image_update", {
                "image": encode_image(frame)
            })
 
        except Exception as e:
            print("Car image error:", e)
 
 
# frontend
@socketio.on("upload_image")
def handle_upload(data):
    global latest_car_frame
 
    try:
        img_data = re.sub('^data:image/.+;base64,', '', data["image"])
        img_bytes = base64.b64decode(img_data)
 
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
        latest_car_frame = frame
 
        socketio.emit("car_image_update", {
            "image": encode_image(frame)
        })
 
        print("UPLOAD RECEIVED")
 
    except Exception as e:
        print("Upload error:", e)
 
 
# routes
@app.route("/")
def index():
    return render_template("index.html")
 
 
# ros thread
def ros_thread():
    print("Starting ROS2 thread...")
 
    rclpy.init()
    node = GestureBridge()
 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
 
    node.destroy_node()
    rclpy.shutdown()
 
 
# main
def main(args=None):
    threading.Thread(target=ros_thread, daemon=True).start()
 
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )
 
 
if __name__ == "__main__":
    main()
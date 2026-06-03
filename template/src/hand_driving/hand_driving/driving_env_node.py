#!/usr/bin/env python3

import math
import numpy as np
from collections import deque
import cv2

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from simple_driving.envs.simple_driving_env import SimpleDrivingEnv


def angle_wrap(angle):
    """
    Keeps an angle between -pi and pi.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def student_reward_callback(
    car_pos,
    goal_pos,
    obstacle_pos,
    has_obstacle,
    prev_dist_to_goal,
    dist_to_goal,
    reached_goal
):
    """
    Simple reward function for the environment.

    Positive reward if the car gets closer to the goal.
    Big reward if it reaches the goal.
    Negative reward if it gets too close to obstacle.
    """

    reward = 0.0

    if prev_dist_to_goal is not None:
        reward += prev_dist_to_goal - dist_to_goal

    if reached_goal:
        reward += 50.0

    if has_obstacle and obstacle_pos is not None:
        dist_to_obs = math.sqrt(
            (car_pos[0] - obstacle_pos[0]) ** 2 +
            (car_pos[1] - obstacle_pos[1]) ** 2
        )

        if dist_to_obs < 1.0:
            reward -= 50.0

    return reward


def student_observation_callback(
    client,
    car_pos,
    car_orn,
    goal_pos,
    goal_orn,
    obstacle_pos,
    has_obstacle
):
    """
    Observation returned to the environment.

    Your environment expects 5 values because its observation space is:
    [car_x, car_y, goal_x, goal_y, obstacle_flag]
    """

    obstacle_flag = 1.0 if has_obstacle else 0.0

    return [
        car_pos[0],
        car_pos[1],
        goal_pos[0],
        goal_pos[1],
        obstacle_flag
    ]


class DrivingEnvNode(Node):
    def __init__(self):
        super().__init__("driving_env_node")

        self.declare_parameter("scenario", "distractors")
        self.declare_parameter("renders", True)

        self.scenario = self.get_parameter("scenario").value
        self.renders = self.get_parameter("renders").value

        self.target_colour = None
        self.stopped = False
        self.manual_action = None
        self.target_queue = deque()

        self.bridge = CvBridge()

        self.get_logger().info("Creating SimpleDrivingEnv...")

        self.env = SimpleDrivingEnv(
            isDiscrete=True,
            renders=self.renders,
            reward_callback=student_reward_callback,
            observation_callback=student_observation_callback
        )

        self.gesture_command_sub = self.create_subscription(
            String,
            "/gesture/command",
            self.gesture_command_callback,
            10
        )

        self.reset_environment(default_target="blue")

        self.target_sub = self.create_subscription(
            String,
            "/driving/target_colour",
            self.target_callback,
            10
        )

        self.stop_sub = self.create_subscription(
            Bool,
            "/driving/stop",
            self.stop_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            "/driving/status",
            10
        )

        self.reward_pub = self.create_publisher(
            Float32,
            "/driving/reward",
            10
        )

        self.done_pub = self.create_publisher(
            Bool,
            "/driving/done",
            10
        )

        # NEW: publishes the PyBullet/environment view for the GUI main box
        self.debug_image_pub = self.create_publisher(
            Image,
            "/driving/debug_image",
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Driving environment node started.")
        self.publish_status("ready_waiting_for_gesture")

    def reset_environment(self, default_target="blue"):
        if self.target_colour in ["red", "blue"]:
            target = self.target_colour
        else:
            target = default_target

        self.get_logger().info(
            f"Resetting environment: scenario={self.scenario}, target={target}"
        )

        self.obs, self.info = self.env.reset(
            options={
                "scenario": self.scenario,
                "target_colour": target
            }
        )

        # Publish one image straight after reset so the GUI does not stay black
        self.publish_debug_image()

    def set_target_without_reset(self, colour):
        """
        Changes the goal target without resetting the whole PyBullet environment.
        This keeps the car and objects where they currently are.
        """

        colour = colour.lower().strip()

        if not hasattr(self.env, "block_positions"):
            self.get_logger().warn(
                "No block_positions found. Cannot change target without reset."
            )
            return

        if colour not in self.env.block_positions:
            self.get_logger().warn(
                f"Target colour '{colour}' does not exist in environment."
            )
            return

        self.target_colour = colour

        self.env.goal = self.env.block_positions[colour]

        colour_to_index = {
            "red": 0,
            "blue": 1,
            "green": 2,
            "yellow": 3,
        }

        if hasattr(self.env, "blocks") and colour in colour_to_index:
            index = colour_to_index[colour]

            if index < len(self.env.blocks):
                self.env.goal_object.goal = self.env.blocks[index]
            else:
                self.get_logger().warn(f"No block object found for {colour}")
                return

        self.env.done = False
        self.env.reached_goal = False
        self.env._envStepCounter = 0
        self._goal_reached_logged = False
        self.stopped = False
        self.manual_action = None

        car_pos, _ = self.env._p.getBasePositionAndOrientation(self.env.car.car)

        self.env.prev_dist_to_goal = math.sqrt(
            (car_pos[0] - self.env.goal[0]) ** 2 +
            (car_pos[1] - self.env.goal[1]) ** 2
        )

        self.publish_status(f"moving_to_{colour}")

    def target_callback(self, msg):
        new_target = msg.data.lower().strip()

        if new_target not in ["red", "blue"]:
            self.get_logger().warn(
                f"Ignoring invalid target colour: {new_target}"
            )
            return

        if new_target != self.target_colour:
            self.set_target_without_reset(new_target)

    def stop_callback(self, msg):
        self.stopped = msg.data

        if self.stopped:
            self.publish_status("stopped")
        else:
            self.publish_status(f"moving_to_{self.target_colour}")

    def choose_action(self):
        """
        Driving controller with obstacle avoidance using potential fields.
        Goal attracts the car, nearby blocks repel it.
        """

        car_pos, car_orn = self.env._p.getBasePositionAndOrientation(
            self.env.car.car
        )

        car_x = car_pos[0]
        car_y = car_pos[1]

        goal_pos = self.env.goal
        goal_x = goal_pos[0]
        goal_y = goal_pos[1]

        # --- Attractive force toward goal ---
        attract_x = goal_x - car_x
        attract_y = goal_y - car_y

        # Normalise so distance doesn't dominate
        attract_dist = math.sqrt(attract_x**2 + attract_y**2) + 1e-6
        attract_x /= attract_dist
        attract_y /= attract_dist

        # --- Repulsive force away from non-target blocks ---
        REPULSE_RADIUS = 2.5   # how close before repulsion kicks in
        REPULSE_STRENGTH = 2.0 # how hard it pushes away

        repulse_x = 0.0
        repulse_y = 0.0

        if hasattr(self.env, "block_positions"):
            for colour, pos in self.env.block_positions.items():
                # Skip the current target block — we want to go TO that one
                if colour == self.target_colour:
                    continue

                bx = pos[0] - car_x
                by = pos[1] - car_y
                dist = math.sqrt(bx**2 + by**2) + 1e-6

                if dist < REPULSE_RADIUS:
                    # Repulsion grows as dist shrinks
                    strength = REPULSE_STRENGTH * (REPULSE_RADIUS - dist) / REPULSE_RADIUS
                    repulse_x -= (bx / dist) * strength
                    repulse_y -= (by / dist) * strength

        # Also repel from obstacle if one exists
        if self.env.has_obstacle and self.env.obstacle_pos is not None:
            ox = self.env.obstacle_pos[0] - car_x
            oy = self.env.obstacle_pos[1] - car_y
            dist = math.sqrt(ox**2 + oy**2) + 1e-6
            if dist < REPULSE_RADIUS:
                strength = REPULSE_STRENGTH * (REPULSE_RADIUS - dist) / REPULSE_RADIUS
                repulse_x -= (ox / dist) * strength
                repulse_y -= (oy / dist) * strength

        # --- Combined steering direction ---
        steer_x = attract_x + repulse_x
        steer_y = attract_y + repulse_y

        target_yaw = math.atan2(steer_y, steer_x)

        _, _, car_yaw = self.env._p.getEulerFromQuaternion(car_orn)

        heading_error = angle_wrap(target_yaw - car_yaw)

        if heading_error > 0.25:
            return 8  # forward right
        elif heading_error < -0.25:
            return 6  # forward left
        else:
            return 7  # forward straight

    def publish_debug_image(self):
        """
        Publishes a camera view of the PyBullet driving environment to:
        /driving/debug_image

        This does not rely on env.render(), because the PyBullet GUI can open
        without returning an image frame to Python.
        """

        try:
            width = 960
            height = 720

            # Follow the car so it stays in frame
            try:
                car_pos, _ = self.env._p.getBasePositionAndOrientation(self.env.car.car)
                cam_target = [car_pos[0], car_pos[1], 0.0]
            except Exception:
                cam_target = [0.0, 0.0, 0.0]

            view_matrix = self.env._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_target,
                distance=12.0,
                yaw=50.0,
                pitch=-45.0,
                roll=0.0,
                upAxisIndex=2
            )

            projection_matrix = self.env._p.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=float(width) / float(height),
                nearVal=0.1,
                farVal=100.0
            )

            _, _, rgba, _, _ = self.env._p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )

            frame = np.array(rgba, dtype=np.uint8)
            frame = frame.reshape((height, width, 4))

            # RGBA to BGR for ROS/OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "driving_camera"

            self.debug_image_pub.publish(image_msg)

        except Exception as e:
            self.get_logger().warn(f"Could not publish driving debug image: {e}")

    def control_loop(self):
        if self.stopped:
            self.publish_debug_image()
            return

        if self.manual_action is not None:
            # Manual gestures always run regardless of done state
            action = self.manual_action
        else:
            # Autonomous mode: pop next target when done, idle if queue empty
            if self.env.done:
                self.publish_status("goal_reached")
                self.env.done = False
                self.env.reached_goal = False
                self.env._envStepCounter = 0
                self._goal_reached_logged = False
                self.target_colour = None
                self._pop_next_target()
                self.publish_debug_image()
                return
            if self.target_colour is None:
                self.publish_debug_image()
                return
            action = self.choose_action()

        obs, reward, done, truncated, info = self.env.step(action)

        self.publish_debug_image()

        self.obs = obs

        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)

        done_msg = Bool()
        done_msg.data = bool(done)
        self.done_pub.publish(done_msg)

        if done and not self.env.reached_goal:
            self.publish_status("episode_done")

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _pop_next_target(self):
        """Pop next target from queue or idle if empty."""
        if self.target_queue:
            next_colour = self.target_queue.popleft()
            self.get_logger().info(f"Next target: {next_colour} (remaining={list(self.target_queue)})")
            self.set_target_without_reset(next_colour)
        else:
            self.target_colour = None
            self.stopped = False
            self.get_logger().info("Queue empty - waiting for next command")

    def _return_to_start(self):
        """
        Teleport the car back to spawn position [0, 0, 0.1] with zero velocity.
        Clears done/reached flags so control_loop keeps running.
        """
        HOME_POS = [0.0, 0.0, 0.1]
        HOME_ORN = self.env._p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.env._p.resetBasePositionAndOrientation(
            self.env.car.car, HOME_POS, HOME_ORN
        )
        self.env._p.resetBaseVelocity(
            self.env.car.car,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0]
        )

        self.env.done = False
        self.env.reached_goal = False
        self.env._envStepCounter = 0
        self._goal_reached_logged = False
        self.get_logger().info(f"Car teleported to home {HOME_POS}")

    def gesture_command_callback(self, msg):
        command = msg.data.strip().upper()

        if command in ["GO_TO_BLUE_BLOCK", "GO_TO_BLUE"]:
            self.target_queue.append("blue")
            self.manual_action = None
            self.stopped = False
            self.get_logger().info(f"Queued: blue (queue={list(self.target_queue)})")
            if self.target_colour is None:
                self._pop_next_target()

        elif command in ["GO_TO_RED_BLOCK", "GO_TO_RED"]:
            self.target_queue.append("red")
            self.manual_action = None
            self.stopped = False
            self.get_logger().info(f"Queued: red (queue={list(self.target_queue)})")
            if self.target_colour is None:
                self._pop_next_target()

        elif command == "STOP":
            self.manual_action = None
            self.stopped = True
            self.publish_status("stopped")

        elif command in ["RETURN_HOME", "HOME"]:
            self.manual_action = None
            self.target_colour = None
            self.target_queue.clear()
            self.stopped = False
            self._return_to_start()
            self.publish_status("returning_home")

        elif command == "GO_FORWARD":
            self.manual_action = 7
            self.stopped = False
            self.env.done = False
            self.env._envStepCounter = 0
            self.publish_status("manual_forward")

        elif command == "TURN_LEFT":
            self.manual_action = 6
            self.stopped = False
            self.env.done = False
            self.env._envStepCounter = 0
            self.publish_status("manual_left")

        elif command == "TURN_RIGHT":
            self.manual_action = 8
            self.stopped = False
            self.env.done = False
            self.env._envStepCounter = 0
            self.publish_status("manual_right")

        elif command in ["NO_COMMAND", ""]:
            # If a manual driving gesture was active, drop to forward straight
            # This lets the car coast forward after a turn is released
            if self.manual_action is not None:
                self.manual_action = 7  # forward straight
            return

        else:
            self.get_logger().warn(f"Ignoring unknown gesture command: {command}")

    def destroy_node(self):
        try:
            self.env.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = DrivingEnvNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from simple_driving.envs.simple_driving_env import SimpleDrivingEnv


def reward_callback(
    car_pos,
    goal_pos,
    obstacle_pos,
    has_obstacle,
    prev_dist_to_goal,
    dist_to_goal,
    reached_goal
):
    # Dummy reward because this file is only for viewing/control
    return 0.0


def observation_callback(
    client,
    car_pos,
    car_orn,
    goal_pos,
    goal_orn,
    obstacle_pos,
    has_obstacle
):
    car_x, car_y, _ = car_pos
    goal_x, goal_y, _ = goal_pos

    if has_obstacle and obstacle_pos is not None:
        obstacle_visible = 1.0
    else:
        obstacle_visible = 0.0

    return np.array(
        [car_x, car_y, goal_x, goal_y, obstacle_visible],
        dtype=np.float32
    )


class DrivingEnvironmentNode(Node):
    def __init__(self, scenario: str, target: str):
        super().__init__("driving_environment_node")

        self.scenario = scenario
        self.target_colour = target

        self.current_command = "stop"
        self.running = True

        # Publishers for GUI updates
        self.status_pub = self.create_publisher(String, "/car/status", 10)
        self.target_pub = self.create_publisher(String, "/car/target", 10)
        self.scenario_pub = self.create_publisher(String, "/car/scenario", 10)

        # Subscriber from hand gesture node
        self.create_subscription(
            String,
            "/gesture/command",
            self.gesture_command_callback,
            10
        )

        self.env = SimpleDrivingEnv(
            renders=True,
            isDiscrete=True,
            reward_callback=reward_callback,
            observation_callback=observation_callback
        )

        self.reset_environment(self.target_colour)

        self.get_logger().info("Driving environment node started.")
        self.get_logger().info("Listening for gesture commands on /gesture/command")

        self.publish_status("Environment ready. Waiting for hand gesture command.")
        self.publish_target()
        self.publish_scenario()

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------

    def gesture_command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.current_command = command

        self.get_logger().info(f"Gesture command received: {command}")

        if command in ["go_blue", "blue"]:
            self.target_colour = "blue"
            self.reset_environment("blue")
            self.publish_status("Target changed to blue block")

        elif command in ["go_red", "red"]:
            self.target_colour = "red"
            self.reset_environment("red")
            self.publish_status("Target changed to red block")

        elif command in ["stop", "open_palm"]:
            self.publish_status("Car stopped")

        elif command in ["forward", "go_forward", "point_forward"]:
            self.publish_status("Moving forward")

        elif command in ["backward", "reverse", "wave"]:
            self.publish_status("Moving backward")

        elif command in ["left", "move_left"]:
            self.publish_status("Turning left")

        elif command in ["right", "move_right"]:
            self.publish_status("Turning right")

        elif command in ["home", "fist"]:
            self.publish_status("Returning home command received")

        else:
            self.publish_status(f"Unknown gesture command: {command}")

    # ------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------

    def reset_environment(self, target_colour: str):
        self.obs, self.info = self.env.reset(options={
            "scenario": self.scenario,
            "target_colour": target_colour
        })

        self.get_logger().info("Environment reset")
        self.get_logger().info(f"Scenario: {self.scenario}")
        self.get_logger().info(f"Target colour: {target_colour}")

        if hasattr(self.env, "block_positions"):
            self.get_logger().info("Block positions:")
            for colour, pos in self.env.block_positions.items():
                self.get_logger().info(f"  {colour}: {pos}")

        self.publish_target()
        self.publish_scenario()

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_target(self):
        msg = String()
        msg.data = self.target_colour
        self.target_pub.publish(msg)

    def publish_scenario(self):
        msg = String()
        msg.data = self.scenario
        self.scenario_pub.publish(msg)

    # ------------------------------------------------------------
    # Main PyBullet loop
    # ------------------------------------------------------------

    def run_environment_loop(self):
        """
        This keeps PyBullet alive and moves the car based on gesture commands.

        IMPORTANT:
        The action numbers may need changing depending on your SimpleDrivingEnv.
        If the car moves the wrong way, swap these action numbers.
        """

        # Common discrete action guess.
        # Change these if your environment uses a different action mapping.
        ACTION_STOP = 0
        ACTION_FORWARD = 1
        ACTION_LEFT = 2
        ACTION_RIGHT = 3
        ACTION_BACKWARD = 4

        while rclpy.ok() and self.running:
            command = self.current_command

            if command in ["forward", "go_forward", "point_forward"]:
                action = ACTION_FORWARD

            elif command in ["backward", "reverse", "wave"]:
                action = ACTION_BACKWARD

            elif command in ["left", "move_left"]:
                action = ACTION_LEFT

            elif command in ["right", "move_right"]:
                action = ACTION_RIGHT

            elif command in ["stop", "open_palm"]:
                action = ACTION_STOP

            else:
                # For go_blue/go_red, we reset the target but do not manually drive.
                action = ACTION_STOP

            try:
                self.env.step(action)
            except Exception:
                # If env.step(action) does not work, still keep PyBullet alive.
                self.env._p.stepSimulation()

            time.sleep(0.01)

    def close(self):
        self.running = False
        self.env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        type=str,
        default="basic",
        choices=["basic", "distractors"],
        help="basic = red and blue only, distractors = red, blue, green and yellow"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="blue",
        choices=["blue", "red"],
        help="Starting target block colour"
    )

    args = parser.parse_args()

    rclpy.init()

    node = DrivingEnvironmentNode(
        scenario=args.scenario,
        target=args.target
    )

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        node.run_environment_loop()

    except KeyboardInterrupt:
        pass

    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
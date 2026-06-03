#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    project_dir = os.path.expanduser("~/git/AI_CNN/template")

    pythonpath = project_dir + ":" + os.environ.get("PYTHONPATH", "")

    return LaunchDescription([
        SetEnvironmentVariable(
            name="PYTHONPATH",
            value=pythonpath
        ),

        ExecuteProcess(
            cmd=[
                "python3",
                os.path.join(project_dir, "src", "hand_driving", "hand_driving", "driving_env_node.py"),
                "--ros-args", "-p", "renders:=false",   # disable PyBullet GUI window
            ],
            cwd=project_dir,
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "python3",
                os.path.join(project_dir, "src", "hand_gesture", "hand_gesture", "gesture_recognition_node.py"),
                "--ros-args",
                "-p", f"model_path:={os.path.join(project_dir, 'models', 'gesture_classifier.pkl')}",
                "-p", "show_window:=false",
                "-p", "smoothing_window:=4",
            ],
            cwd=project_dir,
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "python3",
                os.path.join(project_dir, "src", "gui", "gui", "app.py"),
            ],
            cwd=project_dir,
            output="screen",
        ),
    ])
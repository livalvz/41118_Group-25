import gymnasium as gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle
import matplotlib.pyplot as plt
import time

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'fp_camera', 'tp_camera', 'rgb_array']}

    def __init__(
        self, 
        isDiscrete=True, 
        renders=False, 
        minimum_safe_distance=1.0,
        reward_callback=None,
        observation_callback=None
    ):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -40, -40, 0], dtype=np.float32),
            high=np.array([40, 40, 40, 40, 1], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.obstacle_object = None
        self.obstacle_pos = None
        self.has_obstacle = False
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        
        # --- Configurable Limits ---
        self.minimum_safe_distance = minimum_safe_distance
        
        # Callbacks for Student Assignment
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        # ------------------------------------------------
        
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
          goal_pos, goal_orn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          car_ob = self.getExtendedObservation()

          # Distance to obstacle inside simulation steps
          if self.has_obstacle:
              dist_to_obs = math.sqrt((car_pos[0] - self.obstacle_pos[0])**2 + (car_pos[1] - self.obstacle_pos[1])**2)
              if dist_to_obs < self.minimum_safe_distance:
                  self.done = True
                  break

          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  # (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal = math.sqrt(((car_pos[0] - goal_pos[0]) ** 2 +
                                  (car_pos[1] - goal_pos[1]) ** 2))
                                  
        # Check termination constraints 
        if self.has_obstacle:
            dist_to_obs = math.sqrt((car_pos[0] - self.obstacle_pos[0])**2 + (car_pos[1] - self.obstacle_pos[1])**2)
            if dist_to_obs < self.minimum_safe_distance:
                self.done = True
                
        if dist_to_goal < 1.5 and not self.reached_goal:
            self.done = True
            self.reached_goal = True
            
        if self.reward_callback is not None:
             # Calculate reward via external function
             reward = self.reward_callback(
                 car_pos=car_pos, 
                 goal_pos=goal_pos,
                 obstacle_pos=self.obstacle_pos,
                 has_obstacle=self.has_obstacle,
                 prev_dist_to_goal=self.prev_dist_to_goal,
                 dist_to_goal=dist_to_goal,
                 reached_goal=self.reached_goal
             )
        else:
            raise ValueError("No reward_callback provided to SimpleDrivingEnv! You must inject the reward logic.")

        self.prev_dist_to_goal = dist_to_goal

        ob = np.array(car_ob, dtype=np.float32)
        return ob, float(reward), self.done, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # ----------------------------
        # Scenario / difficulty setup
        # ----------------------------

        scenario = options.get("scenario", "basic") if options else "basic"
        # WILL NEED TO CHANGE WHEN ADDING HAND GESTURES--------------------------------------------?
        target_colour = options.get("target_colour", "blue") if options else "blue"
    #    ADD HAND GESTURE TARGETS
        # if gesture == "one_finger":
        #     target_colour = "blue"
        # elif gesture == "two_fingers":
        #     target_colour = "red"

        # obs, info = env.reset(options={
        #     "scenario": "basic",
        #     "target_colour": target_colour
        # })



        self.done = False
        self.reached_goal = False
        self.has_obstacle = False
        self.obstacle_pos = None
        self.obstacle_object = None

        self.blocks = []
        self.block_positions = {}

        existing_positions = []

        # Red block
        red_pos = self.get_random_position(existing_positions)
        existing_positions.append(red_pos)

        red_block = self.create_coloured_block(
            position=red_pos,
            colour="red",
            collision=False
        )

        self.blocks.append(red_block)
        self.block_positions["red"] = red_pos

        # Blue block
        blue_pos = self.get_random_position(existing_positions)
        existing_positions.append(blue_pos)

        blue_block = self.create_coloured_block(
            position=blue_pos,
            colour="blue",
            collision=False
        )

        self.blocks.append(blue_block)
        self.block_positions["blue"] = blue_pos

        # Scenario 2: add two other coloured blocks as distractors
        if scenario == "distractors":
            distractor_colours = ["green", "yellow"]

            for colour in distractor_colours:
                pos = self.get_random_position(existing_positions)
                existing_positions.append(pos)

                block = self.create_coloured_block(
                    position=pos,
                    colour=colour,
                    collision=False
                )

                self.blocks.append(block)
                self.block_positions[colour] = pos

        # Select the actual goal based on hand gesture / target colour
        if target_colour == "red":
            self.goal = red_pos
            target_block = red_block
        else:
            self.goal = blue_pos
            target_block = blue_block

        # Keep compatibility with your current step() and getExtendedObservation()
        # because they expect self.goal_object.goal to exist.
        self.goal_object = type("GoalWrapper", (), {})()
        self.goal_object.goal = target_block

        print("Scenario:", scenario)
        print("Target colour:", target_colour)
        print("Block positions:", self.block_positions)

        # Get observation to return
        car_pos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(
            ((car_pos[0] - self.goal[0]) ** 2 +
            (car_pos[1] - self.goal[1]) ** 2)
        )

        car_ob = self.getExtendedObservation()

        return np.array(car_ob, dtype=np.float32), dict()


    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        goal_pos, goal_orn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        
        if self.observation_callback is not None:
             # Calculate observation block via external student function
             return self.observation_callback(
                 client=self._p,
                 car_pos=car_pos,
                 car_orn=car_orn,
                 goal_pos=goal_pos,
                 goal_orn=goal_orn,
                 obstacle_pos=self.obstacle_pos,
                 has_obstacle=self.has_obstacle
             )
        else:
             raise ValueError("No observation_callback provided to SimpleDrivingEnv! You must inject the observation logic.")
    
    # creates coloured blocks
    def create_coloured_block(self, position, colour, radius=0.5, height=1.0, collision=False):
        colour_map = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1],
            "green": [0, 1, 0, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [0.5, 0, 0.5, 1],
            "orange": [1, 0.5, 0, 1],
        }

        rgba = colour_map.get(colour, [1, 1, 1, 1])

        col_shape_id = -1
        if collision:
            col_shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_CYLINDER,
                radius=radius,
                height=height
            )

        vis_shape_id = self._p.createVisualShape(
            shapeType=self._p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=rgba
        )

        block_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[position[0], position[1], height / 2]
        )

        return block_id
    
    def get_random_position(self, existing_positions=None, min_distance=2.0):
        if existing_positions is None:
            existing_positions = []

        while True:
            x = self.np_random.uniform(-9, 9)
            y = self.np_random.uniform(-9, 9)

            dist_to_car = math.sqrt(x**2 + y**2)
            if dist_to_car < min_distance:
                continue

            too_close = False
            for pos in existing_positions:
                dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                return (x, y)

    def _termination(self):
        return self._envStepCounter > 4000

    def close(self):
        self._p.disconnect()

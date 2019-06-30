#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
"""
import csv
import math
import pprint
import time
from PIL import Image

import numpy as np

import airsim
import setup_path


class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state.x_val, self.state.y_val, self.state.z_val)
        self.quad_offset = (0, 0, 0)
        initX = 162
        initY = -320
        initZ = -150

        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False

        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()

        self.ep = 0

    def step(self, action):
        """Step"""
        print("doing step")
        self.quad_offset = self.interpret_action(action)
        print("quad_offset: ", self.quad_offset)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            20,
        ).join()
        time.sleep(0.5)

        collision_info = self.client.simGetCollisionInfo()

        if self.next_collision != collision_info.object_name:
            self.collision_change = True

        if collision_info.has_collided:
            if self.cnt_collision == 0:
                self.start_collision = collision_info.object_name
                self.next_collision = collision_info.object_name
                self.cnt_collision = 1
            else:
                self.next_collision = collision_info.object_name

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        print(
            "state x:",
            quad_state.x_val,
            " y: ",
            quad_state.y_val,
            " z: ",
            quad_state.z_val,
        )

        result = self.compute_reward(quad_state, quad_vel, collision_info)
        state = self.get_obs()
        done = self.isDone(result)
        return state, result, done

    def reset(self):
        """Reset to initial state"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state.x_val, self.state.y_val, self.state.z_val)
        self.quad_offset = (0, 0, 0)
        initX = 162
        initY = -320
        initZ = -150

        self.start_collision = "Cube"
        self.next_collision = "Cube"
        self.cnt_collision = 0
        self.collision_change = False

        self.client.takeoffAsync().join()
        print("take off moving positon")
        self.client.moveToPositionAsync(initX, initY, initZ, 5).join()
        responses = self.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )
        obs = self.transform_input(responses)

        return obs

    def get_obs(self):
        """Get observation"""
        responses = self.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )
        obs = self.transform_input(responses)
        return obs

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([-10, 10, -10])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self, quad_state, quad_vel, collision_info):
        """Compute reward"""
        thresh_dist = 7
        max_dist = 500
        beta = 1

        z = -10
        if self.ep == 0:
            if (
                self.collision_change == True
                and self.next_collision != self.start_collision
            ):
                if "Cube" in self.next_collision:
                    dist = 10000000
                    dist = self.get_distance(quad_state)
                    reward = 50000
                else:
                    reward = -100
            else:
                reward = 0
        else:
            if self.next_collision != self.start_collision:
                if "Cube" in self.next_collision:
                    dist = 10000000
                    dist = self.get_distance(quad_state)
                    reward = 50000
                else:
                    reward = -100
            else:
                reward = 0
        if quad_state.z_val < -280:
            reward = -100
        print(reward)
        return reward


    def isDone(self, reward):
        """Check if episode is done"""
        done = 0
        if reward <= -10:
            done = 1
            self.client.armDisarm(False)
            self.client.reset()
            self.client.enableApiControl(False)
            time.sleep(1)
        elif reward > 499:
            done = 1
            self.client.armDisarm(False)
            self.client.reset()
            self.client.enableApiControl(False)
            time.sleep(1)
        return done

    def transform_input(self, responses):
        """Transform input binary array to image"""
        response = responses[0]
        img1d = np.fromstring(
            response.image_data_uint8, dtype=np.uint8
        ) 
        img_rgba = img1d.reshape(
            response.height, response.width, 4
        ) 
        img2d = np.flipud(img_rgba)

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final

    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 5
        if action.item() == 0:
            self.quad_offset = (0, 0, 0)
        elif action.item() == 1:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action.item() == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action.item() == 3:
            self.quad_offset = (0, 0, scaling_factor)
        elif action.item() == 4:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action.item() == 5:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action.item() == 6:
            self.quad_offset = (0, 0, -scaling_factor)

        return self.quad_offset

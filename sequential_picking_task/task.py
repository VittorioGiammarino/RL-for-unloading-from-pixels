#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:51:28 2022

@author: vittoriogiammarino
"""
import random

import pybullet as p
import numpy as np
import time

from sequential_picking_task.robot import Robot, Suction, Cameras, PickPlace, PointCloud
from sequential_picking_task.boxes_generator import fill_template, generate_stack_of_boxes
from pathlib import Path

class env:
    def __init__(self, cfg, input_shape):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        
        self.total_eval_steps = 0
        
        self.main_dir = Path(__file__).parent
        self.KUKA_KR70_WORKSPACE_URDF_PATH = self.main_dir / 'assets/kuka_kr10/workspace.urdf'
        self.KUKA_KR70_PATH = self.main_dir / "assets/urdf/Kuka-KR70-URDF/urdf/Kuka-KR70-URDF.urdf"
        self.PLANE_URDF_PATH = self.main_dir / 'assets/plane/plane.urdf'
        self.BOX_TEMPLATE_URDF = self.main_dir / 'assets/box/box-template.urdf'
        self.SUCTION_BASE_URDF = self.main_dir / 'assets/urdf/ur5/suction/suction-base.urdf'
        self.SUCTION_HEAD_URDF = self.main_dir / 'assets/urdf/ur5/suction/suction-head.urdf'

        if self.cfg.reward_id==1:
            print("This reward provides additional bonus if successful picking happens at certain height")
        else:
            print("Standard reward: +1-accuracy for picking, 0 otherwise")

        self.info = {}
        
        if self.cfg.GUI:
            p.connect(p.GUI) #for image simulation
        else:
            p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.8)
        
        self.robot_in_the_scene = False #boolean describing whether robot is initialized
        self.container_height=7 #number of parcels one on top of another
        self.container_width=1 #number of walls
        self.container_length=6 #number of piles
            
        self.camera = Cameras(self.cfg.add_noise) 
        self.color_image, self.depth, self.segm = self.camera.get_image()
        self.point_cloud_generator = PointCloud(self.camera.CONFIG)
        
        self.resized_img_dim = (input_shape[1], input_shape[0])

        self.initialize_robot()

    def initialize_robot(self):
        if not self.robot_in_the_scene:
            self.planeID = p.loadURDF(self.PLANE_URDF_PATH.as_posix(), useFixedBase=1)
            self.workspaceID = p.loadURDF(self.KUKA_KR70_WORKSPACE_URDF_PATH.as_posix(), [1, 0 ,0])
            self.pose_robot_base = ((0, 0, 0), p.getQuaternionFromEuler((0, 0, 0)))
            self.kuka = Robot(self.KUKA_KR70_PATH.as_posix(), self.pose_robot_base)
            self.gripper = Suction(self.SUCTION_BASE_URDF.as_posix(), self.SUCTION_HEAD_URDF.as_posix(), self.kuka.robotId) 
            self.homing_not_succeeded = True
            self.speed = 0.03
            
            while self.homing_not_succeeded:
                self.homing_not_succeeded = self.kuka.placing_out_of_camera(self.speed)
                
            self.primitive = PickPlace(self.kuka, self.gripper)
                
            self.robot_in_the_scene = True
                
    def generate_boxes(self):
        margin = 0.02
        box_size = [.25, .25, .25]
        initial_position = (1.75+(.25/2), 0.6, 0.1) 
        orientation = p.getQuaternionFromEuler((0, 0, 0))
        box_urdf = fill_template(self.BOX_TEMPLATE_URDF, {'DIM': box_size})
        list_of_boxes = generate_stack_of_boxes(self.container_height, self.container_width, self.container_length, box_urdf, initial_position, orientation, margin)
        return list_of_boxes
    
    def get_image(self):
        self.color_image, self.depth, self.segm = self.camera.get_image()
        self.color_image, _, self.segm = self.kuka.crop_camera_images(self.color_image, self.depth, self.segm)
        camera_view_resized, segm_resized = self.kuka.resize_camera_images(self.color_image, self.segm, resize=self.resized_img_dim)
        input_image = camera_view_resized
        input_segm = segm_resized.reshape(segm_resized.shape[0], segm_resized.shape[1], 1)
        return input_image, input_segm
    
    def update_point_cloud(self):
        _, xyz_reshaped = self.point_cloud_generator.point_cloud_world_frame(self.depth)
        self.xyz_resized = self.kuka.resize_point_cloud(xyz_reshaped, resize=self.resized_img_dim)
        
    def step(self, action):
        
        picking_pixel_y, picking_pixel_x = action
        cartesian_position_box = self.xyz_resized[picking_pixel_y, picking_pixel_x]
            
        print(f"Selected xyz: {cartesian_position_box}")
        position0 = (cartesian_position_box[0], cartesian_position_box[1], cartesian_position_box[2]) 
        in_workspace = self.kuka.check_in_workspace(position0)
        
        # first check: be sure that the selected xyz lies within a pre-selected workspace
        if in_workspace:
            
            if self.cfg.side_pick_only: #this option forces a side pick
                pose0 = self.kuka.compute_picking_pose()  
                not_succeeded, suctioned_object, accuracy_error = self.primitive.SidePick(position0, pose0)
                
            else: #this option forces side or top pick according to 
                if cartesian_position_box[2]>=0.5:
                    pose0 = self.kuka.compute_picking_pose()  
                    not_succeeded, suctioned_object, accuracy_error = self.primitive.SidePick(position0, pose0)
                elif cartesian_position_box[2]<0.5:
                    pose0 = self.kuka.compute_top_picking_pose() 
                    not_succeeded, suctioned_object, accuracy_error = self.primitive.TopPick(position0, pose0)
                else:
                    print("Issues with pose decision, Side pick by default")
                    pose0 = self.kuka.compute_picking_pose()  
                    not_succeeded, suctioned_object, accuracy_error = self.primitive.SidePick(position0, pose0)
                
            if not not_succeeded and suctioned_object is not None:
                try:
                    print(f"Suctioned object: {suctioned_object}")
                    print(f"List of boxes ID: {self.list_of_boxes}")
                    p.removeBody(suctioned_object)
                    self.list_of_boxes.remove(suctioned_object)

                    if self.cfg.reward_id == 1:
                        #print(position0)
                        reward = (1+2*position0[2])
                    else:
                        reward = 1-self.cfg.accuracy_error_weight*accuracy_error

                    print(f"Good Job, {reward:.3f} reward")
                    self.info["num_picked_boxes"]+=1
                    
                    self.accuracy[suctioned_object-self.min_ID_boxes].append(accuracy_error)
                    
                except:
                    print("Error in removing suctioned object from list of boxes")
                    reward=0
                      
            else:
                # the agent might want to attempt a pick and get stuck in some configurations, 
                # this yields penalties and suboptimality of the agent
                self.homing_not_succeeded = True
                speed = 0.03
                reward = 0
                print(f"Something went wrong, {reward} reward")
                while self.homing_not_succeeded:
                    self.kuka.homing_joint_control()
                    self.homing_not_succeeded = self.kuka.placing_out_of_camera(speed)
                    box_to_discard_ID = self.list_of_boxes[-1]
                    p.removeBody(box_to_discard_ID)
                    self.list_of_boxes.remove(box_to_discard_ID)
                    
        else:
            self.homing_not_succeeded = True
            speed = 0.03
            reward = 0
            print(f"Selected point out of safety workspace, {reward} reward")
            self.info["num_out_workspace"]+=1
            while self.homing_not_succeeded:
                self.kuka.homing_joint_control()
                self.homing_not_succeeded = self.kuka.placing_out_of_camera(speed)
                box_to_discard_ID = self.list_of_boxes[-1]
                p.removeBody(box_to_discard_ID)
                self.list_of_boxes.remove(box_to_discard_ID) 
                
        input_image, input_semg = self.get_image()
        self.update_point_cloud()
        
        if len(self.list_of_boxes)==0:
            done = True
        else:
            done = False
                    
        return input_image, input_semg, reward, done, self.info
                             
    def reset(self):
        p.resetSimulation()
        self.robot_in_the_scene = False
        p.setGravity(0, 0, -9.8)
        self.initialize_robot()
        self.list_of_boxes = self.generate_boxes()
        
        self.accuracy = [[] for a in range(self.container_length*self.container_height)]
        self.min_ID_boxes = min(self.list_of_boxes)
        input_image, input_segm = self.get_image()
        self.update_point_cloud()

        self.info["num_picked_boxes"]=0
        self.info["num_out_workspace"]=0
        
        return input_image, input_segm
        
    def eval_episode(self, agent):
        input_image, input_segm = self.reset()
        
        time_step = 0
        episode_reward = 0
        start = time.time()
        
        while True:             
            picking_pixel_y, picking_pixel_x = agent.act(input_image)
            print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
            action = (picking_pixel_y, picking_pixel_x)
                
            input_image, input_segm, reward, done, info = self.step(action) 
            
            episode_reward+=reward
            time_step+=1
            self.total_eval_steps+=1
            
            if done:
                break
            
            if episode_reward<0:
                for i in range(len(self.list_of_boxes)):
                    p.removeBody(self.list_of_boxes[i])
                    p.stepSimulation()
                    
                self.list_of_boxes = []
                break
            
        end = time.time() - start
        print(f"Total Time: {end}")
        
        return episode_reward, self.accuracy
         
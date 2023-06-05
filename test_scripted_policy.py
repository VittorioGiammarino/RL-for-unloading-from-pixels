#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:06:47 2022

@author: vittoriogiammarino
"""
import random
import hydra
import pickle
import time

import pybullet as p
import numpy as np

from sequential_picking_task.robot import Robot, Suction, Cameras, PickPlace, PointCloud
from sequential_picking_task.boxes_generator import fill_template, generate_stack_of_boxes
from pathlib import Path
        
class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        
        self.expert_episodes = self.cfg.num_episodes
        self.seed = self.cfg.seed
        
        self.main_dir = Path(__file__).parent
        self.KUKA_KR70_WORKSPACE_URDF_PATH = self.main_dir / 'sequential_picking_task/assets/kuka_kr10/workspace.urdf'
        self.KUKA_KR70_PATH = self.main_dir / "sequential_picking_task/assets/urdf/Kuka-KR70-URDF/urdf/Kuka-KR70-URDF.urdf"
        self.PLANE_URDF_PATH = self.main_dir / 'sequential_picking_task/assets/plane/plane.urdf'
        self.BOX_TEMPLATE_URDF = self.main_dir / 'sequential_picking_task/assets/box/box-template.urdf'
        self.SUCTION_BASE_URDF = self.main_dir / 'sequential_picking_task/assets/urdf/ur5/suction/suction-base.urdf'
        self.SUCTION_HEAD_URDF = self.main_dir / 'sequential_picking_task/assets/urdf/ur5/suction/suction-head.urdf'
        
        if self.cfg.GUI:
            p.connect(p.GUI) #for image simulation
        else:
            p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.8)
        
        self.set_seed()
        self.robot_in_the_scene = False
        self.container_height=7
        self.container_width=1
        self.container_length=6
        
        self.camera = Cameras(self.cfg.add_noise)
        self.color_image, self.depth, self.segm = self.camera.get_image()
        self.point_cloud_generator = PointCloud(self.camera.CONFIG)
        
        self.resized_img_dim = (self.cfg.resize_width, self.cfg.resize_height)
        
    def set_seed(self):
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        
    def initialize_robot(self):
        if not self.robot_in_the_scene:
            self.planeID = p.loadURDF(self.PLANE_URDF_PATH.as_posix(), useFixedBase=1)
            self.workspaceID = p.loadURDF(self.KUKA_KR70_WORKSPACE_URDF_PATH.as_posix(), [1, 0 ,0])
            self.pose_robot_base = ((0, 0, 0), p.getQuaternionFromEuler((0, 0, 0)))
            self.kuka = Robot(self.KUKA_KR70_PATH.as_posix(), self.pose_robot_base, parcel_chooser=self.cfg.parcel_chooser)
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
    
    def store(self):
        buffer = {}
        buffer["real_image"] = []
        buffer["image"] = []
        buffer["segm"] = []
        buffer["action"] = []
        buffer["top pick"] = []
        buffer["reward"] = []
        buffer["terminal"] = []
        buffer["accuracy error"] = [[] for a in range(self.container_length*self.container_height)]
        
        for _ in range(self.expert_episodes):
            p.resetSimulation()
            self.robot_in_the_scene = False
            p.setGravity(0, 0, -9.8)
            
            step = 0
            episode = 0
            episode_reward = 0
            self.initialize_robot()
            list_of_boxes = self.generate_boxes()
            min_ID_boxes = min(list_of_boxes)
            
            top_pick = False
            start = time.time()
            
            while len(list_of_boxes)>0:      
                cartesian_position_box = []
                self.color_image, self.depth, self.segm = self.camera.get_image() # get an image from camera
                buffer["real_image"].append(self.color_image)

                self.color_image, _, self.segm = self.kuka.crop_camera_images(self.color_image, self.depth, self.segm) # crop the area of interest
                
                camera_view_resized, segm_resized = self.kuka.resize_camera_images(self.color_image, self.segm, resize=self.resized_img_dim) # resize the image in the way you prefer          
                H,W,C = camera_view_resized.shape
                
                _, xyz_reshaped = self.point_cloud_generator.point_cloud_world_frame(self.depth) # generate point cloud for the original image
                xyz_resized = self.kuka.resize_point_cloud(xyz_reshaped, resize=self.resized_img_dim) # resize point cloud in order to align it with the resized image
                picking_pixel_y, picking_pixel_x, discard_box, box_to_discard_ID = self.kuka.select_a_box(segm_resized, list_of_boxes) # select a pixel for picking using the cropped and resized segmentation
                
                # when discard_box is True then the agent decides it is not going to attempt a pick for that box, hence small reward is given
                # when discard box is True then picking_pixel_y and picking_pixel_x are None
                
                if picking_pixel_y is not None and picking_pixel_x is not None: 
                    cartesian_position_box = xyz_resized[picking_pixel_y, picking_pixel_x] # Use the resized point-cloud to obtain a xyz in world frame
                    
                    if self.cfg.side_pick_only:
                        position0 = (cartesian_position_box[0], cartesian_position_box[1], cartesian_position_box[2])
                        pose0 = self.kuka.compute_picking_pose()
                        top_pick = False
                        
                    else:
                        if cartesian_position_box[2]<0.5: # if estimated z value less than 0.5
                            # we go for a top pick
                            picking_pixel_y, picking_pixel_x, discard_box, box_to_discard_ID = self.kuka.select_box_top_pick(segm_resized, list_of_boxes)
                            cartesian_position_box = xyz_resized[picking_pixel_y, picking_pixel_x]
                            position0 = (cartesian_position_box[0], cartesian_position_box[1], cartesian_position_box[2])
                            pose0 = self.kuka.compute_top_picking_pose()
                            top_pick = True
                        else:
                            position0 = (cartesian_position_box[0], cartesian_position_box[1], cartesian_position_box[2])
                            pose0 = self.kuka.compute_picking_pose()
                            top_pick = False
                    
                if discard_box: # box selection is done a priori, it might happen that the box is not in the segmentation view, it gets discarded by default
                    self.homing_not_succeeded = True
                    speed = 0.03
                    
                    while self.homing_not_succeeded:
                        self.kuka.homing_joint_control()
                        self.homing_not_succeeded = self.kuka.placing_out_of_camera(speed)
                        p.removeBody(box_to_discard_ID)
                        list_of_boxes.remove(box_to_discard_ID)
                        reward = 0.01
                        episode_reward+=reward
                        
                        print("Box succesfully discarded, +0.01 reward")
                        
                else:
                    # Here we are going to check that the computed position0 is within a predefined workspace
                    in_workspace = self.kuka.check_in_workspace(position0)
                    
                    if in_workspace: # if the selected point is in the workspace we attempt picking 
                        if top_pick:
                            not_succeeded, suctioned_object, accuracy_error = self.primitive.TopPick(position0, pose0)
                        else:
                            not_succeeded, suctioned_object, accuracy_error = self.primitive.SidePick(position0, pose0)
                            
                        if not not_succeeded and suctioned_object is not None: # if picking succeeded then we store in the buffer
                            try:
                                p.removeBody(suctioned_object)
                                list_of_boxes.remove(suctioned_object)
                                reward = 1
                                episode_reward+=reward
                                print("Good Job, +1 reward")
                                
                                buffer["image"].append(camera_view_resized)
                                buffer["segm"].append(segm_resized.reshape(segm_resized.shape[0], segm_resized.shape[1], 1))
                                buffer["action"].append([picking_pixel_y, picking_pixel_x])
                                buffer["top pick"].append(top_pick)
                                buffer["reward"].append(reward)
                                buffer["accuracy error"][suctioned_object-min_ID_boxes].append(accuracy_error)
                                
                                if len(list_of_boxes)==0:
                                    buffer["terminal"].append(True)
                                else:
                                    buffer["terminal"].append(False)
                                    
                            except:
                                print("Error in remove suctioned object from list of boxes")
                                continue
                            
                        else:
                            # the agent might want to attempt a pick and get stuck in some configurations, 
                            # this yields penalties and suboptimality of the agent
                            reward = -1
                            episode_reward+=reward
                            print("Something wrong with picking success, -1 penalty")
                            self.homing_not_succeeded = True
                            speed = 0.03
                            while self.homing_not_succeeded:
                                self.kuka.homing_joint_control()
                                self.homing_not_succeeded = self.kuka.placing_out_of_camera(speed)
                                p.removeBody(box_to_discard_ID)
                                list_of_boxes.remove(box_to_discard_ID)
                                
                    else:
                        reward = -1
                        episode_reward+=reward
                        print("Selected point out of safety workspace, -1 penalty")
                        self.homing_not_succeeded = True
                        speed = 0.03
                        while self.homing_not_succeeded:
                            self.kuka.homing_joint_control()
                            self.homing_not_succeeded = self.kuka.placing_out_of_camera(speed)
                            p.removeBody(box_to_discard_ID)
                            list_of_boxes.remove(box_to_discard_ID)       
                    
                step += 1
                print(f"Step: {step}, episode reward: {episode_reward}")
         
            end = time.time() - start
            episode+=1
            print(f"Episode: {episode}, Total Time: {end}")
            
        f = open("expert_data.pkl", "wb")
        pickle.dump(buffer, f)
        f.close()
        
@hydra.main(config_path='config_folder', config_name='config_generate_expert_traj')
def main(cfg):
    from test_scripted_policy import Workspace as W
    workspace = W(cfg)
    workspace.store()

if __name__ == '__main__':
    main()

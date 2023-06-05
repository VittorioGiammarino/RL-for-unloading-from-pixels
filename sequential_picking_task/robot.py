#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:30:44 2022

@author: vittoriogiammarino
"""

import cv2
import pybullet as p
import numpy as np
import time
import copy
import pyrealsense2 as rs

class Robot:
    def __init__(self, URDF_PATH, pose_robot_base, parcel_chooser='Heuristic'):
        
        FLAGS = p.URDF_MERGE_FIXED_LINKS or p.URDF_USE_SELF_COLLISION
        self.robotId = p.loadURDF(URDF_PATH, pose_robot_base[0], pose_robot_base[1], useFixedBase=True, flags=FLAGS)
        self.parcel_chooser = parcel_chooser
        
        """ Info taken from Kuka KR70 """
        self.adjust_joints_limits = 6
        self.JointsLowerLimits = [-3.22885912, -3.0543, -2.0943, -3.14159, -2.1817, -6.109]*self.adjust_joints_limits
        self.JointsUpperLimits = [3.22885912, 1.04719, 2.8798, 3.14159, 2.18166, 6.109]*self.adjust_joints_limits
        self.JointRanges = [6.45771824, 4.10149, 4.9741, 6.28318, 4.36332, 12.218]*self.adjust_joints_limits
        
        self.PickFromTopOrientation = p.getQuaternionFromEuler((0, np.pi/2, 0))
        self.PickFromSideOrientation = p.getQuaternionFromEuler((0, 0, 0))
        
        self.homej = np.array([0.5, 0, 0, 0, 1, 0])*np.pi
        
        self.n_joints = p.getNumJoints(self.robotId)
        joints = [p.getJointInfo(self.robotId, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2]==p.JOINT_REVOLUTE]

        self.homing_joint_control()
            
        self.ee_tip_ID = 5
        
        self.placing_pose = ((0.35, -1, 1.5), p.getQuaternionFromEuler((0, np.pi/2, 0))) #0.35, -0.779, 1.5
        self.out_of_camera = ((0.35, -1, 0.5), p.getQuaternionFromEuler((0, np.pi/2, 0)))
        
        self.workspace = [[0.5, 2.5], [-1, 1], [0.14, 2.5]]
            
    def solve_ik(self, targetPose):
        targj = p.calculateInverseKinematics(bodyUniqueId=self.robotId,
                                             endEffectorLinkIndex=self.ee_tip_ID,
                                             targetPosition=targetPose[0],
                                             targetOrientation=targetPose[1],
                                             lowerLimits = self.JointsLowerLimits,
                                             upperLimits = self.JointsUpperLimits,
                                             jointRanges = self.JointRanges,  
                                             restPoses=np.float32(self.homej).tolist(),
                                             maxNumIterations=100,
                                             residualThreshold=1e-5)   
        
        return targj
    
    def movej(self, targj, speed = 0.0005, timeout=10):
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.robotId, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj)<1e-2):
                return False
            
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm >0 else 0
            stepj = currj + v*speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(bodyIndex=self.robotId, 
                                        jointIndices=self.joints, 
                                        controlMode = p.POSITION_CONTROL,
                                        targetPositions = stepj, 
                                        positionGains=gains)       
            p.stepSimulation()
            
        print(f'Warning: movej exceeded {timeout} seconds timeout. Skipping.')
        return True
    
    def movep(self, pose, speed=0.008):
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)
    
    def placing_on_conveyor(self):
        success = self.movep(self.placing_pose)
        return success
    
    def placing_out_of_camera(self, speed=0.001):
        success = self.movep(self.out_of_camera, speed)
        return success
    
    def homing_joint_control(self):
        for k in range(len(self.joints)):
            p.resetJointState(self.robotId, self.joints[k], self.homej[k])
           
    def select_a_box(self, segm, list_of_boxes_ID):

        if self.parcel_chooser == 'Heuristic':
            j=-1
        elif self.parcel_chooser == 'Random':
            j = np.random.choice(list_of_boxes_ID)
        else:
            NotImplementedError

        height_offset = 3
        try:
            boxID = list_of_boxes_ID[j]
        except:
            boxID = list_of_boxes_ID[-1]
        
        box_in_segm = np.where(segm==boxID)
        height_box_pixels = box_in_segm[0]
        length_box_pixels = box_in_segm[1]
        
        if len(height_box_pixels)==0 or len(length_box_pixels)==0:
            discard_box = True
            return None, None, discard_box, boxID
        else:
            discard_box = False

        try:
            middle_height = int(len(height_box_pixels)/2)
            picking_y = height_box_pixels[middle_height]-height_offset
    
            pixels_x_of_interest_index = np.where(height_box_pixels==picking_y)[0]
            pixels_x_of_interest = length_box_pixels[pixels_x_of_interest_index]
            middle_length = int(len(pixels_x_of_interest)/2)
            picking_x = pixels_x_of_interest[middle_length]
        except:
            discard_box = True
            return None, None, discard_box, boxID
        
        return picking_y, picking_x, discard_box, boxID
    
    def select_box_top_pick(self, segm, list_of_boxes_ID):

        if self.parcel_chooser == 'Heuristic':
            j=-1
        elif self.parcel_chooser == 'Random':
            j = np.random.choice(list_of_boxes_ID)
        else:
            NotImplementedError
        
        offset_y = 4 #7 if image size 320x320 works better
        offset_x = 4 #7
        
        try:
            boxID = list_of_boxes_ID[j]
        except:
            boxID = list_of_boxes_ID[-1]
        
        box_in_segm = np.where(segm==boxID)
        height_box_pixels = box_in_segm[0]
        length_box_pixels = box_in_segm[1]
        
        if len(height_box_pixels)==0 or len(length_box_pixels)==0:
            discard_box = True
            return None, None, discard_box, boxID
        else:
            discard_box = False

        try:
            picking_y = min(height_box_pixels)+offset_y
            pixels_x_of_interest_index = np.where(height_box_pixels==picking_y)[0]
            pixels_x_of_interest = length_box_pixels[pixels_x_of_interest_index]
            middle_length = int(len(pixels_x_of_interest)/2)+offset_x
            picking_x = pixels_x_of_interest[middle_length]
        except:
            discard_box = True
            return None, None, discard_box, boxID
        
        return picking_y, picking_x, discard_box, boxID
        
    def select_a_box_w_Oracle(self, list_of_boxes_ID):
        j = -1
        cartesian_position_box = p.getBasePositionAndOrientation(list_of_boxes_ID[j])[0]
        
        return cartesian_position_box
    
    def crop_camera_images(self, color, depth, segm):
        self.height_lower_bound = 75 
        self.height_upper_bound = 599
        self.width_lower_bound = 350
        self.width_upper_bound = 845
        
        color_image_cropped = color[self.height_lower_bound:self.height_upper_bound, 
                                    self.width_lower_bound:self.width_upper_bound, :]
        
        depth_cropped = depth[self.height_lower_bound:self.height_upper_bound, 
                              self.width_lower_bound:self.width_upper_bound]
        
        segm_cropped = segm[self.height_lower_bound:self.height_upper_bound, 
                            self.width_lower_bound:self.width_upper_bound]
        
        return color_image_cropped, depth_cropped, segm_cropped
    
    def resize_camera_images(self, RGB_image, segm, resize=(320,160)):
        # opencv wants WxH rather than HxW
        image_resized = cv2.resize(RGB_image, resize, interpolation = cv2.INTER_LINEAR)
        segm_resized = cv2.resize(segm, resize, interpolation = cv2.INTER_NEAREST)
        
        return image_resized, segm_resized
    
    def resize_point_cloud(self, xyz_reshaped, resize=(320,160)):
        xyz_cropped = xyz_reshaped[self.height_lower_bound:self.height_upper_bound, 
                                   self.width_lower_bound:self.width_upper_bound, :]
        xyz_resized = cv2.resize(xyz_cropped, resize, interpolation = cv2.INTER_LINEAR)
        
        return xyz_resized
    
    def compute_picking_pose(self):
        pose = self.PickFromSideOrientation
        return pose
    
    def compute_top_picking_pose(self):
        pose = self.PickFromTopOrientation
        return pose
    
    def check_in_workspace(self, cartesian_position_box):
        in_workspace = False
        x = cartesian_position_box[0]
        y = cartesian_position_box[1]
        z = cartesian_position_box[2]
        
        print(f"Selected Pick: {cartesian_position_box}")
        
        if x>=self.workspace[0][0] and x<=self.workspace[0][1]:
            if y>=self.workspace[1][0] and y<=self.workspace[1][1]:
                if z>=self.workspace[2][0] and z<=self.workspace[2][1]:
                    in_workspace = True
                    
        return in_workspace
         
class Suction:
    def __init__(self, URDF_BASE, URDF_HEAD, robotId):
        self.robotId = robotId
        self.suction = p.loadURDF(URDF_BASE)
        p.createConstraint(parentBodyUniqueId=self.robotId, 
                           parentLinkIndex=5, 
                           childBodyUniqueId=self.suction, 
                           childLinkIndex=-1, 
                           jointType=p.JOINT_FIXED, 
                           jointAxis=(0, 0, 0), 
                           parentFramePosition=(0, 0, 0), 
                           childFramePosition=(0, 0, -0.01), 
                           parentFrameOrientation=p.getQuaternionFromEuler((0, np.pi, 0)), 
                           childFrameOrientation=p.getQuaternionFromEuler((0, np.pi/2, 0)))
        
        self.suction_tip = p.loadURDF(URDF_HEAD)
        p.createConstraint(parentBodyUniqueId=self.robotId, 
                           parentLinkIndex=5, 
                           childBodyUniqueId=self.suction_tip, 
                           childLinkIndex=-1, 
                           jointType=p.JOINT_FIXED, 
                           jointAxis=(0, 0, 0), 
                           parentFramePosition=(0, 0, 0), 
                           childFramePosition=(0, 0, -0.1),
                           parentFrameOrientation=p.getQuaternionFromEuler((0, np.pi, 0)), 
                           childFrameOrientation=p.getQuaternionFromEuler((0, np.pi/2, 0)))     
        
        self.ee_tip = 5
        
        # Indicates whether gripping or not
        self.activated = False
        self.contact_constraint = None
        
    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        error = None
        if not self.activated:
            points = p.getContactPoints(bodyA=self.suction_tip, linkIndexA=0)
            
            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                    
                if obj_id != 0 and obj_id != 1:
                    body_pose = p.getLinkState(self.suction_tip, 0)
                    print(f"ee position: {body_pose[0]}")
                    
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    print(f"box position: {obj_pose[0]}")
                    
                    error = np.sqrt((body_pose[0][0]-obj_pose[0][0])**2+(body_pose[0][1]-obj_pose[0][1])**2+(body_pose[0][2]-obj_pose[0][2])**2)
                    print(f"accuracy error: {error}")
                    
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                       world_to_body[1],
                                                       obj_pose[0], 
                                                       obj_pose[1])
                    self.contact_constraint = p.createConstraint(parentBodyUniqueId=self.suction_tip,
                                                                 parentLinkIndex=0,
                                                                 childBodyUniqueId=obj_id,
                                                                 childLinkIndex=contact_link,
                                                                 jointType=p.JOINT_FIXED,
                                                                 jointAxis=(0, 0, 0),
                                                                 parentFramePosition=obj_to_body[0],
                                                                 parentFrameOrientation=obj_to_body[1],
                                                                 childFramePosition=(0, 0, 0),
                                                                 childFrameOrientation=(0, 0, 0))
    
                    self.activated = True
                    
        return error
                
    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.suction_tip, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        # print(points)
        # exit()
        if self.activated:
            points = [point for point in points if point[2] != self.suction_tip]

        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None, suctioned_object
        
class Cameras:
    def __init__(self, add_noise=False, image_size = (720, 1280, 3), position = (-0.205, 0.859, 1.764), euler_rotation = (1.155, 3.156, -5.154)):
        
        self._random = np.random.RandomState(seed=10)

        self.image_size = image_size
        self.intrinsics = (639.2, 0, 640.53, 0, 639.2, 335.89, 0, 0, 1) #real
        # self.rotation_euler_angles = (1.174, 3.283, 1.31)
        self.rotation_euler_angles = euler_rotation
        self.rotation = p.getQuaternionFromEuler(self.rotation_euler_angles)
        # self.position = (-0.44, 1.053, 1.824)
        self.position = position
        self.CONFIG = [{'image_size': self.image_size,
                        'intrinsics': self.intrinsics,
                        'position': self.position,
                        'rotation': self.rotation,
                        'zrange': (0.6, 6),
                        'noise': add_noise}]
        
    def setup(self):
        config = self.CONFIG[0]
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1) # np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (config['image_size'][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)
        
        return viewm, projm
    
    def get_image(self):
        # Render with OpenGL camera settings.
        config = self.CONFIG[0]
        znear, zfar = config['zrange']
        viewm, projm = self.setup()
        _, _, color, depth, segm = p.getCameraImage(width=config['image_size'][1],
                                                    height=config['image_size'][0],
                                                    viewMatrix=viewm,
                                                    projectionMatrix=projm,
                                                    shadow=1,
                                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # Get color image.
        color_image_size = (config['image_size'][0],
                            config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config['image_size']))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        # one approach
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        
        # another approach
        # depth = zfar * znear / (zfar - (zfar - znear) * zbuffer)
        
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        
        return color, depth, segm
    
    
class PointCloud:
    def __init__(self, camera_config):
        self.camera_config = camera_config[0]
        
        H,W,ch = self.camera_config['image_size']
        self.height_image = H
        self.width_image = W
        self.intrinsics = np.array(self.camera_config['intrinsics']).reshape(3, 3)
        self.position = np.array(self.camera_config['position']).reshape(3,1)
        rotation = p.getMatrixFromQuaternion(self.camera_config['rotation'])
        self.rotation = np.array(rotation).reshape(3,3)
        
        self.get_camera_intrinsics_object()
        
    def get_camera_intrinsics_object(self):
        self.intrinsics_object = rs.intrinsics()
        self.intrinsics_object.width = self.width_image
        self.intrinsics_object.height = self.height_image
        self.intrinsics_object.ppx = self.intrinsics[0,2]
        self.intrinsics_object.ppy = self.intrinsics[1,2]
        self.intrinsics_object.fx = self.intrinsics[0,0]
        self.intrinsics_object.fy = self.intrinsics[1,1]
        
    def point_cloud_sensor_frame(self, depth):
        
        xlin = np.linspace(0, self.width_image - 1, self.width_image)
        ylin = np.linspace(0, self.height_image - 1, self.height_image)
        px, py = np.meshgrid(xlin, ylin)
        
        pixels = np.zeros((self.height_image, self.width_image, 2))
        pixels[:,:,0] = px
        pixels[:,:,1] = py
        
        pixels = pixels.reshape(self.height_image*self.width_image, 2)
        depth_reshaped = depth.reshape(self.height_image*self.width_image)
        
        point_cloud = np.array([rs.rs2_deproject_pixel_to_point(self.intrinsics_object, list(pixel), d) for pixel, d in zip(pixels, depth_reshaped)])
        point_cloud_reshaped = point_cloud.reshape((self.height_image, self.width_image, 3))
        
        return point_cloud, point_cloud_reshaped
    
    def get_transformation_camera_to_world_frame(self):
        mat = np.eye(4)
        mat[:3, 3] = self.camera_config['position']
        mat[:3, :3] = self.rotation
        
        return mat
    
    def point_cloud_world_frame(self, depth):
        
        H,W = depth.shape
        
        point_cloud, _ = self.point_cloud_sensor_frame(depth)
        transformation_matrix = self.get_transformation_camera_to_world_frame()
        
        out = np.matmul(transformation_matrix[:3, :3], np.asarray(point_cloud).T).T
        xyz = np.add(out, transformation_matrix[:3, 3])
        xyz_reshaped = xyz.reshape(H, W, 3)
        
        return xyz, xyz_reshaped
        
        
class PickPlace:
    def __init__(self, robot, gripper):
        self.robot = robot 
        self.gripper = gripper
        self.heigth_delta_postpick = 0.1 
        self.y_delta_postpick = 0.15
        self.x_delta_postpick = 1
        
        self.heigth_delta_top_pick = 1 
        
    def multiply(self, pose0, pose1):
        return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])
        
    def SidePick(self, position0, pose0):
        pick_position = position0
        pick_pose = pose0
        # Execute picking primitive.
        
        prepick_position = list(copy.deepcopy(pick_position))
        prepick_position[0]= prepick_position[0]-0.5
        prepick_orientation = pick_pose #self.robot.PickFromSideOrientation
        prepick_pose = (tuple(prepick_position), prepick_orientation)
        timeout = self.robot.movep(prepick_pose)
        
        delta = +0.001
        Delta = 0
        targ_position = list(prepick_position)
        while not self.gripper.detect_contact():  # and target_pose[2] > 0:
            targ_position[0] = targ_position[0]+delta
            target_pose = (tuple(targ_position), prepick_orientation)
            
            Delta += delta
            if Delta>2:
                return True, None, None #extra check, tentative bug removal
            
            timeout |= self.robot.movep(target_pose)
            if timeout:
                print("Prepick failed")
                return True, None, None
            
        error = self.gripper.activate()
        postpick_position = list(copy.deepcopy(pick_position))
        postpick_position[0] = postpick_position[0] - self.x_delta_postpick
        postpick_position[1] = postpick_position[1] - self.y_delta_postpick
        postpick_position[2] = postpick_position[2] + self.heigth_delta_postpick
        postpick_orientation = pick_pose #self.robot.PickFromSideOrientation
        postpick_pose = (tuple(postpick_position), postpick_orientation)
        timeout |= self.robot.movep(postpick_pose)
        pick_success, suctioned_object = self.gripper.check_grasp()
        
        if pick_success:
            preplace_position_1 = list(copy.deepcopy(postpick_position))
            
            preplace_position_1_offset_z = 1.5 - preplace_position_1[2]
            
            preplace_position_1[2]= preplace_position_1[2]+preplace_position_1_offset_z
            preplace_orientation_1 = pick_pose #self.robot.PickFromSideOrientation
            preplace_pose_1 = (tuple(preplace_position_1), preplace_orientation_1)
            timeout |= self.robot.movep(preplace_pose_1)
            preplace_position_2 = list(copy.deepcopy(preplace_position_1))
            preplace_orientation_2 = self.robot.PickFromTopOrientation
            preplace_pose_2 = (tuple(preplace_position_2), preplace_orientation_2)
            timeout |= self.robot.movep(preplace_pose_2)
            
            if not timeout:
                timeout |= self.robot.placing_on_conveyor()
                delta = -0.001
                placing_position = list(self.robot.placing_pose[0])
                placing_orientation = self.robot.placing_pose[1]
                while not self.gripper.detect_contact():
                    placing_position[2] = placing_position[2] + delta
                    placing_pose = (tuple(placing_position), placing_orientation)
                    timeout |= self.robot.movep(placing_pose)
                    if timeout:
                        print("Placing failed")
                        return True, None, None
                self.gripper.release()
                timeout |= self.robot.placing_on_conveyor()
                
                
        else:
            self.gripper.release()
            timeout |= self.robot.movep(prepick_pose)
            
        return timeout, suctioned_object, error
                
    def TopPick(self, position0, pose0):
        pick_position = position0
        pick_pose = pose0
        # Execute picking primitive.
        
        prepick_position = list(copy.deepcopy(pick_position))
        prepick_position[2]= prepick_position[2]+0.5
        prepick_orientation = pick_pose #self.robot.PickFromSideOrientation
        prepick_pose = (tuple(prepick_position), prepick_orientation)
        timeout = self.robot.movep(prepick_pose)
        
        delta = +0.001
        Delta = 0
        targ_position = list(prepick_position)
        while not self.gripper.detect_contact():  # and target_pose[2] > 0:
            targ_position[2] = targ_position[2]-delta
            target_pose = (tuple(targ_position), prepick_orientation)
            
            Delta += delta
            if Delta>2:
                return True, None, None #extra check, tentative bug removal
            
            timeout |= self.robot.movep(target_pose)
            if timeout:
                print("Prepick failed")
                return True, None, None
            
        error = self.gripper.activate()
        postpick_position = list(copy.deepcopy(pick_position))
        postpick_position[0] = postpick_position[0] - self.x_delta_postpick
        postpick_position[1] = postpick_position[1] - self.y_delta_postpick
        postpick_position[2] = postpick_position[2] + self.heigth_delta_top_pick
        postpick_orientation = pick_pose #self.robot.PickFromSideOrientation
        postpick_pose = (tuple(postpick_position), postpick_orientation)
        timeout |= self.robot.movep(postpick_pose)
        pick_success, suctioned_object = self.gripper.check_grasp()
        
        if pick_success:
            preplace_position_1 = list(copy.deepcopy(postpick_position))
            
            preplace_position_1_offset_z = 1.5 - preplace_position_1[2]
            
            preplace_position_1[2]= preplace_position_1[2]+preplace_position_1_offset_z
            preplace_orientation_1 = pick_pose #self.robot.PickFromSideOrientation
            preplace_pose_1 = (tuple(preplace_position_1), preplace_orientation_1)
            timeout |= self.robot.movep(preplace_pose_1)
            preplace_position_2 = list(copy.deepcopy(preplace_position_1))
            preplace_orientation_2 = self.robot.PickFromTopOrientation
            preplace_pose_2 = (tuple(preplace_position_2), preplace_orientation_2)
            timeout |= self.robot.movep(preplace_pose_2)
            
            if not timeout:
                timeout |= self.robot.placing_on_conveyor()
                delta = -0.001
                placing_position = list(self.robot.placing_pose[0])
                placing_orientation = self.robot.placing_pose[1]
                while not self.gripper.detect_contact():
                    placing_position[2] = placing_position[2] + delta
                    placing_pose = (tuple(placing_position), placing_orientation)
                    timeout |= self.robot.movep(placing_pose)
                    if timeout:
                        print("Placing failed")
                        return True, None, None
                self.gripper.release()
                timeout |= self.robot.placing_on_conveyor()
                
                
        else:
            self.gripper.release()
            timeout |= self.robot.movep(prepick_pose)
            
        return timeout, suctioned_object, error        
        
        
        
        
        
        
        
        
        
        
    
        


        
        
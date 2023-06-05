#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:58:55 2022

@author: vittoriogiammarino
"""

import os
import string
import random
import tempfile
import pybullet as p
import numpy as np

def fill_template(template, replace):
    """Read a file and replace key strings."""
    full_template_path = template
    with open(full_template_path, 'r') as file:
        fdata = file.read()
    for field in replace:
        for i in range(len(replace[field])):
            fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    tmpdir = tempfile.gettempdir()
    template_filename = os.path.split(template)[-1]
    fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
    with open(fname, 'w') as file:
        file.write(fdata)
    return fname

def generate_boxes(box_urdf, initial_position, orientation):
    box = p.loadURDF(box_urdf, initial_position, orientation, useFixedBase=0)
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    p.changeVisualShape(box, -1, rgbaColor=color)
    return box

def generate_stack_of_boxes(height, width, length, box_urdf, initial_position, 
                            orientation, margin, noise = np.random.random((1))):
    list_of_box_objects = []
    for k in range(height):
        for i in range(width):
            for j in range(length):
                dim_noise = margin/2*noise
                cartesian_position = (initial_position[0]+0.25*i+dim_noise+margin, initial_position[1]-0.25*j-margin-dim_noise, initial_position[2]+0.25*k+margin+dim_noise)
                box = generate_boxes(box_urdf, cartesian_position, orientation)
                list_of_box_objects.append(box)
                
    return list_of_box_objects

def generate_surfaces_for_single_box(surface_urdf, surface_size, initial_position, orientation):
    
    surfaces_ID = []
    
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    
    #front box surface
    front_position = (initial_position[0]-surface_size[2]/2, 
                      initial_position[1], 
                      initial_position[2])
    front_orientation = p.getQuaternionFromEuler(orientation) 
    front = p.loadURDF(surface_urdf, front_position, front_orientation, useFixedBase=1) 
    p.changeVisualShape(front, -1, rgbaColor=color)
    surfaces_ID.append(front)
    
    #top box surface
    top_position = (initial_position[0], 
                    initial_position[1], 
                    initial_position[2]+surface_size[2]/2)
    top_orientation = p.getQuaternionFromEuler((orientation[0], orientation[1]+np.pi/2, orientation[2]))
    top = p.loadURDF(surface_urdf, top_position, top_orientation, useFixedBase=1) 
    p.changeVisualShape(top, -1, rgbaColor=color)
    surfaces_ID.append(top)
    
    #left-side box surface
    left_side_position = (initial_position[0], 
                          initial_position[1]-surface_size[1]/2, 
                          initial_position[2])
    left_side_orientation = p.getQuaternionFromEuler((orientation[0], orientation[1], orientation[2]+np.pi/2))
    left_side = p.loadURDF(surface_urdf, left_side_position, left_side_orientation, useFixedBase=1) 
    p.changeVisualShape(left_side, -1, rgbaColor=color)
    surfaces_ID.append(left_side)
    
    #right-side box surface
    right_side_position = (initial_position[0], 
                          initial_position[1]+surface_size[1]/2, 
                          initial_position[2])
    right_side_orientation = p.getQuaternionFromEuler((orientation[0], orientation[1], orientation[2]-np.pi/2))
    right_side = p.loadURDF(surface_urdf, right_side_position, right_side_orientation, useFixedBase=1) 
    p.changeVisualShape(right_side, -1, rgbaColor=color)
    surfaces_ID.append(right_side)
    
    #back box surface
    back_position = (initial_position[0]+surface_size[2]/2, 
                          initial_position[1], 
                          initial_position[2])
    back_orientation = p.getQuaternionFromEuler((orientation[0], orientation[1], orientation[2]))
    back = p.loadURDF(surface_urdf, back_position, back_orientation, useFixedBase=1) 
    p.changeVisualShape(back, -1, rgbaColor=color)
    surfaces_ID.append(back)
    
    #bottom box surface
    bottom_position = (initial_position[0], 
                       initial_position[1], 
                       initial_position[2]-surface_size[2]/2)
    bottom_orientation = p.getQuaternionFromEuler((orientation[0], orientation[1]-np.pi/2, orientation[2]))
    bottom = p.loadURDF(surface_urdf, bottom_position, bottom_orientation, useFixedBase=1) 
    p.changeVisualShape(bottom, -1, rgbaColor=color)
    surfaces_ID.append(bottom)
    
    return surfaces_ID

def generate_single_surface(surface_urdf, surface_size, initial_position, orientation):
    surfaces_ID = []
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    
    #front box surface
    front_position = (initial_position[0]-surface_size[2]/2, 
                      initial_position[1], 
                      initial_position[2])
    front_orientation = p.getQuaternionFromEuler(orientation) 
    front = p.loadURDF(surface_urdf, front_position, front_orientation, useFixedBase=1) 
    p.changeVisualShape(front, -1, rgbaColor=color)
    surfaces_ID.append(front)
    
    return surfaces_ID
    
    
    
    
    
    
    
    
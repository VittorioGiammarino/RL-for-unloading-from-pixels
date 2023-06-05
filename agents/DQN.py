#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:17:52 2022

@author: vittoriogiammarino
"""

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import autograd
from torch import distributions as torchd

import utils_folder.utils as utils

from models.resnet import Encoder as encoder_net
from models.resnet import Decoder
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical

class Encoder(nn.Module):
    def __init__(self, input_shape, device, from_segm):
        super().__init__()

        self.repr_dim = 512*20*20
        
        self.device = device
        self.from_segm = from_segm
        
        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        self.net = encoder_net(self.in_shape[2])
        self.apply(utils.init_xavier_weights)

    def forward(self, input_img):
        if len(input_img.shape)==3:
            input_img = self.resize_input_img(input_img)
        elif len(input_img.shape)==4:
            input_img = self.process_img(input_img)

        in_tens = torch.split(input_img, 1, dim=0)
        h = ()
        
        for x in in_tens:
            h += (self.net(x),)

        h = torch.cat(h, dim=0)
        return h
        
    def get_feature_vector(self, h):
        return h.reshape(h.shape[0], -1)

    def process_img(self, input_img):
        img = self.normalize(input_img)
        return img
        
    def normalize(self, input_img):
        img = (input_img / 255) - 0.5
        return img
    
    def resize_input_img(self, input_img):
        in_data = np.pad(input_img, self.padding, mode='constant')
        in_data_processed = self.process_img(in_data)
        in_shape = (1,) + in_data_processed.shape
        in_data_processed = in_data_processed.reshape(in_shape)
        in_tens = torch.tensor(in_data_processed, dtype=torch.float32).to(self.device)
        return in_tens  
    
class Critic(nn.Module):
    def __init__(self, input_shape, output_channels):
        super().__init__()

        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        
        self.critic_1 = Decoder(output_channels)
        self.critic_2 = Decoder(output_channels)
        self.apply(utils.init_xavier_weights)
        
    def forward(self, h):
        
        logits_critic_1 = self.critic_1(h)
        logits_critic_2 = self.critic_2(h)
        
        output_critic_1 = self.pad_rearrange(logits_critic_1)
        output_critic_2 = self.pad_rearrange(logits_critic_2)
        
        return output_critic_1, output_critic_2
            
    def pad_rearrange(self, logits):
        c0 = self.padding[:2, 0]
        c1 = c0 + self.in_shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        output = Rearrange('b h w c -> b (h w c)')(logits)
        
        return output
        
class DQN_Agent:
    def __init__(self, input_shape, workspace, device, use_tb, critic_target_tau, update_every_steps, decoder_nc, learning_rate, 
                exploration_rate, num_expl_steps, from_segm=False, safety_mask=False):
        
        self.device = device
        self.use_tb = use_tb
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.from_segm = from_segm
        self.workspace = workspace
        self.exploration_rate = exploration_rate
        self.num_expl_steps = num_expl_steps
        self.safety_mask = safety_mask
        
        output_channels = decoder_nc
        self.encoder = Encoder(input_shape, device, from_segm).to(self.device)

        print("Training using only exogenous reward")
        
        self.critic = Critic(input_shape, output_channels).to(self.device)
        self.critic_target = Critic(input_shape, output_channels).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #optimizers
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        self.train()
        self.critic_target.train()
        
    def train(self, training=True):
        self.training = training 
        self.critic.train(training)

    def check_in_workspace(self, cartesian_position_box):
        in_workspace = False
        x = cartesian_position_box[0]
        y = cartesian_position_box[1]
        z = cartesian_position_box[2]
        
        if x>=self.workspace[0][0] and x<=self.workspace[0][1]:
            if y>=self.workspace[1][0] and y<=self.workspace[1][1]:
                if z>=self.workspace[2][0] and z<=self.workspace[2][1]:
                    in_workspace = True
                    
        return in_workspace

    def compute_safety_mask(self, target_V, xyz, input_image_shape):

        num_pixels = target_V.shape[1]
        mask = torch.ones([1, num_pixels], dtype=torch.float).to(self.device)
        valid = torch.zeros([1, num_pixels], dtype=torch.int).to(self.device)

        for i in range(num_pixels):
            matrix_pixels = target_V.reshape(input_image_shape)
            index_reshaped = np.unravel_index(i, shape=matrix_pixels.shape)
            pick = index_reshaped[:2]
            pick_y = int(pick[0])
            pick_x = int(pick[1]) 
            pick_position = xyz[pick_y, pick_x]

            in_workspace = self.check_in_workspace(pick_position)

            if in_workspace:
                valid[0,i] = 1
                mask[0,i] = 200 #100
            else:
                mask[0,i] = 0 #-100 is also an option

        return mask, valid

    def act(self, input_image, xyz, step, eval_mode=True):

        obs = self.encoder(input_image)
        input_image_shape = input_image.shape[:2]
        target_Q1, target_Q2 = self.critic_target(obs)         
        target_V = torch.min(target_Q1, target_Q2)
                    
        if eval_mode:
            action, picking_y, picking_x = self.act_eval(target_V, xyz, input_image_shape)

        elif step <= self.num_expl_steps:
            action, picking_y, picking_x = self.explore(target_V, xyz, input_image_shape)
            
        else:
            action, picking_y, picking_x = self.act_training(target_V, xyz, input_image_shape)
            
        return action, picking_y, picking_x

    def act_eval(self, target_V, xyz, input_image_shape):

        if self.safety_mask:
            mask, _ = self.compute_safety_mask(target_V, xyz, input_image_shape)
            pick_conf = target_V + mask

        else:
            pick_conf = target_V

        pick_conf = pick_conf.detach().cpu().numpy()
        pick_conf = np.float32(pick_conf).reshape(input_image_shape)

        #this one works cause we are processing a single image at the time during eval mode        
        action = np.argmax(pick_conf) 
        action = np.unravel_index(action, shape=pick_conf.shape)
        
        picking_pixel = action[:2]
        picking_y = picking_pixel[0]
        picking_x = picking_pixel[1]   

        return action, picking_y, picking_x

    def explore(self, target_V, xyz, input_image_shape):
        print("Explore")

        if self.safety_mask:
            _, valid = self.compute_safety_mask(target_V, xyz, input_image_shape)
            try:
                valid = valid.detach().cpu().numpy()
                action_numpy = np.random.choice(valid.nonzero()[1], size=1)
            except:
                num_pixels = target_V.shape[1]
                action_numpy = np.random.randint(num_pixels, size=(1,))

        else:
            num_pixels = target_V.shape[1]
            action_numpy = np.random.randint(num_pixels, size=(1,))

        action_reshaped = np.unravel_index(action_numpy, shape=input_image_shape)
        
        picking_pixel = action_reshaped[:2]
        picking_y = int(picking_pixel[0])
        picking_x = int(picking_pixel[1])

        action = action_numpy

        return action, picking_y, picking_x

    def act_training(self, target_V, xyz, input_image_shape):

        pick_conf = torch.clone(target_V)
        expl_rv = np.random.rand()

        if self.safety_mask:
            mask, valid = self.compute_safety_mask(target_V, xyz, input_image_shape)

            if expl_rv <= self.exploration_rate:
                print("Explore")
                try:
                    valid = valid.detach().cpu().numpy()
                    action_numpy = np.random.choice(valid.nonzero()[1], size=1)
                except:
                    num_pixels = target_V.shape[1]
                    action_numpy = np.random.randint(num_pixels, size=(1,))

            else:
                pick_conf = target_V + mask
                pi = Categorical(logits = pick_conf)
                action = pi.sample()
                action_numpy = action.detach().cpu().numpy()
        else:

            if expl_rv <= self.exploration_rate:
                print("Explore")
                num_pixels = target_V.shape[1]
                action_numpy = np.random.randint(num_pixels, size=(1,))

            else:
                pi = Categorical(logits=pick_conf)
                action = pi.sample()
                action_numpy = action.detach().cpu().numpy()

        action_reshaped = np.unravel_index(action_numpy, shape=input_image_shape)
        
        picking_pixel = action_reshaped[:2]
        picking_y = int(picking_pixel[0])
        picking_x = int(picking_pixel[1])

        action = action_numpy

        return action, picking_y, picking_x
    
    def act_batch(self, input_image):        
        target_Q1, target_Q2 = self.critic_target(input_image)
        target_V = torch.min(target_Q1, target_Q2) 
        pick_conf = target_V
        argmax = torch.argmax(pick_conf, dim=-1)
        pi = Categorical(logits = pick_conf)
            
        return target_V, argmax, pi
        
    def update_critic(self, obs, action, reward, discount, next_obs):
        metrics = dict()
        
        with torch.no_grad():
            target_V, next_pick, pi = self.act_batch(next_obs)
            target_V_pick = target_V.gather(1, next_pick.reshape(-1,1))
            target_Q = reward + (discount*target_V_pick)
            
        Q1, Q2 = self.critic.forward(obs)
        
        pick = action[:,0]
        Q1_picked = Q1.gather(1, pick.reshape(-1,1).long())
        Q2_picked = Q2.gather(1, pick.reshape(-1,1).long())
        
        critic_loss = F.mse_loss(Q1_picked, target_Q) + F.mse_loss(Q2_picked, target_Q) 
                        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1_picked.mean().item()
            metrics['critic_q2'] = Q2_picked.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['policy_entropy'] = pi.entropy().mean().item()
                        
        self.optimizer_encoder.zero_grad(set_to_none=True)
        self.optimizer_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optimizer_critic.step()
        self.optimizer_encoder.step()
        
        return metrics  
        
    def update(self, replay_iter, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        obs = obs.float()
        next_obs = next_obs.float()    

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs))

        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        return metrics
            
            
            
            
                
            
            
            
                
        
        
        
        
        
        
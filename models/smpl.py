"""
Parts of the code are taken from https://github.com/microsoft/MeshTransformer
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np

import utils.config as config
from utils.geometry import rodrigues, rot6d_to_rotmat



# SMPL model
class SMPL(nn.Module):
    def __init__(self, gender=None):
        super(SMPL, self).__init__()
        if gender == 0:
            _gender = 'female'
        elif gender == 1:
            _gender = 'male'
        else:
            _gender = 'neutral'
            
        smpl_model = torch.load(config.J_REGRESSOR_DIR[_gender])
        self.register_buffer('J_regressor', smpl_model['J_regressor'])
        self.register_buffer('weights', smpl_model['weights'])
        self.register_buffer('posedirs', smpl_model['posedirs'])
        self.register_buffer('v_template', smpl_model['v_template'])
        self.register_buffer('shapedirs', smpl_model['shapedirs'])
        self.register_buffer('parent', smpl_model['parent'])
        
    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72) or (bs,144)
        elif pose.ndimension() == 2:
            if pose.shape[-1] == 72:
                pose_cube = pose.view(-1, 3) # (batch_size * 24, 1, 3)
                R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
                R = R.view(batch_size, 24, 3, 3)
            elif pose.shape[-1] == 144:
                R = rot6d_to_rotmat(pose).view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:,1:,:] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1,0,2,3).contiguous().view(24,-1)).view(6890, batch_size, 4, 4).transpose(0,1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v
    
    
    
# SMPL model with static pose
class SMPLStaticPose(SMPL):
    def __init__(self, gender=None):
        super(SMPLStaticPose, self).__init__(gender)
        mean_params = np.load(config.SMPL_MEAN_PARAMS_DIR)
        mean_pose = torch.from_numpy(mean_params['pose']).unsqueeze(0)
        mean_pose = rot6d_to_rotmat(mean_pose).view(1, 24, 3, 3)
        self.register_buffer('mean_pose', mean_pose)
        
    def forward(self, beta):
        device = beta.device
        batch_size = beta.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        
        # Mean pose fixed.
        R = self.mean_pose.expand(batch_size, -1, -1, -1)
        
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:,1:,:] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1,0,2,3).contiguous().view(24,-1)).view(6890, batch_size, 4, 4).transpose(0,1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v
    
    
    
# SMPL model to fit pose with shape fixed
class SMPLStaticShape(SMPL):
    def __init__(self, n_batch, pose_dim=144, gender=None):
        super(SMPLStaticShape, self).__init__(gender)
        shape = torch.zeros(n_batch, 10)
        self.register_buffer('shape', shape)
        self.pose = nn.Parameter(torch.zeros(n_batch, pose_dim))
        self.n_batch = n_batch
        
        mean_params = np.load(config.SMPL_MEAN_PARAMS_DIR)
        mean_pose = torch.from_numpy(mean_params['pose']).unsqueeze(0).expand(n_batch, -1)
        mean_shape = torch.from_numpy(mean_params['shape']).unsqueeze(0).expand(n_batch, -1)
        self.register_buffer('mean_pose', mean_pose)
        self.register_buffer('mean_shape', mean_shape)
        
    def set_init_params(self, shape=None, pose=None):
        with torch.no_grad():
            _shape = shape if shape != None else self.mean_shape
            if len(_shape) < self.n_batch:
                _shape = torch.cat([_shape, self.mean_shape[len(_shape):]], dim=0)
            _pose = pose if pose != None else self.mean_pose
            if len(_pose) < self.n_batch:
                _pose = torch.cat([_pose, self.mean_pose[len(_pose):]], dim=0)
            self.shape.copy_(_shape)
            self.pose.copy_(_pose)
            
    def forward(self):
        return super(SMPLStaticShape, self).forward(self.pose, self.shape)
    
    
    
# Blend SMPL male and female model.
def blend_smpl_gender(smpl_gender_models, gender, shape):
    """
    smpl_gender_models = [smpl_female_model, smpl_male_model]
    gender : a collection of gender for each shape parameter
    shape : shape parameters
    """
    device = shape.device
    female = (gender == 0)
    male = (gender == 1)
    
    vertices_female = smpl_gender_models[0](shape[female]).reshape(-1, 20670) if female.any() else torch.empty(0, 20670, device=device)
    vertices_male = smpl_gender_models[1](shape[male]).reshape(-1, 20670) if male.any() else torch.empty(0, 20670, device=device)
    
    row_op_female = torch.zeros(len(gender), len(vertices_female), device=device)
    row_op_male = torch.zeros(len(gender), len(vertices_male), device=device)
    
    row_op_female[female] = torch.eye(len(vertices_female), device=device)
    row_op_male[male] = torch.eye(len(vertices_male), device=device)
    
    blended_vertices = torch.matmul(row_op_female, vertices_female) + torch.matmul(row_op_male, vertices_male)
    
    return blended_vertices.view(-1, 6890, 3)
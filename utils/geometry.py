"""
Parts of the code are taken from https://github.com/nkolot/SPIN,
https://github.com/microsoft/MeshTransformer,
and https://github.com/gulvarol/surreal
"""

import transforms3d
import numpy as np
import torch
import torch.nn.functional as F


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)

# Get axis-angle representation of the global rotation from SURREAL dataset.
def get_global_rotation(z_rot, pelvis_rot):
    angle = np.linalg.norm(pelvis_rot)
    pelvis_R = transforms3d.axangles.axangle2mat(pelvis_rot / angle, angle)
    
    z_cos, z_sin = np.cos(z_rot), np.sin(z_rot)
    z_R = np.array([[z_cos, -z_sin, 0], [z_sin, z_cos, 0], [0, 0, 1]])
    
    global_R = np.dot(z_R, pelvis_R)
    global_axis, global_angle = transforms3d.axangles.mat2axangle(np.dot(R90, global_R))
    return global_axis * global_angle


# Get rotation matrices from the data augmentation to transform global rotations.
def get_rotation_matrix(angle):
    """
    angle : (n_batch), type=ndarray, angles from the data augmentation
    """
    _angle = angle * (np.pi / 180)
    x_cos, x_sin = np.cos(_angle), np.sin(_angle)
    x_R = np.zeros((len(_angle), 3, 3))
    x_R[:, 0, 0] = 1
    
    # (-angle) with x-axis
    x_R[:, 1, 1] = x_cos
    x_R[:, 2, 2] = x_cos
    x_R[:, 1, 2] = x_sin
    x_R[:, 2, 1] = -x_sin
    
    return x_R


R_pre_reflection = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
R_post_reflection = torch.Tensor([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])

# Align rotation matrices from other dataset to those from SURREAL dataset.
def align_rotation(rotation_R):
    """
    rotation_R : (n_batch, 3, 3), type=torch.Tensor, rotation matrices from dataset
    """
    return torch.matmul(R_post_reflection, torch.matmul(rotation_R, R_pre_reflection))
"""
Parts of the code are taken from https://github.com/nkolot/SPIN
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

import utils.config as config


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=1000, img_res=224):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        
        self.faces = np.load(config.SMPL_FACES_DIR)

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image=0, gender=None, color='blue'):
        if color == 'blue':
            color_factor = (0.8, 0.3, 0.3, 1.0)
        elif color == 'red':
            color_factor = (0.3, 0.3, 0.8, 1.0)
        else:
            color_factor = (color[2], color[1], color[0], 1.0 if len(color)<4 else color[3])  # BGR order
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=color_factor)

        camera_translation[0] *= -1.
        
        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
#         output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        output_img = np.concatenate([color, np.ones((color.shape[0], color.shape[1], 1)) * color_factor[3]], axis=-1) * valid_mask\
                     + (1 - valid_mask) * image
        return output_img

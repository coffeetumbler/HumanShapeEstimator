"""
Parts of the code are taken from https://github.com/nkolot/SPIN
"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import json
# import scipy.misc
from PIL import Image

from utils.config import IMG_NORM_MEAN, IMG_NORM_STD



def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def get_batch_transform(center, scale, res):
    """Generate multiple transformation matrices."""
    n_batch = len(scale)
    
    h = 200 * scale
    t = np.zeros((n_batch, 3, 3))
    t[:, 0, 0] = float(res[1]) / h
    t[:, 1, 1] = float(res[0]) / h
    t[:, 0, 2] = -float(res[1] * center[0]) / h + (.5 * res[1])
    t[:, 1, 2] = -float(res[0] * center[1]) / h + (.5 * res[0])
    t[:, 2, 2] = 1
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1


def batch_transform(pt, center, scale, res):
    """Transform pixel location to different reference."""
    t = get_batch_transform(center, scale, res)
    t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:,:2].astype(int)+1


def imrotate(img, angle):
    h, w = img.shape[:2]
    center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=0)
    return rotated


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype='uint8')

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if rot != 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot)
        new_img = imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    
    # imresize function is deprecated.
    # new_img = scipy.misc.imresize(new_img, res)
    new_img = np.array(Image.fromarray(new_img).resize(res))
    
    return new_img


def crop_batch(img, center, scale, res, rot):
    """Crop multiple images according to the supplied bounding box.
       Assume all images have the same center and the rescale factor."""
    # Upper left point
    ul = batch_transform([1, 1], center, scale, res)-1
    # Bottom right point
    br = batch_transform([res[0]+1,res[1]+1], center, scale, res)-1
    
    # Padding so that when rotated proper amount of context is included
    pad = (np.linalg.norm(br - ul, axis=1) / 2 - (br[:,1] - ul[:,1]) / 2).astype(int)
    ul[rot != 0] -= pad[rot != 0][:, np.newaxis]
    br[rot != 0] += pad[rot != 0][:, np.newaxis]

    new_img = []
    for i in range(len(rot)):
        _img = img[i]
        
        new_shape = [br[i, 1] - ul[i, 1], br[i, 0] - ul[i, 0]]
        if len(_img.shape) > 2:
            new_shape += [_img.shape[2]]
        _new_img = np.zeros(new_shape, dtype='uint8')

        # Range to fill new array
        new_x = max(0, -ul[i, 0]), min(br[i, 0], len(_img[0])) - ul[i, 0]
        new_y = max(0, -ul[i, 1]), min(br[i, 1], len(_img)) - ul[i, 1]
        # Range to sample from original image
        old_x = max(0, ul[i, 0]), min(len(_img[0]), br[i, 0])
        old_y = max(0, ul[i, 1]), min(len(_img), br[i, 1])
        _new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = _img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        if rot[i] != 0:
            # Remove padding
            # new_img = scipy.misc.imrotate(new_img, rot)
            _new_img = imrotate(_new_img, rot[i])
            _new_img = _new_img[pad[i]:-pad[i], pad[i]:-pad[i]]

        # imresize function is deprecated.
        # new_img = scipy.misc.imresize(new_img, res)
        new_img.append(np.array(Image.fromarray(_new_img).resize(res)))
    
    return np.array(new_img)


def flip_img(img, flip):
    """Flip rgb images or masks.
    channels come first, e.g. (n,3,256,256).
    """
    img[flip < 1] = np.flip(img[flip < 1], axis=-1)
    return img



def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file=None, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
    img = cv2.imread(img_file).copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        center, scale = bbox_from_json(bbox_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img
"""
Parts of the code are taken from https://github.com/microsoft/MeshTransformer
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle, cv2, shutil, json
import numpy as np
import torch
import scipy.io

import utils.config as config
from utils.geometry import get_global_rotation, rodrigues



if __name__ == '__main__':
    # Preprocess SMPL models.
    model_path = 'data/smpl/'
    models = {'male' : 'basicModel_m_lbs_10_207_0_v1.0.0.pkl',
              'female' : 'basicModel_f_lbs_10_207_0_v1.0.0.pkl',
              'neutral' : 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'}

    for key, value in models.items():
        with open(model_path + value, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            smpl_model = u.load()

        J_regressor = smpl_model['J_regressor'].tocoo()

        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]

        item = {}
        item['J_regressor'] = torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense()

        item['weights'] = torch.FloatTensor(smpl_model['weights'])
        item['posedirs'] = torch.FloatTensor(smpl_model['posedirs'])
        item['v_template'] = torch.FloatTensor(smpl_model['v_template'])
        item['shapedirs'] = torch.FloatTensor(np.array(smpl_model['shapedirs']))
        kintree_table = torch.from_numpy(smpl_model['kintree_table'].astype(np.int64))
        id_to_col = {kintree_table[1, i].item(): i for i in range(kintree_table.shape[1])}
        item['parent'] = torch.LongTensor([id_to_col[kintree_table[0, it].item()] for it in range(1, kintree_table.shape[1])])

        faces = smpl_model['f'].astype(np.int64)

        torch.save(item, 'data/J_regressor_'+key+'.pt')
        if key == 'neutral':
            np.save('data/smpl_faces_neutral', faces)


    # Preprocess 3DPW dataset.
    if os.path.exists('data/3DPW_infos/'):
        shutil.move('data/3DPW_infos/', config.BASE_DATASET_DIR + '3DPW/infos/')


    # Preprocess SURREAL dataset.
    base_dir = config.BASE_DATASET_DIR + 'SURREAL/data/cmu/'

    for set_type in ['val', 'test', 'train']:
        if os.path.exists(base_dir+set_type):
            if os.path.exists('data/SURREAL_infos/'+set_type):
                shutil.move('data/SURREAL_infos/' + set_type + '/data_infos', base_dir + set_type)
                shutil.move('data/SURREAL_infos/' + set_type + '/person_infos', base_dir + set_type)
                os.rmdir('data/SURREAL_infos/'+set_type)

            for i in range(3):
                file_path = base_dir + set_type + '/run{}/'.format(i)
                subjects = os.listdir(file_path)

                for subject in subjects:
                    subject_path = file_path + subject + '/'
                    files = os.listdir(subject_path)

                    for file in files:
                        # Split vidoes into images.
                        if 'mp4' in file:
                            video_path = subject_path + file
                            seg_path = video_path[:-4] + "_segm.mat"
                            frames_path = video_path[:-4] + "_frames/"
                            os.mkdir(frames_path)

                            video_capture = cv2.VideoCapture(video_path)
                            success, image = video_capture.read()  # image capture

                            seg_mat = scipy.io.loadmat(seg_path)  # segmentation info

                            if image.shape == (240, 320, 3):
                                # Get maximum number of pixels of human among all frames.
                                seg_max_pixel = 0
                                for seg in seg_mat.values():
                                    if isinstance(seg, np.ndarray):
                                        seg_max_pixel = np.max((np.sum(seg[8:232, 48:272] > 0), seg_max_pixel))
                                threshold = int(seg_max_pixel * 0.4)  # threshold number of human pixels : 40% of maximum pixels

                                frame = 1
                                while success and frame < 500:
                                    seg = seg_mat['segm_{}'.format(frame)][8:232, 48:272]

                                    # Images with more than 12 body parts and pixels with more than threshold are saved.
                                    if len(np.unique(seg)) > 13 and np.sum(seg > 0) > threshold:
                                        image_name = "{}.jpg".format(frame)
                                        cv2.imwrite(frames_path + image_name, image[8:232, 48:272])

                                    success, image = video_capture.read()
                                    frame += 1

                            else:
                                print('Video is not in proper size.')
                                print('Video path :', video_path)

                            # Extract pose and camera parameters.
                            info_path = video_path[:-4] + '_info.mat'
                            folder_path = info_path[:-8] + 'params/'
                            os.mkdir(folder_path)

                            info_mat = scipy.io.loadmat(info_path)
                            shape = info_mat['shape'][:, 0]  # 10 dim
                            pose = np.transpose(info_mat['pose'])  # T*72 dim
                            cam = np.append(info_mat['camDist'], info_mat['camLoc'])  # 1+3 dim
                            zrot = info_mat['zrot'][:, 0]  # T dim

                            global_rotations = []
                            params = []
                            param_names = []

                            for frame_name in os.listdir(frames_path):
                                if '.jpg' in frame_name:
                                    frame_id = int(frame_name[:-4])
                                    _frame_id = frame_id - 1
                                    global_rotation = get_global_rotation(zrot[_frame_id], pose[_frame_id, :3])
                                    global_rotations.append(global_rotation)

                                    # 4(cam) + 1(zrot) + 3(global_rot) + 72(pose) + 10(shape) dim
                                    params.append(np.concatenate((cam, zrot[_frame_id], global_rotation, pose[_frame_id], shape), axis=None))
                                    param_names.append(folder_path+frame_name[:-4])

                            global_R = rodrigues(torch.Tensor(global_rotations)).view(-1, 9).contiguous()
                            for R, param, param_name in zip(global_R, params, param_names):
                                # 4(cam) + 1(zrot) + 3(global_rot(axis-angle)) + 9(global_rot(matrix))
                                #        + 72(pose) + 10(shape) dim
                                np.save(param_name, np.concatenate((param[:8], R.numpy(), param[8:]), axis=None))
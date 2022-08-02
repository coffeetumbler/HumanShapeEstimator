"""
Parts of the code are taken from https://github.com/nkolot/SPIN and https://github.com/microsoft/MeshTransformer
"""

import os, sys, cv2, copy
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms import Normalize

import utils.config as config
from utils.image_processing import crop, crop_batch, flip_img
from utils.geometry import rodrigues, get_rotation_matrix, align_rotation



# Simple dataset for simple network
class SimpleDataset(Dataset):
    def __init__(self, dataset='surreal', set_type='train', augmentation='full', use_smpl_params=False):
        super(SimpleDataset, self).__init__()
        
        self.dataset_dir = config.DATASET_DIR[dataset][set_type]
        self.is_train = True if set_type == 'train' else False
        self.augmentation = augmentation if self.is_train else 'none'
        self.use_smpl_params = use_smpl_params
        
        with np.load(self.dataset_dir + config.DATASET_INFO['simple'], allow_pickle=True) as data_infos:
            self.gender = data_infos['gender'].astype(int)
            self.shape = data_infos['shape']
            self.file_dir = data_infos['data']
            self.person_id = data_infos['person_id'].astype(int)
            self.n_frame = data_infos['n_frame'].astype(int)
            self.length = len(data_infos['gender'])
            
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
        
    def get_n_frames(self, copy=False):
        if copy:
            return self.n_frame.copy()
        else:
            return self.n_frame
        
        
    def augm_params(self, n):
        """Get n-number of augmentation parameters."""        
        # We flip with probability 1/2
        flip = np.random.randint(2, size=n)
        
        # Each channel is multiplied with a number 
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(n,3))

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor], with probability 0.4
        rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                         np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn(n)*config.IMG_ROTATION_FACTOR))
        rot[np.random.uniform(size=n) < 0.6] = 0

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn(n)*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
        
        return flip, pn, rot, sc
    
    def rgb_processing(self, rgb_img, scale, rot, flip, pn):
        """Process rgb images and do augmentation."""
        rgb_img = crop_batch(rgb_img, config.IMG_CENTER, scale, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        # flip the image 
        rgb_img = flip_img(rgb_img, flip)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0
    
    def rgb_noising(self, rgb_img):
        """Noise rgb images."""
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(rgb_img.shape[0],3))

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0

    def nonflip_augm(self, rgb_img):
        """
        Non-flip augmentation of rgb images
        Output augmentated images and rotation matrix.
        """
        n = rgb_img.shape[0]
        
        # noise, rotation, scale factors
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(n,3))
        rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                         np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn(n)*config.IMG_ROTATION_FACTOR))
        rot[np.random.uniform(size=n) < 0.6] = 0
        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn(n)*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
        
        rgb_img = crop_batch(rgb_img, config.IMG_CENTER, sc, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0, get_rotation_matrix(rot)
    
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        item = {}
        img_list = []
        
        series_dir = self.dataset_dir + self.file_dir[index[0]] + '_frames/'
        frame_indices = sorted(os.listdir(series_dir))
            
        for frame_index in index[1]:
            img_list.append(cv2.imread(series_dir + frame_indices[frame_index]).copy())
        
        # Data augmentation
        if self.augmentation == 'full':
            flip, pn, rot, sc = self.augm_params(len(img_list))
            img_list = self.rgb_processing(np.array(img_list), sc, rot, flip, pn)
        elif self.augmentation == 'noise':
            img_list = self.rgb_noising(np.array(img_list))
        elif self.augmentation == 'nonflip':
            img_list, rotation_R = self.nonflip_augm(np.array(img_list))
            item['rotation'] = torch.from_numpy(rotation_R).float()
        else:
            img_list = np.transpose(np.array(img_list), (0, 3, 1, 2)).astype(np.float32) / 255.
            
        item['img'] = self.normalize_img(torch.from_numpy(img_list))
        
        item['series_index'] = index[0]
        item['frame_index'] = index[1]
        
        item['gender'] = self.gender[index[0]]
        item['shape'] = torch.from_numpy(self.shape[index[0]].copy())
        item['person_id'] = self.person_id[index[0]]
        
        if self.use_smpl_params:
            param_dir = series_dir[:-7] + 'params/'
            param_names = sorted(os.listdir(param_dir))
            param_list = []
            for frame_index in index[1]:
                param_list.append(np.load(param_dir + param_names[frame_index]))
            item['smpl'] = torch.Tensor(param_list)
        
        return item
    
    
    
# Sampler for simple dataset
class SimpleDatasetSampler(Sampler):
    def __init__(self, data_source, test_all, replacement, frames_in_series):
        """
        data_source : SimpleDataset
        test_all : whether to test all images in test dataset
        replacement : allowance to duplicated data
        frames_in_series : number of selected frames in an image series
        """
        self.n_frames = data_source.get_n_frames()
        self.test_sample = False if test_all else True
        self.replacement = replacement
        self.frames_in_series = frames_in_series
        
        self.length = len(data_source) if self.test_sample else np.sum(self.n_frames)
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        if self.test_sample:
            if self.replacement:
                series_index = np.random.randint(self.length, size=self.length)
            else:
                series_index = np.random.permutation(self.length)
            frame_index = np.random.randint(np.zeros(self.frames_in_series), self.n_frames[series_index][:, np.newaxis])
            yield from list(zip(series_index, map(tuple, frame_index)))
        else:
            series_index = range(len(self.n_frames))
            frame_index = []
            for i in series_index:
                frame_index.extend([(i, (j,)) for j in range(self.n_frames[i])])
            yield from frame_index
        
        
        
# Collate function for simple dataset
def simple_dataset_collate_fn(batch):
    _batch = {}
    img_list = []
    shape_list = []
    frame_indices = []
    series_indices = []
    gender_list = []
    person_ids = []
    
    frames_in_series = len(batch[0]['frame_index'])

    for item in batch:
        img_list.append(item['img'])
        frame_indices.extend(item['frame_index'])
        series_indices.extend([item['series_index']] * frames_in_series)
        gender_list.extend([item['gender']] * frames_in_series)
        person_ids.extend([item['person_id']] * frames_in_series)
        shape_list.append(item['shape'].expand(frames_in_series, -1))
        
    _batch['img'] = torch.cat(img_list, dim=0)
    _batch['shape'] = torch.cat(shape_list, dim=0)
    _batch['frame_index'] = frame_indices
    _batch['series_index'] = series_indices
    _batch['gender'] = np.array(gender_list)
    _batch['person_id'] = person_ids
    
    if 'smpl' in batch[0].keys():
        _batch['smpl'] = torch.cat([item['smpl'] for item in batch], dim=0)
    if 'rotation' in batch[0].keys():
        _batch['rotation'] = torch.cat([item['rotation'] for item in batch], dim=0)
    
    return _batch



# Get a data loader for simple dataset.
def get_simple_dataset_loader(dataset='surreal',
                              set_type='train',
                              test_all=False,
                              augmentation='full',
                              use_smpl_params=False,
                              batch_size=32,
                              frames_in_series=1,
                              replacement=False,
                              pin_memory=True,
                              drop_last=True):
    
    simple_dataset = SimpleDataset(dataset=dataset, set_type=set_type, augmentation=augmentation, use_smpl_params=use_smpl_params)
    _test_all = True if test_all and set_type == 'test' else False
    simple_sampler = SimpleDatasetSampler(simple_dataset, test_all=_test_all, replacement=replacement, frames_in_series=frames_in_series)
    return DataLoader(simple_dataset, sampler=simple_sampler, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last, collate_fn=simple_dataset_collate_fn, num_workers=4)




# Shape-focused dataset for the shape estimator
class ShapeFocusedDataset(Dataset):
    def __init__(self, dataset='surreal', set_type='train', augmentation='full', use_smpl_params=False):
        super(ShapeFocusedDataset, self).__init__()
        
        self.dataset_dir = config.DATASET_DIR[dataset][set_type]
        self.is_train = True if set_type == 'train' else False
        self.augmentation = augmentation if self.is_train else 'none'
        if self.augmentation == 'full':
            self.augmentation = 0
        elif self.augmentation == 'noise':
            self.augmentation = 1
        elif self.augmentation == 'nonflip':
            self.augmentation = 2
        else:
            self.augmentation = -1
        self.use_smpl_params = use_smpl_params
        
        with np.load(self.dataset_dir + config.DATASET_INFO['shape_focused'], allow_pickle=True) as data_infos:
            self.gender = data_infos['gender'].astype(int)
            self.shape = data_infos['shape']
            self.file_dir = data_infos['data']
            self.n_person = data_infos['n_person'].astype(int)
            self.n_frame = data_infos['n_frame']
            self.length = len(data_infos['gender'])
            
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
        
    def get_data_item(self, item, copy=False):
        if copy:
            return copy.deepcopy(getattr(self, item))
        else:
            return getattr(self, item)
        
        
    def augm_params(self, n):
        """Get n-number of augmentation parameters."""        
        # We flip with probability 1/2
        flip = np.random.randint(2, size=n)
        
        # Each channel is multiplied with a number 
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(n,3))

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor], with probability 0.4
        rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                         np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn(n)*config.IMG_ROTATION_FACTOR))
        rot[np.random.uniform(size=n) < 0.6] = 0

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn(n)*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
	
        return flip, pn, rot, sc
    
    def rgb_processing(self, rgb_img, scale, rot, flip, pn):
        """Process rgb images and do augmentation."""
        rgb_img = crop_batch(rgb_img, config.IMG_CENTER, scale, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        # flip the image 
        rgb_img = flip_img(rgb_img, flip)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0
    
    def rgb_noising(self, rgb_img):
        """Noise rgb images."""
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(rgb_img.shape[0],3))

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0

    def nonflip_augm(self, rgb_img):
        """
        Non-flip augmentation of rgb images
        Output augmentated images and rotation matrix.
        """
        n = rgb_img.shape[0]
        
        # noise, rotation, scale factors
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=(n,3))
        rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                         np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn(n)*config.IMG_ROTATION_FACTOR))
        rot[np.random.uniform(size=n) < 0.6] = 0
        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn(n)*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
        
        rgb_img = crop_batch(rgb_img, config.IMG_CENTER, sc, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (0, 3, 1, 2))
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,:,np.newaxis,np.newaxis]))

        return rgb_img.astype(np.float32) / 255.0, get_rotation_matrix(rot)
    
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        item = {}
        img_list = []
        
        series_dir = self.file_dir[index[0]]
        for frame_index in index[1]:
            folder_name = self.dataset_dir + series_dir[frame_index[0]] + '_frames/'
            frame_name = sorted(os.listdir(folder_name))[frame_index[1]]
            img_list.append(cv2.imread(folder_name + frame_name).copy())
            
        # Data augmentation
        if self.augmentation == 0:
            flip, pn, rot, sc = self.augm_params(len(img_list))
            img_list = self.rgb_processing(np.array(img_list), sc, rot, flip, pn)
        elif self.augmentation == 1:
            img_list = self.rgb_noising(np.array(img_list))
        elif self.augmentation == 2:
            img_list, rotation_R = self.nonflip_augm(np.array(img_list))
            item['rotation'] = torch.from_numpy(rotation_R).float()
        else:
            img_list = np.transpose(np.array(img_list), (0, 3, 1, 2)).astype(np.float32) / 255.
            
        item['img'] = self.normalize_img(torch.from_numpy(img_list))

        item['person_id'] = index[0]
        item['frame_index'] = index[1]
        
        item['gender'] = self.gender[index[0]]
        item['shape'] = torch.from_numpy(self.shape[index[0]].copy())
        
        if self.use_smpl_params:
            param_list = []
            for frame_index in index[1]:
                folder_name = self.dataset_dir + series_dir[frame_index[0]] + '_params/'
                param_name = sorted(os.listdir(folder_name))[frame_index[1]]
                param_list.append(np.load(folder_name + param_name))
            item['smpl'] = torch.Tensor(param_list)
        
        return item
    

    
# Sampler for shape-focused dataset
class ShapeFocusedDatasetSampler(Sampler):
    def __init__(self,
                 data_source,
                 protocol=False,
                 batch_size=4,
                 replacement=False,
                 equal_n_seq_per_person=True,
                 n_seq_per_person=10,
                 shuffle_series=True,
                 static_n_frames=False,
                 mean_frames_in_seq=5,
                 max_frames_in_seq=16,
                 drop_last=False):
        """
        data_source : ShapeFocusedDataset
        protocol : whether to use dataset protocol(fixed indices)
        batch_size : batch size for people in one process
        replacement : allowance to duplicated data
        equal_n_seq_per_person : equal number of sequences per person
        n_seq_per_person : number of sequences per person, not used when equal_n_seq_per_person=False
        shuffle_series : shuffle series when choosing frames for a person
        static_n_frames : fix the number of frames in a sequence when True,
                          pick a number of frames from a Gaussian distribution when False
        mean_frames_in_seq : number of frames in a sequence when static_n_frames=True,
                             mean number of frames in a sequence when static_n_frames=False
        max_frames_in_seq : maximum number of frames in a sequence
        drop_last : whether to drop the last insufficient batch
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.protocol = protocol
        
        if protocol:
            with np.load(config.TEST_PROTOCOL_DIR['surreal'], allow_pickle=True) as fixed_indices:
                self.fixed_index = fixed_indices[str(mean_frames_in_seq)]
            self.length = len(self.fixed_index)
            
        else:
            self.replacement = replacement
            self.equal_n_seq_per_person = equal_n_seq_per_person
            self.n_seq_per_person = n_seq_per_person
            self.shuffle_series = shuffle_series
            self.static_n_frames = static_n_frames
            self.mean_frames_in_seq = mean_frames_in_seq
            self.max_frames_in_seq = max_frames_in_seq
            
            self.n_person = data_source.get_data_item('n_person')
            self.n_frames = data_source.get_data_item('n_frame')
        
            self.indices = []
            # Set indices frame-wise.
            if shuffle_series:
                for n_frames in self.n_frames:
                    frame_indices = []
                    for i_series, _n_frames in enumerate(n_frames):
                        frame_indices.extend(zip([i_series] * _n_frames, range(_n_frames)))
                    self.indices.append(frame_indices.copy())
            # Set indices series-wise.
            else:
                for _id, n_person in enumerate(self.n_person):
                    self.indices.extend(zip([_id] * n_person, range(n_person)))

            if equal_n_seq_per_person:
                self.person_id = list(range(len(data_source))) * n_seq_per_person
            else:
                self.person_id = []
                for _id, n_person in enumerate(self.n_person):
                    self.person_id.extend([_id] * n_person)

            if not static_n_frames:
                self.sigma = mean_frames_in_seq - 2 if mean_frames_in_seq > 2 else 1

            self.length = len(self.person_id)
        
        
    def __len__(self):
        return self.length
    
    
    def __iter__(self):
        if self.protocol:
            samples = map(tuple, self.fixed_index)
            
        else:
            if self.shuffle_series:
                if self.replacement:
                    person_index = np.random.choice(self.person_id, size=self.length)
                else:
                    person_index = np.random.permutation(self.person_id)

                if self.static_n_frames:
                    frame_index = [tuple(map(tuple, np.random.permutation(self.indices[i])[:self.mean_frames_in_seq]))\
                                   for i in person_index]
                else:
                    # Random sequence lengths
                    # Guarantee At least 16% probability for length 1.
                    n_frames = np.random.randn(self.length // self.batch_size + 1) * self.sigma + self.mean_frames_in_seq
                    n_frames = n_frames.astype(int)
                    n_frames[(n_frames < 1) | (n_frames > self.max_frames_in_seq)] = 1
                    n_frames = np.repeat(n_frames, self.batch_size)[:self.length]

                    frame_index = [tuple(map(tuple, np.random.permutation(self.indices[i])[:j]))\
                                   for i, j in zip(person_index, n_frames)]

                # yield from list(zip(person_index, frame_index))
                samples = zip(person_index, frame_index)

            else:
                if self.replacement:
                    person_index = np.random.choice(self.person_id, size=self.length)
                    series_index = [np.random.choice(self.n_person[i]) for i in person_index]
                elif self.equal_n_seq_per_person:
                    person_index = np.random.permutation(self.person_id)
                    series_index = [np.random.choice(self.n_person[i]) for i in person_index]
                else:
                    indices = np.random.permutation(self.indices)
                    person_index = [i[0] for i in indices]
                    series_index = [i[1] for i in indices]

                _n_frames = [self.n_frames[i][j] for i, j in zip(person_index, series_index)]
                if self.static_n_frames:
                    frame_index = np.random.randint(np.zeros(self.mean_frames_in_seq), np.array(_n_frames)[:, np.newaxis])
                else:
                    # Random sequence lengths
                    # Guarantee At least 16% probability for length 1.
                    n_frames = np.random.randn(self.length // self.batch_size + 1) * self.sigma + self.mean_frames_in_seq
                    n_frames = n_frames.astype(int)
                    n_frames[(n_frames < 1) | (n_frames > self.max_frames_in_seq)] = 1
                    n_frames = np.repeat(n_frames, self.batch_size)[:self.length]

                    frame_index = [np.random.randint(i, size=j) for i, j in zip(_n_frames, n_frames)]

                samples = zip(person_index, map(tuple, [[(i, k) for k in j] for i, j in zip(series_index, frame_index)]))
            
        batch = []
        id_count = 0
        for sample in samples:
            batch.append(sample)
            id_count += 1
            if id_count >= self.batch_size:
                yield batch
                id_count = 0
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
            
            
            
# Collate function for shape-focused dataset
def shape_focused_dataset_collate_fn(batch):
    _batch = {}
    img_list = []
    shape_list = []
    frame_indices = []
    gender_list = []
    person_ids = []
    seq_len_info = []

    for item in batch:
        frames_in_seq = len(item['frame_index'])
        img_list.append(item['img'])
        frame_indices.append(item['frame_index'])
        gender_list.append(item['gender'])
        person_ids.append(item['person_id'])
        shape_list.append(item['shape'])
        seq_len_info.append(frames_in_seq)
        
    _batch['img'] = torch.cat(img_list, dim=0)
    _batch['shape'] = torch.stack(shape_list)
    _batch['frame_index'] = frame_indices
    _batch['gender'] = np.array(gender_list)
    _batch['person_id'] = person_ids
    _batch['seq_len_info'] = np.array(seq_len_info, dtype=int)
    
    if 'smpl' in batch[0].keys():
        _batch['smpl'] = torch.cat([item['smpl'] for item in batch], dim=0)
    if 'rotation' in batch[0].keys():
        _batch['rotation'] = torch.cat([item['rotation'] for item in batch], dim=0)
    
    return _batch



# Get a data loader for shape-focused dataset.
def get_shape_focused_dataset_loader(dataset='surreal',
                                     set_type='train',
                                     protocol=False,
                                     augmentation='full',
                                     use_smpl_params=False,
                                     batch_size=4,
                                     replacement=False,
                                     equal_n_seq_per_person=True,
                                     n_seq_per_person=10,
                                     shuffle_series=True,
                                     static_n_frames=False,
                                     mean_frames_in_seq=5,
                                     max_frames_in_seq=16,
                                     drop_last=False,
                                     pin_memory=True):
    
    dataset = ShapeFocusedDataset(dataset=dataset, set_type=set_type, augmentation=augmentation, use_smpl_params=use_smpl_params)
    _protocol = True if protocol and set_type == 'test' else False
    sampler = ShapeFocusedDatasetSampler(dataset,
                                         protocol=_protocol,
                                         batch_size=batch_size,
                                         replacement=replacement,
                                         equal_n_seq_per_person=equal_n_seq_per_person,
                                         n_seq_per_person=n_seq_per_person,
                                         shuffle_series=shuffle_series,
                                         static_n_frames=static_n_frames,
                                         mean_frames_in_seq=mean_frames_in_seq,
                                         max_frames_in_seq=max_frames_in_seq,
                                         drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=pin_memory, collate_fn=shape_focused_dataset_collate_fn, num_workers=16)




# Dataset for test
class TestDataset(Dataset):
    def __init__(self, dataset='3dpw', model_type='simple', set_type='test', augmentation='full', use_smpl_params=False):
        super(TestDataset, self).__init__()
        
        self.dataset_dir = config.DATASET_DIR[dataset][set_type]
        self.base_dir = config.DATASET_DIR[dataset]['base']
        self.is_train = True if set_type == 'train' else False
        
        self.shape_focused = False if model_type == 'simple' else True
        
        self.augmentation = augmentation if self.is_train else 'none'
        if self.augmentation == 'full':
            self.augmentation = 0
        elif self.augmentation == 'nonflip':
            self.augmentation = 2
        else:
            self.augmentation = -1
        self.use_smpl_params = use_smpl_params
        
        with np.load(self.dataset_dir + config.DATASET_INFO['simple'], allow_pickle=True) as data_infos:
            gender = data_infos['gender']
            self.gender = np.array([1 if str(g) == 'm' else 0 for g in gender]).astype(int)  # 0 for female
            self.shape = data_infos['shape']
            self.imgname = data_infos['imgname']
            self.center = data_infos['center']
            self.scale = data_infos['scale']
            if use_smpl_params:
                self.pose = np.array(data_infos['pose'][:, :3])  # Global rotation only for now
            if self.shape_focused:  # Series index info
                self.series_index = data_infos['series_index']
            self.length = len(self.gender)
            
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)
        
    def get_data_item(self, item, copy=False):
        if copy:
            return copy.deepcopy(getattr(self, item))
        else:
            return getattr(self, item)
        
        
    def augm_params(self):
        """Get augmentation parameters."""        
        flip = np.random.randint(2)
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=3)

        if np.random.uniform() <= 0.6:
            rot = 0
        else:
            rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                             np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn()*config.IMG_ROTATION_FACTOR))

        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn()*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
        return flip, pn, rot, sc
    
    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb images and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        # flip the image
        if flip < 1:
            rgb_img = np.flip(rgb_img, axis=-1)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,np.newaxis,np.newaxis]))
        return rgb_img.astype(np.float32) / 255.0
        
    def rgb_centering(self, rgb_img, center, scale):
        """Process rgb image with center and scale information."""
        rgb_img = crop(rgb_img, center, config.IMG_DEFAULT_SCALE*scale, config.IMG_RES)
        return np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0
    
    def nonflip_augm(self, rgb_img, center, scale):
        """
        Non-flip augmentation of rgb image
        Output augmentated image and rotation matrix.
        """
        pn = np.random.uniform(*config.IMG_NOISE_RANGE, size=3)

        if np.random.uniform() <= 0.6:
            rot = 0
        else:
            rot = np.minimum(2*config.IMG_ROTATION_FACTOR,
                             np.maximum(-2*config.IMG_ROTATION_FACTOR, np.random.randn()*config.IMG_ROTATION_FACTOR))

        sc = np.minimum(1+config.IMG_SCALE_FACTOR, 
                        np.maximum(1-config.IMG_SCALE_FACTOR, np.random.randn()*config.IMG_SCALE_FACTOR+1)) * config.IMG_DEFAULT_SCALE
        
        rgb_img = crop(rgb_img, center, sc*scale, config.IMG_RES, rot)
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        rgb_img = np.minimum(255., np.maximum(0., rgb_img * pn[:,np.newaxis,np.newaxis]))
        return rgb_img.astype(np.float32) / 255.0, get_rotation_matrix(np.array([rot]))
    
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        item = {}
        if not self.shape_focused:
            index = (index,)
        
        img_list = []
        rotation_list = []
        for _index in index:
            scale = self.scale[_index].copy()
            center = self.center[_index].copy()

            frame_name = os.path.join(self.base_dir, self.imgname[_index])
            img = cv2.imread(frame_name).copy().astype(np.float32)

            # Data augmentation
            if self.augmentation == 0:
                flip, pn, rot, sc = self.augm_params()
                img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
            elif self.augmentation == 2:
                img, rotation_R = self.nonflip_augm(img, center, scale)
                rotation_list.append(torch.from_numpy(rotation_R))
            else:
                img = self.rgb_centering(img, center, scale)
                
            img_list.append(img)
            
        item['img'] = self.normalize_img(torch.Tensor(img_list))
        if len(rotation_list) > 0:
            item['rotation'] = torch.cat(rotation_list, dim=0).float()

        item['shape'] = torch.from_numpy(self.shape[index[0]].copy()).float()
        item['gender'] = self.gender[index[0]]
        
        if self.use_smpl_params:
            item['smpl'] = torch.Tensor(self.pose[list(index)])
        
        return item
    
    
# Sampler for test dataset
class TestDatasetSampler(Sampler):
    def __init__(self,
                 data_source,
                 protocol=None,
                 batch_size=4,
                 n_seq_per_person=10,
                 static_n_frames=False,
                 mean_frames_in_seq=5,
                 max_frames_in_seq=16,
                 drop_last=False):
        """
        data_source : TestDataset
        protocol : dataset protocol(fixed indices), None for no protocol
        batch_size : batch size for people in one process
        n_seq_per_person : number of sequences per person in a series
        static_n_frames : fix the number of frames in a sequence when True,
                          pick a number of frames from a Gaussian distribution when False
        mean_frames_in_seq : number of frames in a sequence when static_n_frames=True,
                             mean number of frames in a sequence when static_n_frames=False
        max_frames_in_seq : maximum number of frames in a sequence
        drop_last : whether to drop the last insufficient batch
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if protocol != None:
            self.protocol = True
            with np.load(config.TEST_PROTOCOL_DIR[protocol]) as fixed_indices:
                self.fixed_index = fixed_indices[str(mean_frames_in_seq)]
            self.length = len(self.fixed_index)
        else:
            self.protocol = False
            self.static_n_frames = static_n_frames
            self.mean_frames_in_seq = mean_frames_in_seq
            self.max_frames_in_seq = max_frames_in_seq
        
            self.last_index = np.array(data_source.get_data_item('series_index'))
            self.first_index = np.concatenate(([0], self.last_index[:-1]))
            self.person_id = list(range(len(self.last_index))) * n_seq_per_person
            self.length = len(self.person_id)

            if not static_n_frames:
                self.sigma = mean_frames_in_seq - 2 if mean_frames_in_seq > 2 else 1
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        if self.protocol:
            frame_index = self.fixed_index
        
        else:
            person_index = np.random.permutation(self.person_id)

            if self.static_n_frames:
                frame_index = np.random.randint(self.first_index[self.person_id], self.last_index[self.person_id], size=(self.mean_frames_in_seq, self.length))
                frame_index = np.transpose(frame_index)

            else:
                # Random sequence lengths
                # Guarantee At least 16% probability for length 1.
                n_frames = np.random.randn(self.length // self.batch_size + 1) * self.sigma + self.mean_frames_in_seq
                n_frames = n_frames.astype(int)
                n_frames[(n_frames < 1) | (n_frames > self.max_frames_in_seq)] = 1
                n_frames = np.repeat(n_frames, self.batch_size)[:self.length]

                frame_index = [np.random.randint(self.first_index[i], self.last_index[i], size=j) for i, j in zip(person_index, n_frames)]
            
        batch = []
        id_count = 0
        for sample in map(tuple, frame_index):
            batch.append(sample)
            id_count += 1
            if id_count >= self.batch_size:
                yield batch
                id_count = 0
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
    
    
# Collate function for test dataset
def test_dataset_collate_fn(batch):
    _batch = {}
    img_list = []
    shape_list = []
    gender_list = []

    for item in batch:
        img_list.append(item['img'])
        gender_list.append(item['gender'])
        shape_list.append(item['shape'])
        
    _batch['img'] = torch.cat(img_list, dim=0)
    _batch['shape'] = torch.stack(shape_list)
    _batch['gender'] = np.array(gender_list)
    
    if 'smpl' in batch[0].keys():
        _batch['smpl'] = align_rotation(rodrigues(torch.cat([item['smpl'] for item in batch], dim=0))).flatten(1)
    if 'rotation' in batch[0].keys():
        _batch['rotation'] = torch.cat([item['rotation'] for item in batch], dim=0)

    return _batch


# Get a data loader for test dataset.
def get_test_dataset_loader(dataset='3dpw',
                            model_type='simple',
                            set_type='test',
                            protocol=False,
                            batch_size=32,
                            augmentation='full',
                            use_smpl_params=False,
                            n_seq_per_person=10,
                            static_n_frames=False,
                            mean_frames_in_seq=5,
                            max_frames_in_seq=12,
                            pin_memory=False,
                            drop_last=False):
    
    _dataset = TestDataset(dataset=dataset, model_type=model_type, set_type=set_type, augmentation=augmentation, use_smpl_params=use_smpl_params)
    if model_type == 'simple':
        return DataLoader(_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=drop_last, collate_fn=test_dataset_collate_fn, num_workers=16)
    else:
        _protocol = dataset if protocol and set_type == 'test' else None
        sampler = TestDatasetSampler(_dataset,
                                     protocol=_protocol,
                                     batch_size=batch_size,
                                     n_seq_per_person=n_seq_per_person,
                                     static_n_frames=static_n_frames,
                                     mean_frames_in_seq=mean_frames_in_seq,
                                     max_frames_in_seq=max_frames_in_seq,
                                     drop_last=drop_last)
        return DataLoader(_dataset, batch_sampler=sampler, pin_memory=pin_memory, collate_fn=test_dataset_collate_fn, num_workers=16)
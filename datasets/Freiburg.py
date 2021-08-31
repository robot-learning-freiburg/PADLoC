import h5py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from scipy.io import loadmat
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import pickle

import utils.rotation_conversion as RT


class FreiburgDataset(Dataset):
    def __init__(self, base_folder, without_ground=False):
        super(FreiburgDataset, self).__init__()
        self.base_folder = base_folder
        self.without_ground = without_ground
        self.all_files = os.listdir(os.path.join(self.base_folder, 'gps_aligned'))
        self.all_files = sorted(self.all_files)
        # self.files_with_gt = []
        self.poses = []
        for i in tqdm(range(len(self.all_files))):
            path = os.path.join(self.base_folder, 'gps_aligned', self.all_files[i])
            pose = np.eye(4)
            with open(path, 'r') as f:
                line = f.read()
                split = line.split(',')
                pose[0, -1] = split[0]
                pose[1, -1] = split[1]
            self.poses.append(pose)

    def get_velo(self, idx):
        if not self.without_ground:
            path = os.path.join(self.base_folder, 'velodyne', f'{self.all_files[idx][:-4]}.npy')
            scan = np.load(path)
            scan = scan.reshape(-1, 4)
            scan[:, -1] /= 255.
        else:
            path = os.path.join(self.base_folder, 'velodyne_no_ground', f'{self.all_files[idx][:-4]}.npy')
            with h5py.File(path, 'r') as hf:
                scan = hf['PC'][:]
            scan = scan.reshape((-1, 4))
        # scan[:, 2] -= 0.55  # TEST
        return scan

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        pc = torch.from_numpy(self.get_velo(idx)).float()
        return {'anchor': pc}


class FreiburgRegistrationDataset(Dataset):
    def __init__(self, base_folder, without_ground=False, get_pc=True):
        super(FreiburgRegistrationDataset, self).__init__()
        self.base_folder = base_folder
        self.without_ground = without_ground
        self.get_pc = get_pc
        self.pairs = []
        # root_path = '/home/cattaneo/Datasets/iter200'
        # root_path = '/media/RAIDONE/vaghi/Freiburg/gt_icp/iter200'
        root_path = os.path.join(base_folder, 'iter1000')
        pairs = os.path.join(root_path, 'icp_pairs.csv')
        self.pairs = pd.read_csv(pairs)
        RT_source = os.path.join(root_path, 'RT_source.pickle')
        with open(RT_source, 'rb') as f:
            self.RT_source = pickle.load(f)
        self.RT_source = [f[1] for f in list(self.RT_source.items())]
        self.RT_source = np.stack(self.RT_source)
        RT_target = os.path.join(root_path, 'RT_target.pickle')
        with open(RT_target, 'rb') as f:
            self.RT_target = pickle.load(f)
        self.RT_target = [f[1] for f in list(self.RT_target.items())]
        self.RT_target = np.stack(self.RT_target)
        RT_icp = os.path.join(root_path, 'RT_icp.pickle')
        with open(RT_icp, 'rb') as f:
            self.RT_icp = pickle.load(f)
        self.RT_icp = [f[1] for f in list(self.RT_icp.items())]
        self.RT_icp = np.stack(self.RT_icp)
        self.lidar2gps = np.array([[0., -1., 0., -0.157],
                                   [1., 0., 0., -0.883],
                                   [0., 0., 1., -1.453],
                                   [0., 0., 0., 1.]])
        valid_idxs = self.pairs.fitness > 0.6
        valid_idxs = valid_idxs & (self.pairs.rmse < 0.3)
        self.pairs = self.pairs[valid_idxs]
        self.RT_source = self.RT_source[valid_idxs]
        self.RT_target = self.RT_target[valid_idxs]
        self.RT_icp = self.RT_icp[valid_idxs]

        self.all_files = os.listdir(os.path.join(self.base_folder, 'gps_aligned'))
        self.all_files = sorted(self.all_files)
        self.all_files = [f[:-4] for f in self.all_files]


    def get_velo(self, idx):
        if not self.without_ground:
            path = os.path.join(self.base_folder, 'velodyne', f'{idx}.npy')
            scan = np.load(path)
            scan = scan.reshape(-1, 4)
            scan[:, -1] /= 255.
        else:
            path = os.path.join(self.base_folder, 'velodyne_no_ground', f'{idx}.npy')
            with h5py.File(path, 'r') as hf:
                scan = hf['PC'][:]
            scan = scan.reshape((-1, 4))
        # scan[:, 2] -= 0.55  # TEST
        return scan

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        id_source = self.pairs['source'].iloc[idx][:-4]
        id_target = self.pairs['target'].iloc[idx][:-4]

        pc_anchor = None
        pc_positive = None
        if self.get_pc:
            pc_anchor = torch.from_numpy(self.get_velo(id_source)).float()
            pc_positive = torch.from_numpy(self.get_velo(id_target)).float()

        transformation = self.lidar2gps @ np.linalg.inv(self.RT_source[idx])
        transformation = transformation @ np.linalg.inv(self.RT_icp[idx])
        transformation = transformation @ self.RT_target[idx]
        transformation = transformation @ np.linalg.inv(self.lidar2gps)
        # transformation = np.linalg.inv(transformation)

        return {'anchor': pc_anchor, 'positive': pc_positive, 'transformation': transformation,
                'anchor_id': self.all_files.index(id_source), 'positive_id': self.all_files.index(id_target),
                'anchor_stamp': id_source, 'positive_stamp': id_target,
                'diff_level': self.pairs['diff_level'].iloc[idx]}


class FreiburgRegistrationRPMDataset(Dataset):
    def __init__(self, base_folder, without_ground=False):
        super(FreiburgRegistrationRPMDataset, self).__init__()
        self.base_folder = base_folder
        self.without_ground = without_ground
        self.pairs = []
        # root_path = '/home/cattaneo/Datasets/iter200'
        # root_path = '/media/RAIDONE/vaghi/Freiburg/gt_icp/iter200'
        root_path = os.path.join(base_folder, 'iter1000')
        pairs = os.path.join(root_path, 'icp_pairs.csv')
        self.pairs = pd.read_csv(pairs)
        RT_source = os.path.join(root_path, 'RT_source.pickle')
        with open(RT_source, 'rb') as f:
            self.RT_source = pickle.load(f)
        self.RT_source = [f[1] for f in list(self.RT_source.items())]
        self.RT_source = np.stack(self.RT_source)
        RT_target = os.path.join(root_path, 'RT_target.pickle')
        with open(RT_target, 'rb') as f:
            self.RT_target = pickle.load(f)
        self.RT_target = [f[1] for f in list(self.RT_target.items())]
        self.RT_target = np.stack(self.RT_target)
        RT_icp = os.path.join(root_path, 'RT_icp.pickle')
        with open(RT_icp, 'rb') as f:
            self.RT_icp = pickle.load(f)
        self.RT_icp = [f[1] for f in list(self.RT_icp.items())]
        self.RT_icp = np.stack(self.RT_icp)
        self.lidar2gps = np.array([[0., -1., 0., -0.157],
                                   [1., 0., 0., -0.883],
                                   [0., 0., 1., -1.453],
                                   [0., 0., 0., 1.]])
        valid_idxs = self.pairs.fitness > 0.6
        valid_idxs = valid_idxs & (self.pairs.rmse < 0.3)
        self.pairs = self.pairs[valid_idxs]
        self.RT_source = self.RT_source[valid_idxs]
        self.RT_target = self.RT_target[valid_idxs]
        self.RT_icp = self.RT_icp[valid_idxs]

        self.all_files = os.listdir(os.path.join(self.base_folder, 'gps_aligned'))
        self.all_files = sorted(self.all_files)
        self.all_files = [f[:-4] for f in self.all_files]


    def get_velo(self, idx):
        if not self.without_ground:
            path = os.path.join(self.base_folder, 'velodyne_rpmnet', f'{idx}.h5')
            with h5py.File(path, 'r') as hf:
                scan = hf['PC'][:]

            scan = scan.reshape((-1, 6))
        else:
            raise NotImplementedError()
        # scan[:, 2] -= 0.55  # TEST
        return scan

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        id_source = self.pairs['source'].iloc[idx][:-4]
        id_target = self.pairs['target'].iloc[idx][:-4]

        pc_anchor = torch.from_numpy(self.get_velo(id_source)).float()
        pc_positive = torch.from_numpy(self.get_velo(id_target)).float()

        transformation = self.lidar2gps @ np.linalg.inv(self.RT_source[idx])
        transformation = transformation @ np.linalg.inv(self.RT_icp[idx])
        transformation = transformation @ self.RT_target[idx]
        transformation = transformation @ np.linalg.inv(self.lidar2gps)
        # transformation = np.linalg.inv(transformation)

        return {'anchor': pc_anchor, 'positive': pc_positive, 'transformation': transformation,
                'anchor_id': self.all_files.index(id_source), 'positive_id': self.all_files.index(id_target),
                'anchor_stamp': id_source, 'positive_stamp': id_target,
                'diff_level': self.pairs['diff_level'].iloc[idx]}

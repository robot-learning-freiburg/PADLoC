import pickle

import h5py
import numpy as np
import os
import torch

from torch.utils.data import Dataset

import utils.rotation_conversion as RT


class NCLTDataset(Dataset):
    def __init__(self, base_path, sequence, loop_gt_file=None):
        super(NCLTDataset, self).__init__()
        self.base_path = base_path
        self.sequence = sequence
        self.all_files = os.listdir(os.path.join(base_path, sequence, 'velodyne_subsample_no_ground'))
        self.all_files.sort()
        pose_file = os.path.join(base_path, sequence, 'poses_subsample.txt')
        poses = np.loadtxt(pose_file)
        self.poses = np.zeros((poses.shape[0], 4, 4))
        for i in range(poses.shape[0]):
            self.poses[i] = RT.to_rotation_matrix_XYZRPY(*poses[i])
        if loop_gt_file is not None:
            with open(os.path.join(base_path, sequence, loop_gt_file), 'rb') as f:
                loop_gt = pickle.load(f)
            self.have_matches = [elem['idx'] for elem in loop_gt]

    def get_velo(self, idx):
        velo_path = os.path.join(self.base_path, self.sequence, 'velodyne_subsample_no_ground', f'{idx:06d}.h5')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
        # scan[:, 3] = scan[:, 3] / 255.
        return scan

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        anchor_pcd = torch.from_numpy(self.get_velo(idx)).float()
        sample = {'anchor': anchor_pcd}
        return sample


class NCLTDatasetPairs(Dataset):
    def __init__(self, base_path, map_seq, query_seq, loop_gt_file):
        super(NCLTDatasetPairs, self).__init__()
        self.base_path = base_path
        self.map_seq = map_seq
        self.query_seq = query_seq
        self.all_files = os.listdir(os.path.join(base_path, map_seq, 'velodyne_subsample_no_ground'))
        self.all_files2 = os.listdir(os.path.join(base_path, query_seq, 'velodyne_subsample_no_ground'))
        self.all_files.sort()
        self.all_files2.sort()
        pose_file = os.path.join(base_path, map_seq, 'poses_subsample.txt')
        self.poses_map = np.loadtxt(pose_file)
        pose_file2 = os.path.join(base_path, query_seq, 'poses_subsample.txt')
        self.poses_query = np.loadtxt(pose_file2)
        with open(os.path.join(base_path, query_seq, loop_gt_file), 'rb') as f:
            loop_gt = pickle.load(f)
        self.loop_gt = loop_gt

    def get_velo(self, seq, idx):
        velo_path = os.path.join(self.base_path, self.sequence, 'velodyne_subsample_no_ground', f'{idx:06d}.h5')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
        # scan[:, 3] = scan[:, 3] / 255.
        return scan

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        anchor_pcd = torch.from_numpy(self.get_velo(self.query_seq, frame_idx)).float()
        anchor_pose = self.poses_query[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3]).float()
        anchor_rot_torch = torch.tensor(anchor_pose[3:]).float()

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])
        positive_pcd = torch.from_numpy(self.get_velo(self.map_seq, positive_idx)).float()
        positive_pose = self.poses_map[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3]).float()
        positive_rot_torch = torch.tensor(positive_pose[3:]).float()

        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx
                  }
        return sample


class NCLTDatasetTriplets(Dataset):
    def __init__(self, base_path, map_seq, query_seq, loop_gt_file, hard_negative=False):
        super(NCLTDatasetTriplets, self).__init__()
        self.base_path = base_path
        self.map_seq = map_seq
        self.query_seq = query_seq
        self.hard_negative = hard_negative
        self.all_files = os.listdir(os.path.join(base_path, map_seq, 'velodyne_subsample_no_ground'))
        self.all_files2 = os.listdir(os.path.join(base_path, query_seq, 'velodyne_subsample_no_ground'))
        self.all_files.sort()
        self.all_files2.sort()
        pose_file = os.path.join(base_path, map_seq, 'poses_subsample.txt')
        self.poses_map = np.loadtxt(pose_file)
        pose_file2 = os.path.join(base_path, query_seq, 'poses_subsample.txt')
        self.poses_query = np.loadtxt(pose_file2)
        with open(os.path.join(base_path, query_seq, loop_gt_file), 'rb') as f:
            loop_gt = pickle.load(f)
        self.loop_gt = loop_gt

    def get_velo(self, seq, idx):
        velo_path = os.path.join(self.base_path, self.sequence, 'velodyne_subsample_no_ground', f'{idx:06d}.h5')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
        # scan[:, 3] = scan[:, 3] / 255.
        return scan

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        anchor_pcd = torch.from_numpy(self.get_velo(self.query_seq, frame_idx)).float()
        anchor_pose = self.poses_query[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3]).float()
        anchor_rot_torch = torch.tensor(anchor_pose[3:]).float()

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])
        positive_pcd = torch.from_numpy(self.get_velo(self.map_seq, positive_idx)).float()
        positive_pose = self.poses_map[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3]).float()
        positive_rot_torch = torch.tensor(positive_pose[3:]).float()

        if self.hard_negative:
            negative_idx = np.random.choice(list(self.loop_gt[idx]['hard_idxs']))
        else:
            negative_idx = np.random.choice(self.loop_gt[idx]['negative_idxs'])
        negative_pcd = torch.from_numpy(self.get_velo(self.map_seq, negative_idx)).float()
        negative_pose = self.poses_map[negative_idx]
        negative_transl = torch.tensor(negative_pose[:3]).float()
        negative_rot_torch = torch.tensor(negative_pose[3:]).float()

        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  'negative': negative_pcd,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'negative_pose': negative_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'negative_rot': negative_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx,
                  'negative_idx': negative_idx
                  }
        return sample

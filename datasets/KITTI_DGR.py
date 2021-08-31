import h5py
import torch
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset

from skimage import io
from PIL import Image
import os, os.path
import pandas as pd
import numpy as np
import random
import pickle
import pykitti

import utils.rotation_conversion as RT


def get_velo(idx, dir, sequence, without_ground, jitter=False):
    if without_ground:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}',
                                 'velodyne_no_ground', f'{idx:06d}.h5')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
    else:
        velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}', 'velodyne', f'{idx:06d}.bin')
        scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    return scan


def get_velo_with_panoptic(idx, dir, sequence, without_ground, jitter=False):
    raise NotImplementedError()


def get_velo_360(idx, dir, sequence, without_ground, jitter=False):
    velo_path = os.path.join(dir, 'data_3d_raw', sequence,
                             'velodyne_rpmnet', f'{idx:010d}.h5')
    with h5py.File(velo_path, 'r') as hf:
        scan = hf['PC'][:]

    scan = scan.reshape((-1, 6))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    return scan


class KITTIDGR3DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, without_ground=False, loop_file='loop_GT', jitter=False,
                 use_semantic=False, use_panoptic=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTIDGR3DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = int(sequence)
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        data = read_calib_file(os.path.join(dir, 'sequences', sequence, 'calib.txt'))
        cam0_to_velo = np.reshape(data['Tr'], (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)
        poses2 = []
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[0:4])
                pose[1, 0:4] = torch.tensor(x[4:8])
                pose[2, 0:4] = torch.tensor(x[8:12])
                pose[3, 3] = 1.0
                pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
                poses2.append(pose.float().numpy())
        self.poses = poses2
        self.npoints = npoints
        self.device = device
        self.without_ground = without_ground

        Ts = np.stack(poses2)[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
        pdist = np.sqrt(pdist.sum(-1))
        more_than_10 = pdist > 10.
        inames = range(len(poses2))
        curr_time = inames[0]
        self.files = []
        while curr_time in inames:
            # Find the min index
            next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0:
                curr_time += 1
            else:
                # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                next_time = next_time[0] + curr_time - 1

            if next_time in inames:
                self.files.append((self.sequence, curr_time, next_time))
                curr_time = next_time + 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frame_idx = self.files[idx][1]
        if frame_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = self.files[idx][2]
        # positive_idx = self.files[idx][1]

        positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        if positive_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        positive_pose = self.poses[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        r_anch = anchor_pose
        r_pos = positive_pose
        r_anch = RT.npto_XYZRPY(r_anch)[3:]
        r_pos = RT.npto_XYZRPY(r_pos)[3:]

        anchor_rot_torch = torch.tensor(r_anch.copy(), dtype=torch.float32)
        positive_rot_torch = torch.tensor(r_pos.copy(), dtype=torch.float32)

        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  'sequence': self.sequence,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx,
                  'anchor_rt': anchor_pose,
                  'positive_rt': positive_pose
                  }

        return sample
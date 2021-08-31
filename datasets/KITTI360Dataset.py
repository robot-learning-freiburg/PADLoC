import faiss
import h5py
import torch
from torch.utils.data import Dataset

import os, os.path
import numpy as np
import random
import pickle
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from datasets.KITTI_data_loader import KITTILoader3DPoses

mpl.rcParams['figure.dpi'] = 300

import utils.rotation_conversion as RT
from sklearn.neighbors import KDTree


def get_velo(idx, dir, sequence, without_ground, jitter=False):
    if without_ground:
        velo_path = os.path.join(dir, 'data_3d_raw', sequence,
                                 'velodyne_no_ground', f'{idx:010d}.npy')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
    else:
        velo_path = os.path.join(dir, 'data_3d_raw', sequence,
                                 'velodyne_points', 'data', f'{idx:010d}.bin')
        scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    return scan


def get_velo_with_panoptic(idx, dir, sequence, use_semantic=True, use_panoptic=False, jitter=False):
    raise NotImplementedError()


class KITTI3603DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, without_ground=False,
                 train=True, loop_file='loop_GT', jitter=False, use_semantic=False, use_panoptic=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """
        super(KITTI3603DPoses, self).__init__()

        self.dir = dir
        self.sequence = sequence
        self.jitter = jitter
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = []
        poses = os.path.join(dir, 'data_poses', sequence, 'cam0_to_world.txt')
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                self.frames_with_gt.append(int(x[0]))
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                poses2.append(pose.float().numpy())
        self.poses = poses2
        self.train = train
        self.without_ground = without_ground

        gt_file = os.path.join(dir, 'data_poses', sequence, f'{loop_file}.pickle')
        self.loop_gt = []
        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            for elem in temp:
                temp_dict = {'idx': elem['idx'], 'positive_idxs': elem['positive_idxs']}
                self.loop_gt.append(temp_dict)
            del temp
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.frames_with_gt)

    def __getitem__(self, idx):
        frame_idx = self.frames_with_gt[idx]

        if self.use_panoptic or self.use_semantic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(frame_idx, self.dir, self.sequence,
                                                               self.use_semantic, self.use_panoptic, self.jitter)
            anchor_pcd = torch.from_numpy(anchor_pcd)
            anchor_logits = torch.from_numpy(anchor_logits)
        else:
            anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        if self.train:
            x = self.poses[idx][0, 3]
            y = self.poses[idx][1, 3]
            z = self.poses[idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])
            negative_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = frame_idx
            negative_idx = frame_idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[frame_idx][0, 3]
                possible_match_pose[1] = self.poses[frame_idx][1, 3]
                possible_match_pose[2] = self.poses[frame_idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 5 and frame_idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 25 and frame_idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
                    negative_idx = i
                    cont += 1
            if self.use_panoptic or self.use_semantic:
                positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                   self.use_semantic, self.use_panoptic, self.jitter)
                positive_pcd = torch.from_numpy(positive_pcd)
                positive_logits = torch.from_numpy(positive_logits)

                negative_pcd, negative_logits = get_velo_with_panoptic(negative_idx, self.dir, self.sequence,
                                                                       self.use_semantic, self.use_panoptic, self.jitter)
                negative_pcd = torch.from_numpy(negative_pcd)
                negative_logits = torch.from_numpy(negative_logits)
            else:
                positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))
                negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            sample = {'anchor': anchor_pcd,
                      'positive': positive_pcd,
                      'negative': negative_pcd}
            if self.use_panoptic or self.use_semantic:
                sample['anchor_logits'] = anchor_logits
                sample['positive_logits'] = positive_logits
                sample['negative_logits'] = negative_logits
        else:
            sample = {'anchor': anchor_pcd}
            if self.use_panoptic or self.use_semantic:
                sample['anchor_logits'] = anchor_logits

        return sample


class KITTI3603DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, without_ground=False, loop_file='loop_GT', jitter=False,
                 use_semantic=False, use_panoptic=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTI3603DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = {}
        poses = os.path.join(dir, 'data_poses', sequence, 'cam0_to_world.txt')
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                self.frames_with_gt.append(int(x[0]))
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                # poses2.append(pose.float().numpy())
                poses2[int(x[0])] = pose.float().numpy()
        self.poses = poses2
        self.without_ground = without_ground
        gt_file = os.path.join(dir, 'data_poses', sequence, f'{loop_file}.pickle')
        self.loop_gt = []
        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            for elem in temp:
                temp_dict = {'idx': elem['idx'], 'positive_idxs': elem['positive_idxs']}
                self.loop_gt.append(temp_dict)
            del temp
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        if self.use_panoptic or self.use_semantic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(frame_idx, self.dir, self.sequence,
                                                               self.use_semantic, self.use_panoptic, self.jitter)
            anchor_pcd = torch.from_numpy(anchor_pcd)
            anchor_logits = torch.from_numpy(anchor_logits)
        else:
            anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])

        if self.use_panoptic or self.use_semantic:
            positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                   self.use_semantic, self.use_panoptic, self.jitter)
            positive_pcd = torch.from_numpy(positive_pcd)
            positive_logits = torch.from_numpy(positive_logits)
        else:
            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        if positive_idx not in self.poses:
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
                  'sequence': self.sequence_int,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx
                  }
        if self.use_panoptic or self.use_semantic:
            sample['anchor_logits'] = anchor_logits
            sample['positive_logits'] = positive_logits

        return sample


class KITTI3603DDictTriplets(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, without_ground=False,
                 loop_file='loop_GT', hard_negative=False, jitter=False, use_semantic=False, use_panoptic=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTI3603DDictTriplets, self).__init__()

        self.jitter = jitter
        self.dir = dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.hard_negative = hard_negative
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        calib_file = os.path.join(dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        self.frames_with_gt = []
        poses2 = {}
        poses = os.path.join(dir, 'data_poses', sequence, 'cam0_to_world.txt')
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                self.frames_with_gt.append(int(x[0]))
                pose = torch.zeros((4, 4), dtype=torch.float64)
                pose[0, 0:4] = torch.tensor(x[1:5])
                pose[1, 0:4] = torch.tensor(x[5:9])
                pose[2, 0:4] = torch.tensor(x[9:13])
                pose[3, 3] = 1.0
                pose = pose @ cam0_to_velo.inverse()
                # poses2.append(pose.float().numpy())
                poses2[int(x[0])] = pose.float().numpy()
        self.poses = poses2

        self.without_ground = without_ground
        gt_file = os.path.join(dir, 'data_poses', sequence, f'{loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        if self.use_panoptic or self.use_semantic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(frame_idx, self.dir, self.sequence,
                                                               self.use_semantic, self.use_panoptic, self.jitter)
            anchor_pcd = torch.from_numpy(anchor_pcd)
            anchor_logits = torch.from_numpy(anchor_logits)
        else:
            anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])
        if self.use_panoptic or self.use_semantic:
            positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                   self.use_semantic, self.use_panoptic, self.jitter)
            positive_pcd = torch.from_numpy(positive_pcd)
            positive_logits = torch.from_numpy(positive_logits)
        else:
            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        if positive_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        positive_pose = self.poses[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        if self.hard_negative and len(self.loop_gt[idx]['hard_idxs']) > 0:
            negative_idx = np.random.choice(list(self.loop_gt[idx]['hard_idxs']))
        else:
            negative_idx = np.random.choice(self.loop_gt[idx]['negative_idxs'])

        if self.use_panoptic or self.use_semantic:
            negative_pcd, negative_logits = get_velo_with_panoptic(negative_idx, self.dir, self.sequence,
                                                                   self.use_semantic, self.use_panoptic, self.jitter)
            negative_pcd = torch.from_numpy(negative_pcd)
            negative_logits = torch.from_numpy(negative_logits)
        else:
            negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.without_ground, self.jitter))

        negative_pose = self.poses[negative_idx]
        negative_transl = torch.tensor(negative_pose[:3, 3]).float()

        r_anch = anchor_pose
        r_pos = positive_pose
        r_anch = RT.npto_XYZRPY(r_anch)[3:]
        r_pos = RT.npto_XYZRPY(r_pos)[3:]
        r_neg = RT.npto_XYZRPY(negative_pose)[3:]

        anchor_rot_torch = torch.tensor(r_anch.copy(), dtype=torch.float32)
        positive_rot_torch = torch.tensor(r_pos.copy(), dtype=torch.float32)
        negative_rot_torch = torch.tensor(r_neg.copy(), dtype=torch.float32)

        sample = {'anchor': anchor_pcd,
                  'positive': positive_pcd,
                  'negative': negative_pcd,
                  'sequence': self.sequence_int,
                  'anchor_pose': anchor_transl,
                  'positive_pose': positive_transl,
                  'negative_pose': negative_transl,
                  'anchor_rot': anchor_rot_torch,
                  'positive_rot': positive_rot_torch,
                  'negative_rot': negative_rot_torch,
                  'anchor_idx': frame_idx,
                  'positive_idx': positive_idx,
                  'negative_idx': negative_idx,
                  }

        if self.use_panoptic or self.use_semantic:
            sample['anchor_logits'] = anchor_logits
            sample['positive_logits'] = positive_logits
            sample['negative_logits'] = negative_logits

        return sample

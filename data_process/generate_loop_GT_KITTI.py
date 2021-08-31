import torch
from scipy.io import savemat
from torch.utils.data import Dataset
import pandas as pd
import pykitti
import os
from sklearn.neighbors import KDTree
import pickle
import numpy as np


class KITTILoader3DPosesOnlyLoopPositives(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, positive_range=5., negative_range=25., hard_range=None):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """
        super(KITTILoader3DPosesOnlyLoopPositives, self).__init__()

        self.positive_range = positive_range
        self.negative_range = negative_range
        self.hard_range = hard_range
        self.dir = dir
        self.sequence = sequence
        self.data = pykitti.odometry(dir, sequence)
        # self.poses = pd.read_csv(poses)
        poses2 = []
        T_cam_velo = np.array(self.data.calib.T_cam0_velo)
        with open(poses, 'r') as f:
            for x in f:
                x = x.strip().split()
                x = [float(v) for v in x]
                pose = np.zeros((4, 4))
                pose[0, 0:4] = np.array(x[0:4])
                pose[1, 0:4] = np.array(x[4:8])
                pose[2, 0:4] = np.array(x[8:12])
                pose[3, 3] = 1.0
                pose = np.linalg.inv(T_cam_velo) @ (pose @ T_cam_velo)
                poses2.append(pose)
        self.poses = np.stack(poses2)
        self.npoints = npoints
        self.device = device
        self.kdtree = KDTree(self.poses[:, :3, 3])

    def __len__(self):
        return len(self.data.timestamps)

    def __getitem__(self, idx):

        x = self.poses[idx, 0, 3]
        y = self.poses[idx, 1, 3]
        z = self.poses[idx, 2, 3]

        anchor_pose = torch.tensor([x, y, z])
        possible_match_pose = torch.tensor([0., 0., 0.])
        negative_pose = torch.tensor([0., 0., 0.])

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.positive_range)
        num_loop = 0
        min_range = max(0, idx-50)
        # min_range = max(0, idx-1)  # FOR SG_PR
        max_range = min(idx+50, len(self.data.timestamps))
        # max_range = min(idx+1, len(self.data.timestamps))  # FOR SG_PR
        positive_idxs = list(set(indices[0]) - set(range(min_range, max_range)))
        positive_idxs.sort()
        num_loop = len(positive_idxs)

        indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.negative_range)
        indices = set(indices[0])
        negative_idxs = set(range(len(self.data.timestamps))) - indices
        negative_idxs = list(negative_idxs)
        negative_idxs.sort()

        hard_idxs = None
        if self.hard_range is not None:
            inner_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[0])
            outer_indices = self.kdtree.query_radius(anchor_pose.unsqueeze(0).numpy(), self.hard_range[1])
            hard_idxs = set(outer_indices[0]) - set(inner_indices[0])
            pass

        return num_loop, positive_idxs, negative_idxs, hard_idxs


if __name__ == '__main__':
    base_dir = "/media/RAIDONE/DATASETS/KITTI/ODOMETRY"
    sequence = "00"
    for sequence in ["00", "03", "04", "05", "06", "07", "08", "09"]:
        # poses_file = base_dir + "/sequences/" + sequence + "/poses.csv"
        poses_file = base_dir + "/sequences/" + sequence + "/poses_SEMANTICKITTI.txt"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # dataset = KITTILoader3DPoses(dir, sequence, poses_file, 4096, device)

        dataset = KITTILoader3DPosesOnlyLoopPositives(base_dir, sequence, poses_file, 1024, device, 4, 10, [6, 10])
        lc_gt = []
        lc_gt_file = os.path.join(base_dir, 'sequences', sequence, 'loop_GT_4m.pickle')

        # savemat(f'GTPoses_{sequence}.mat', {'GTPoses': dataset.poses[:, :3, 3]})
        map_tree_poses = KDTree(dataset.poses[:, :3, 3])
        real_loop = []
        for i in range(100, dataset.poses.shape[0]):
            min_range = max(0, i-50)
            current_pose = dataset.poses[i, :3, 3]
            indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), 4)
            valid_idxs = list(set(indices[0]) - set(range(min_range, dataset.poses.shape[0])))
            if len(valid_idxs) > 0:
                real_loop.append(i)
        with open(f'real_loop_4m_{sequence}.pickle', 'wb') as f:
            pickle.dump(real_loop, f)

        for i in range(len(dataset)):

            sample, pos, neg, hard = dataset[i]
            if sample > 0.:
                print(i, sample)
                sample_dict = {}
                sample_dict['idx'] = i
                sample_dict['positive_idxs'] = pos
                sample_dict['negative_idxs'] = neg
                sample_dict['hard_idxs'] = hard
                lc_gt.append(sample_dict)
        with open(lc_gt_file, 'wb') as f:
            pickle.dump(lc_gt, f)

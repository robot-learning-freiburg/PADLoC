from collections import namedtuple
import h5py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from scipy.io import loadmat
from tqdm import tqdm

import utils.rotation_conversion as RT


SequenceCfg = namedtuple("SequenceCfg", ["start_frame", "end_frame"])


class FordCampusDataset(Dataset):
    SEQUENCES = {
        "1":        SequenceCfg(  75, 3891),
        "1-subset": SequenceCfg(1000, 1199),
        "2":        SequenceCfg(   1, 6103),
    }

    def __init__(self, base_folder, seq, without_ground=False):
        super(FordCampusDataset, self).__init__()
        self.base_folder = base_folder
        if seq not in self.SEQUENCES:
            raise ValueError(f"Invalid sequence {seq}. Valid values: {self.SEQUENCES}")
        self.without_ground = without_ground
        self.sequence_path = os.path.join(self.base_folder, f"dataset-{seq}", f"IJRR-Dataset-{seq}")
        # Use the Scans containing the reflectivities, converted using the create_ijrr_dataset() function in MATLAB
        self.scan_path = os.path.join(self.sequence_path, "RSCANS")
        self.poses_path = os.path.join(self.sequence_path, "poses_rscans.txt")

        seq_cfg = self.SEQUENCES[seq]
        start_frame = seq_cfg.start_frame
        end_frame = seq_cfg.end_frame
        self.frames = np.arange(start_frame, end_frame + 1)

        # Takes too long to load the poses, since the entire scan has to be read.
        # Better to save them to a different file for future reusability
        if os.path.exists(self.poses_path):
            self.poses = np.loadtxt(self.poses_path).reshape((-1, 4, 4))
            assert self.poses.shape[0] == self.frames.shape[0]
        else:
            print(f"Extracting poses into {self.poses_path}")
            poses = [self.extract_pose(i) for i in tqdm(self.frames)]
            self.poses = np.stack(poses)
            np.savetxt(self.poses_path, self.poses.reshape(-1, 16))

    def extract_pose(self, idx):
        path = os.path.join(self.scan_path, f"Scan{idx:04d}.mat")
        scan = loadmat(path)
        pose = scan["SCAN"][0, 0]
        pose_fields = pose.dtype.names
        #  Only use the World to Sensor transformation (X_ws) as a pose, not the World to Vehicle one (X_wv)
        if "X_ws" in pose_fields:
            pose = pose["X_ws"]
        # elif "X_wv" in pose_fields:
        #     pose = pose["X_wv"]
        else:
            raise KeyError("Pose not found!")

        pose = RT.to_rotation_matrix_XYZRPY(*pose.squeeze())

        return pose

    def get_velo(self, idx):

        frame = self.frames[idx]
        if not self.without_ground:
            path = os.path.join(self.scan_path, f'Scan{frame:04d}.mat')
            scan = loadmat(path)
            scan = scan['SCAN'][0, 0]
            xyz = scan['XYZ'].T
            refl = scan['reflectivity']/255
            return np.concatenate((xyz, refl), axis=1)
        else:
            path = os.path.join(self.base_folder, 'velodyne_no_ground', f'Scan{frame:04d}.npy')
            with h5py.File(path, 'r') as hf:
                scan = hf['PC'][:]
            return scan.reshape((-1, 4))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pc = torch.from_numpy(self.get_velo(idx)).float()
        return {'anchor': pc}


if __name__ == '__main__':
    dataset = FordCampusDataset('/media/RAIDONE/DATASETS/FordCampus/IJRR-Dataset-1', without_ground=True)
    print(dataset.__getitem__(1200))

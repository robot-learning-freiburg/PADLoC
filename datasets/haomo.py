from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from models.get_models import get_model
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

import open3d as o3d


HaomoSequence = namedtuple("HaomoSequence", ["label", "dir", "poses", "start_frame", "end_frame", "tf_file", "tf_inv"])


class HaomoLoader(Dataset):
    SEQUENCES = {
        #                Lbl    Dir   Pose File       StrF  EndF  TF File                          TF Inv
        1: HaomoSequence("1-1", "1-1", "1-1and1-2.txt",    0, 7000, None,                          False),
        2: HaomoSequence("1-2", "1-1", "1-1and1-2.txt", 7001, None, None,                          False),
        3: HaomoSequence("1-3", "1-3", "1-3.txt",          0, None, "transformation_bet_traj.txt", True)
    }

    def __init__(self, data_dir: Path, sequence: int, stride: int = 1, z_offset=-0.11):

        if sequence not in self.SEQUENCES:
            raise KeyError(f"Invalid sequence {sequence}. Valid values: [{self.SEQUENCES.keys()}].")

        self.sequence = sequence
        seq_cfg = self.SEQUENCES[self.sequence]
        self.sequence_label = seq_cfg.label
        self.dir = data_dir / seq_cfg.dir
        self.poses_file = self.dir / seq_cfg.poses
        self.scans_dir = self.dir / "scans"
        self.z_offset = z_offset

        poses = np.loadtxt(self.poses_file).reshape((-1, 3, 4))
        idx = np.arange(len(poses))
        scan_files = self.scans_dir.iterdir()
        scan_files = sorted([f for f in scan_files if f.is_file()])

        assert len(poses) == len(scan_files)

        start_frame = seq_cfg.start_frame if seq_cfg.start_frame else 0
        end_frame = seq_cfg.end_frame if seq_cfg.end_frame else len(poses)
        frame_step = stride if stride else 1

        self.idx = idx[start_frame:end_frame:frame_step]
        poses = poses[self.idx]
        self.scan_files = scan_files[start_frame:end_frame:frame_step]

        homo = np.zeros((len(poses), 4, 4))
        homo[:, :3, :] = poses
        homo[:, 3, 3] = 1.

        poses = homo

        if seq_cfg.tf_file:
            tf = np.loadtxt(str(self.dir / seq_cfg.tf_file))
            if seq_cfg.tf_inv:
                tf = np.linalg.inv(tf)
            poses = tf @ poses

        self.poses = poses

    def get_velo(self, idx):
        scan = np.fromfile(self.scan_files[idx], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        scan[:, 2] += self.z_offset
        return scan

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return {
            "anchor": self.get_velo(idx),
            "anchor_pose": self.poses[idx],
            "anchor_idx": self.idx[idx]
        }

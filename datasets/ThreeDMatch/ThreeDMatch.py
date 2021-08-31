import os
import logging
import random

import glob
import torch
import torch.utils.data
import numpy as np
from scipy.linalg import expm, norm


def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class PairDataset(torch.utils.data.Dataset):
    AUGMENT = None

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform

        self.random_scale = random_scale
        self.min_scale = 0.8
        self.max_scale = 1.2
        self.random_rotation = random_rotation
        self.rotation_range = 360
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)


class IndoorPairDataset(PairDataset):
    '''
    Train dataset
    '''
    AUGMENT = None

    def __init__(self,
                 phase,
                 root,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                             manual_seed, config)
        self.root = root
        logging.info(f"Loading the subset {phase} from {root}")
        self.OVERLAP_RATIO = 0.3

        subset_names = open(f'{phase}.txt').read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]

        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                    (self.max_scale - self.min_scale) * random.random()
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        sample = {'anchor': xyz0_th,
                  'positive': xyz1_th,
                  'anchor_rt': T1,
                  'positive_rt': T1
                  }

        return sample

if __name__ == '__main__':
    dataset = IndoorPairDataset('train', '/home/cattaneo/Datasets/3DMatch/threedmatch')
    asd = dataset.__getitem__(100)
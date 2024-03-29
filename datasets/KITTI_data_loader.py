import h5py
import torch
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset
import os, os.path
import pandas as pd
import numpy as np
import random
import pickle
import pykitti

import utils.rotation_conversion as RT
from utils.semantic_superclass_mapper import SemanticSuperclassMapper


def get_velo(idx, kitti_dir, sequence, without_ground, jitter=False):
    # Original Without_ground implementation, where the scans were already filtered in a different file
    # if without_ground:
    #     velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}',
    #                              'velodyne_no_ground', f'{idx:06d}.h5')
    #     with h5py.File(velo_path, 'r') as hf:
    #         scan = hf['PC'][:]
    # else:
    #     velo_path = os.path.join(dir, 'sequences', f'{int(sequence):02d}', 'velodyne', f'{idx:06d}.bin')
    #     scan = np.fromfile(velo_path, dtype=np.float32)

    velo_path = os.path.join(kitti_dir, "sequences", f"{int(sequence):02d}", "velodyne", f"{idx:06d}.bin")
    scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # New without-ground implementation,
    # where the indices of the ground points are stored in a file, instead of the filtered scan.
    if without_ground:
        no_ground_idx_path = os.path.join(kitti_dir, "sequences", f"{int(sequence):02d}", "velodyne_no_ground_idxs",
                                       f"{idx:06d}.npy")
        with h5py.File(no_ground_idx_path, "r") as hf:
            no_ground_idx = hf["PC"][:]

        ground_mask = np.zeros(scan.shape[0], bool)
        ground_mask[no_ground_idx] = True

        scan = scan[ground_mask]

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    return scan


def get_velo_with_panoptic(idx, kitti_dir, sequence,
                           use_semantic=True, use_panoptic=False,
                           use_logits=True,
                           filter_dynamic_classes=False, dynamic_classes=None,
                           without_ground=False,
                           jitter=False,
                           **_):

    velo_path = os.path.join(kitti_dir, 'sequences', f'{int(sequence):02d}', 'velodyne', f'{idx:06d}.bin')
    scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    n_points = scan.shape[0]

    if without_ground:
        no_ground_idx_path = os.path.join(kitti_dir, "sequences", f"{int(sequence):02d}", "velodyne_no_ground_idxs",
                                       f"{idx:06d}.npy")
        with h5py.File(no_ground_idx_path, "r") as hf:
            no_ground_idx = hf["PC"][:]

        ground_mask = np.zeros(scan.shape[0], bool)
        ground_mask[no_ground_idx] = True

        scan = scan[ground_mask]

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    if use_logits and not filter_dynamic_classes:

        panoptic_path = os.path.join(kitti_dir, 'sequences', f'{int(sequence):02d}',
                                     'labels', f'{idx:06d}.label')
        panoptic = np.fromfile(panoptic_path, dtype=np.float32)
        panoptic = panoptic.reshape(n_points, -1)

        if use_semantic and use_panoptic:
            logits = panoptic
            pad_width = ((0, 0), (0, 75 - logits.shape[1]))
            assert logits.shape[1] <= 75, "Panoptic Logits Shape Error"
            logits = np.pad(logits, pad_width, constant_values=0)
        elif use_semantic:
            logits = panoptic[:, :19]
        else:
            logits = panoptic[:, 19:]
            pad_width = ((0, 0), (0, 56 - logits.shape[1]))
            assert logits.shape[1] <= 56, "Panoptic Logits Shape Error"
            logits = np.pad(logits, pad_width, constant_values=0)
    else:
        panoptic_path = os.path.join(kitti_dir, 'sequences', f'{int(sequence):02d}',
                                      'labels', f'{idx:06d}.label')
        panoptic = np.fromfile(panoptic_path, dtype=np.int32)
        panoptic = panoptic.reshape(n_points, -1)

        if without_ground:
            panoptic = panoptic[ground_mask]

        semantic = np.bitwise_and(panoptic, 0xFFFF).astype(np.float32)
        instance = np.right_shift(panoptic, 16).astype(np.float32)
        panoptic = panoptic.astype(np.float32)

        if filter_dynamic_classes:
            if dynamic_classes is None:
                dynamic_classes = [
                    1,  # Outlier
                    # Vehicles
                    10,  # Car
                    11,  # Bicycle
                    13,  # Bus
                    15,  # Motorcycle
                    16,  # On-Rails
                    18,  # Truck
                    20,  # Other-Vehicle

                    # Moving Vehicles
                    252,  # Moving-Car
                    257,  # Moving-Bus
                    256,  # Moving-On-Rails
                    258,  # Moving-Truck
                    259,  # Moving-Other-Vehicle

                    # People
                    30,  # Person
                    31,  # Bicyclist
                    32,  # Motorcyclist

                    # Moving People
                    254,  # Moving-Person
                    253,  # Moving-Bicyclist
                    255,  # Moving-Motorcyclist
                ]

            mask = np.ones_like(semantic)

            for k in dynamic_classes:
                mask = np.logical_and(mask, semantic != k)

            mask = np.where(mask)[0]

            scan = scan[mask]
            panoptic = panoptic[mask]
            semantic = semantic[mask]
            instance = instance[mask]

        logits = (panoptic, semantic, instance)

    return scan, logits


def unpack_logits(logits, use_logits, superclass_mapper, prefix, permute=None):

    if use_logits:
        return {prefix + 'logits': torch.from_numpy(logits)}

    panoptic, semantic, instance = logits

    if permute is not None:
        panoptic = panoptic[permute]
        semantic = semantic[permute]
        instance = instance[permute]

    supersem = superclass_mapper.get_superclass(semantic)

    sample = {
        prefix + '_panoptic': torch.from_numpy(panoptic.astype(np.float32)),
        prefix + '_supersem': torch.from_numpy(supersem.astype(np.float32)),
        prefix + '_semantic': torch.from_numpy(semantic.astype(np.float32)),
        prefix + '_instance': torch.from_numpy(instance.astype(np.float32))
    }

    return sample


class KITTILoader3DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, **kwargs):
            #without_ground=False,
            #train=True, loop_file='loop_GT', jitter=False, use_semantic=False, use_panoptic=False, use_logits=True):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        self.dir = dir
        self.sequence = sequence
        self.jitter = kwargs.get("jitter", False)
        self.use_semantic = kwargs.get("use_semantic", False)
        self.use_panoptic = kwargs.get("use_panoptic", False)
        self.use_logits = kwargs.get("use_logits", True)
        self.filter_dynamic = kwargs.get("filter_dynamic", False)
        self.dynamic_classes = kwargs.get("dynamic_classes")

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
        self.train = kwargs.get("train", True)
        self.without_ground = kwargs.get("without_ground", False)
        self.loop_file = kwargs.get("loop_file") or "loop_GT"

        gt_file = os.path.join(dir, 'sequences', sequence, f'{self.loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

        self.superclass_mapper = SemanticSuperclassMapper(cfg_file=kwargs.get("superclass_cfg_file"))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(idx, self.dir, self.sequence,
                                                               use_semantic=self.use_semantic,
                                                               use_panoptic=self.use_panoptic,
                                                               jitter=self.jitter,
                                                               use_logits=self.use_logits,
                                                               filter_dynamic=self.filter_dynamic,
                                                               without_ground=self.without_ground,
                                                               dynamic_classes=self.dynamic_classes)
            anchor_pcd = torch.from_numpy(anchor_pcd)
        else:
            anchor_pcd = torch.from_numpy(get_velo(idx, self.dir, self.sequence, self.without_ground, self.jitter))

        sample = {'anchor': anchor_pcd}

        if self.use_panoptic or self.use_semantic:
            sample.update(unpack_logits(anchor_logits, self.use_logits, self.superclass_mapper, "anchor"))

        if self.train:
            x = self.poses[idx][0, 3]
            y = self.poses[idx][1, 3]
            z = self.poses[idx][2, 3]

            anchor_pose = torch.tensor([x, y, z])
            possible_match_pose = torch.tensor([0., 0., 0.])
            negative_pose = torch.tensor([0., 0., 0.])

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = idx
            negative_idx = idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_pose[0] = self.poses[idx][0, 3]
                possible_match_pose[1] = self.poses[idx][1, 3]
                possible_match_pose[2] = self.poses[idx][2, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 5 and idx == positive_idx:
                    positive_idx = i
                    cont += 1
                elif distance > 25 and idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
                    negative_idx = i
                    cont += 1
            if self.use_panoptic or self.use_semantic or self.filter_dynamic:
                positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                       use_semantic=self.use_semantic,
                                                                       use_panoptic=self.use_panoptic,
                                                                       jitter=self.jitter,
                                                                       use_logits=self.use_logits,
                                                                       filter_dynamic=self.filter_dynamic,
                                                                       dynamic_classes=self.dynamic_classes)
                positive_pcd = torch.from_numpy(positive_pcd)


                negative_pcd, negative_logits = get_velo_with_panoptic(negative_idx, self.dir, self.sequence,
                                                                       use_semantic=self.use_semantic,
                                                                       use_panoptic=self.use_panoptic,
                                                                       jitter=self.jitter,
                                                                       use_logits=self.use_logits,
                                                                       filter_dynamic=self.filter_dynamic,
                                                                       dynamic_classes=self.dynamic_classes)
                negative_pcd = torch.from_numpy(negative_pcd)

            else:
                positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))
                negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            sample['positive'] = positive_pcd
            sample['negative'] = negative_pcd

            if self.use_panoptic or self.use_semantic:
                sample.update(unpack_logits(positive_logits, self.use_logits, self.superclass_mapper, "positive"))
                sample.update(unpack_logits(negative_logits, self.use_logits, self.superclass_mapper, "negative"))
                sample.update(self.superclass_mapper.one_hot_maps)

        return sample


class KITTILoaderRGBPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, transforms=None):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        self.dir = dir
        self.sequence = sequence
        self.data = pykitti.odometry(dir, sequence)
        self.poses = pd.read_csv(poses)
        self.transform = transforms

    def __len__(self):
        return len(self.data.timestamps)

    def __getitem__(self, idx):

        anchor_img = self.data.get_cam3(idx)
        x = self.poses.iloc[idx, 1]
        y = self.poses.iloc[idx, 2]
        z = self.poses.iloc[idx, 3]

        anchor_pose = torch.tensor([x, y, z])
        possible_match_pose = torch.tensor([0., 0., 0.])
        negative_pose = torch.tensor([0., 0., 0.])

        indices = list(range(len(self.data.timestamps)))
        cont = 0
        positive_idx = idx
        negative_idx = idx
        while cont < 2:
            i = random.choice(indices)
            possible_match_pose[0] = self.poses.iloc[i, 1]
            possible_match_pose[1] = self.poses.iloc[i, 2]
            possible_match_pose[2] = self.poses.iloc[i, 3]
            distance = torch.norm(anchor_pose - possible_match_pose)
            if distance <= 5 and idx == positive_idx:
                positive_idx = i
                cont += 1
            elif distance > 25 and idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
                negative_idx = i
                cont += 1

        positive_img = self.data.get_cam3(positive_idx)
        negative_img = self.data.get_cam3(negative_idx)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        sample = {'anchor': anchor_img,
                  'positive': positive_img,
                  'negative': negative_img}

        return sample


class KITTILoader3DDictSingle(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, **kwargs):
        #without_ground=False, loop_file='loop_GT', jitter=False,
        #         use_semantic=False, use_panoptic=False, use_logits=True):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTILoader3DDictSingle, self).__init__()

        self.jitter = kwargs.get("jitter", False)
        self.dir = dir
        self.sequence = int(sequence)
        self.use_semantic = kwargs.get("use_semantic", False)
        self.use_panoptic = kwargs.get("use_panoptic", False)
        self.use_logits = kwargs.get("use_logits", False)
        self.filter_dynamic = kwargs.get("filter_dynamic", False)
        self.dynamic_classes = kwargs.get("dynamic_classes")
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
        self.without_ground = kwargs.get("without_ground", False)
        # self.loop_file = kwargs.get("loop_file") or "loop_GT"
        # gt_file = os.path.join(dir, 'sequences', sequence, f'{self.loop_file}.pickle')
        # with open(gt_file, 'rb') as f:
        #     self.loop_gt = pickle.load(f)
        # self.have_matches = []
        # for i in range(len(self.loop_gt)):
        #     self.have_matches.append(self.loop_gt[i]['idx'])

        self.superclass_mapper = SemanticSuperclassMapper(cfg_file=kwargs.get("superclass_cfg_file"))

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        if idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {idx} ")

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(idx, self.dir, self.sequence,
                                                               use_semantic=self.use_semantic,
                                                               use_panoptic=self.use_panoptic,
                                                               jitter=self.jitter,
                                                               use_logits=self.use_logits,
                                                               filter_dynamic=self.filter_dynamic,
                                                               without_ground=self.without_ground,
                                                               dynamic_classes=self.dynamic_classes)
            anchor_pcd = torch.from_numpy(anchor_pcd)

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]

            anchor_logits_dict = unpack_logits(anchor_logits, self.use_logits, self.superclass_mapper, "anchor",
                                               random_permute)

        else:
            anchor_pcd = torch.from_numpy(get_velo(idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]

        anchor_pose = self.poses[idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        # positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])

        # if self.use_panoptic or self.use_semantic or self.filter_dynamic:
        #     positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
        #                                                            use_semantic=self.use_semantic,
        #                                                            use_panotpic=self.use_panoptic,
        #                                                            jitter=self.jitter,
        #                                                            use_logits=self.use_logits,
        #                                                            filter_dynamic=self.filter_dynamic,
        #                                                            dynamic_classes=self.dynamic_classes)
        #     positive_pcd = torch.from_numpy(positive_pcd)
        #
        #     #Random permute points
        #     random_permute = torch.randperm(positive_pcd.shape[0])
        #     positive_pcd = positive_pcd[random_permute]
        #
        #     positive_logits_dict = unpack_logits(positive_logits, self.use_logits, self.superclass_mapper, "positive",
        #                                          random_permute)
        # else:
        #     positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))
        #
        #     #Random permute points
        #     random_permute = torch.randperm(positive_pcd.shape[0])
        #     positive_pcd = positive_pcd[random_permute]
        #
        #
        # if positive_idx >= len(self.poses):
        #     print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        # positive_pose = self.poses[positive_idx]
        # positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        r_anch = anchor_pose
        # r_pos = positive_pose
        r_anch = RT.npto_XYZRPY(r_anch)[3:]
        # r_pos = RT.npto_XYZRPY(r_pos)[3:]

        anchor_rot_torch = torch.tensor(r_anch.copy(), dtype=torch.float32)
        # positive_rot_torch = torch.tensor(r_pos.copy(), dtype=torch.float32)

        sample = {'anchor': anchor_pcd,
                  # 'positive': positive_pcd,
                  'sequence': self.sequence,
                  'anchor_pose': anchor_transl,
                  # 'positive_pose': positive_transl,
                  'anchor_rot': anchor_rot_torch,
                  # 'positive_rot': positive_rot_torch,
                  'anchor_idx': idx,
                  # 'positive_idx': positive_idx
                  }

        if self.use_panoptic or self.use_semantic:
            sample.update(anchor_logits_dict)
            # sample.update(positive_logits_dict)
            sample.update(self.superclass_mapper.one_hot_maps)

        return sample


class KITTILoader3DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, **kwargs):
        #without_ground=False, loop_file='loop_GT', jitter=False,
        #         use_semantic=False, use_panoptic=False, use_logits=True):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTILoader3DDictPairs, self).__init__()

        self.jitter = kwargs.get("jitter", False)
        self.dir = dir
        self.sequence = int(sequence)
        self.use_semantic = kwargs.get("use_semantic", False)
        self.use_panoptic = kwargs.get("use_panoptic", False)
        self.use_logits = kwargs.get("use_logits", False)
        self.filter_dynamic = kwargs.get("filter_dynamic", False)
        self.dynamic_classes = kwargs.get("dynamic_classes")
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
        self.without_ground = kwargs.get("without_ground", False)
        self.loop_file = kwargs.get("loop_file") or "loop_GT"
        gt_file = os.path.join(dir, 'sequences', sequence, f'{self.loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

        self.superclass_mapper = SemanticSuperclassMapper(cfg_file=kwargs.get("superclass_cfg_file"))

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(frame_idx, self.dir, self.sequence,
                                                               use_semantic=self.use_semantic,
                                                               use_panoptic=self.use_panoptic,
                                                               jitter=self.jitter,
                                                               use_logits=self.use_logits,
                                                               filter_dynamic=self.filter_dynamic,
                                                               without_ground=self.without_ground,
                                                               dynamic_classes=self.dynamic_classes)
            anchor_pcd = torch.from_numpy(anchor_pcd)

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]

            anchor_logits_dict = unpack_logits(anchor_logits, self.use_logits, self.superclass_mapper, "anchor",
                                               random_permute)

        else:
            anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                   use_semantic=self.use_semantic,
                                                                   use_panotpic=self.use_panoptic,
                                                                   jitter=self.jitter,
                                                                   use_logits=self.use_logits,
                                                                   filter_dynamic=self.filter_dynamic,
                                                                   without_ground=self.without_ground,
                                                                   dynamic_classes=self.dynamic_classes)
            positive_pcd = torch.from_numpy(positive_pcd)

            #Random permute points
            random_permute = torch.randperm(positive_pcd.shape[0])
            positive_pcd = positive_pcd[random_permute]

            positive_logits_dict = unpack_logits(positive_logits, self.use_logits, self.superclass_mapper, "positive",
                                                 random_permute)
        else:
            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(positive_pcd.shape[0])
            positive_pcd = positive_pcd[random_permute]


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
                  'positive_idx': positive_idx
                  }

        if self.use_panoptic or self.use_semantic:
            sample.update(anchor_logits_dict)
            sample.update(positive_logits_dict)
            sample.update(self.superclass_mapper.one_hot_maps)

        return sample


class KITTILoader3DDictTriplets(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, dir, sequence, poses, npoints, device, **kwargs):
        #without_ground=False,
        #         loop_file='loop_GT', hard_negative=False, jitter=False, use_semantic=False, use_panoptic=False):
        """

        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTILoader3DDictTriplets, self).__init__()

        self.jitter = kwargs.get("jitter", False)
        self.dir = dir
        self.sequence = int(sequence)
        self.hard_negative = kwargs.get("hard_negative", False)
        self.use_semantic = kwargs.get("use_semantic", False)
        self.use_panoptic = kwargs.get("use_panoptic", False)
        self.filter_dynamic = kwargs.get("filter_dynamic", False)
        self.dynamic_classes = kwargs.get("dynamic_classes")

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
        self.without_ground = kwargs.get("without_ground", False)
        self.loop_file = kwargs.get("loop_file") or "loop_GT"
        gt_file = os.path.join(dir, 'sequences', sequence, f'{self.loop_file}.pickle')
        with open(gt_file, 'rb') as f:
            self.loop_gt = pickle.load(f)
        self.have_matches = []
        for i in range(len(self.loop_gt)):
            self.have_matches.append(self.loop_gt[i]['idx'])

        self.use_logits = kwargs.get("use_logits", False)
        self.superclass_mapper = SemanticSuperclassMapper(cfg_file=kwargs.get("superclass_cfg_file"))

    def __len__(self):
        return len(self.loop_gt)

    def __getitem__(self, idx):
        frame_idx = self.loop_gt[idx]['idx']
        if frame_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            anchor_pcd, anchor_logits = get_velo_with_panoptic(frame_idx, self.dir, self.sequence,
                                                               use_semantic=self.use_semantic,
                                                               use_panotpic=self.use_panoptic,
                                                               jitter=self.jitter,
                                                               use_logits=self.use_logits,
                                                               filter_dynamic=self.filter_dynamic,
                                                               without_ground=self.without_ground,
                                                               dynamic_classes=self.dynamic_classes)
            anchor_pcd = torch.from_numpy(anchor_pcd)

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]
            anchor_logits_dict = unpack_logits(anchor_logits, self.use_logits, self.superclass_mapper, "anchor",
                                               random_permute)
        else:
            anchor_pcd = torch.from_numpy(get_velo(frame_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(anchor_pcd.shape[0])
            anchor_pcd = anchor_pcd[random_permute]

        anchor_pose = self.poses[frame_idx]
        anchor_transl = torch.tensor(anchor_pose[:3, 3], dtype=torch.float32)

        positive_idx = np.random.choice(self.loop_gt[idx]['positive_idxs'])
        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
                                                                   use_semantic=self.use_semantic,
                                                                   use_panoptic=self.use_panoptic,
                                                                   jitter=self.jitter,
                                                                   use_logits=self.use_logits,
                                                                   filter_dynamic=self.filter_dynamic,
                                                                   without_ground=self.without_ground,
                                                                   dynamic_classes=self.dynamic_classes)
            positive_pcd = torch.from_numpy(positive_pcd)

            #Random permute points
            random_permute = torch.randperm(positive_pcd.shape[0])
            positive_pcd = positive_pcd[random_permute]
            positive_logits_dict = unpack_logits(positive_logits, self.use_logits, self.superclass_mapper, "positive",
                                                 random_permute)
        else:
            positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(positive_pcd.shape[0])
            positive_pcd = positive_pcd[random_permute]

        if positive_idx >= len(self.poses):
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        positive_pose = self.poses[positive_idx]
        positive_transl = torch.tensor(positive_pose[:3, 3], dtype=torch.float32)

        if self.hard_negative and len(self.loop_gt[idx]['hard_idxs']) > 0:
            negative_idx = np.random.choice(list(self.loop_gt[idx]['hard_idxs']))
        else:
            negative_idx = np.random.choice(self.loop_gt[idx]['negative_idxs'])

        if self.use_panoptic or self.use_semantic or self.filter_dynamic:
            negative_pcd, negative_logits = get_velo_with_panoptic(negative_idx, self.dir, self.sequence,
                                                                   use_semantic=self.use_semantic,
                                                                   use_panoptic=self.use_panoptic,
                                                                   jitter=self.jitter,
                                                                   use_logits=self.use_logits,
                                                                   filter_dynamic=self.filter_dynamic,
                                                                   without_ground=self.without_ground,
                                                                   dynamic_classes=self.dynamic_classes)
            negative_pcd = torch.from_numpy(negative_pcd)

            #Random permute points
            random_permute = torch.randperm(negative_pcd.shape[0])
            negative_pcd = negative_pcd[random_permute]
            negative_logits_dict = unpack_logits(negative_logits, self.use_logits, self.superclass_mapper, "negative",
                                                 random_permute)
        else:
            negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.without_ground, self.jitter))

            #Random permute points
            random_permute = torch.randperm(negative_pcd.shape[0])
            negative_pcd = negative_pcd[random_permute]

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
                  'sequence': self.sequence,
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
            sample.update(anchor_logits_dict)
            sample.update(positive_logits_dict)
            sample.update(negative_logits_dict)
            sample.update(self.superclass_mapper.one_hot_maps)

        return sample

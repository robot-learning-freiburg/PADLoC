from enum import Enum
import os
import random

import h5py
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from datasets.KITTI360SuperClassMapper import SemanticSuperclassMapper
import utils.rotation_conversion as rt


class SamplePrefix(Enum):
    Anchor = "anchor"
    Positive = "positive"
    Negative = "negative"


def get_velo(idx, data_dir, sequence, prefix: SamplePrefix, pose=None, without_ground=False,
             use_semantic=False, use_panoptic=False, use_logits=False,
             superclass_mapper=None,
             jitter=False):

    if without_ground:
        velo_path = os.path.join(data_dir, 'data_3d_raw', sequence,
                                 'velodyne_no_ground', f'{idx:010d}.npy')
        with h5py.File(velo_path, 'r') as hf:
            scan = hf['PC'][:]
    else:
        velo_path = os.path.join(data_dir, 'data_3d_raw', sequence,
                                 'velodyne_points', 'data', f'{idx:010d}.bin')
        scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    if jitter:
        noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        scan = scan + noise

    scan = torch.from_numpy(scan.astype(np.float32))

    prefix = str(prefix.value)

    sample = {
        prefix: scan,
    }

    if pose is not None:
        tra = pose[:3, 3].clone().detach().type(torch.float32)
        rot = rt.npto_XYZRPY(pose)[3:]
        rot = torch.from_numpy(rot.astype(np.float32))

        sample[prefix + "_pose"] = tra
        sample[prefix + "_rot"] = rot

    if not (use_semantic or use_panoptic):
        return sample

    if use_logits:
        raise NotImplementedError

    labels_path = os.path.join(data_dir, "labels", sequence, f"{idx:010d}.bin")
    panoptic = np.fromfile(labels_path, dtype=np.int32)
    semantic = np.bitwise_and(panoptic, 0xFFFF).astype(np.int16)
    instance = np.right_shift(panoptic, 16).astype(np.uint16)

    sample[prefix + '_panoptic'] = torch.from_numpy(panoptic.astype(np.float32))
    sample[prefix + '_semantic'] = torch.from_numpy(semantic.astype(np.float32))
    sample[prefix + '_instance'] = torch.from_numpy(instance.astype(np.float32))

    if superclass_mapper is not None:
        supersem = superclass_mapper.get_superclass(semantic)
        sample[prefix + '_supersem'] = torch.from_numpy(supersem.astype(np.float32))

    return sample


def has_labels(data_dir, sequence, frame_id):
    label_path = os.path.join(data_dir, "labels", sequence, f"{frame_id:010d}.bin")
    return os.path.exists(label_path)


class KITTI3603DPoses(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, data_dir, sequence, without_ground=False,
                 train=True, loop_file='loop_GT', jitter=False,
                 use_semantic=False, use_panoptic=False, use_logits=False,
                 **_):
        """
        :param data_dir: directory where dataset is located
        :param sequence: KITTI sequence
        """
        super(KITTI3603DPoses, self).__init__()

        self.dir = data_dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.jitter = jitter
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        self.use_logits = use_logits
        calib_file = os.path.join(data_dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        pose_path = os.path.join(data_dir, 'data_poses', sequence, 'cam0_to_world.txt')
        loaded_poses = np.loadtxt(pose_path)

        self.frames_with_gt = loaded_poses[:, 0].astype(int)

        # Check if frame has labels
        if self.use_panoptic or self.use_semantic:
            with_labels = [i for i, f in enumerate(self.frames_with_gt) if has_labels(self.dir, self.sequence, f)]
            self.frames_with_gt = self.frames_with_gt[with_labels]
            loaded_poses = loaded_poses[with_labels]

        self.frame_to_idx = {f: i for i, f in enumerate(self.frames_with_gt)}
        poses = np.zeros((loaded_poses.shape[0], 4, 4))
        poses[:, :3, :] = loaded_poses[:, 1:13].reshape((-1, 3, 4))
        poses[:, 3, 3] = 1.0
        poses = torch.from_numpy(poses) @ cam0_to_velo.inverse()
        self.poses = poses
        self.train = train
        self.without_ground = without_ground

        gt_file = os.path.join(data_dir, 'data_poses', sequence, f'{loop_file}.pickle')

        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            self.loop_gt = [{"idx": elem["idx"], "positive_idxs": elem["positive_idxs"]} for elem in temp]

        if self.use_panoptic or self.use_semantic:
            # Filter not only if they form a loop, but also if they have panoptic labels
            loop_labeled_gt = []

            for frame in self.loop_gt:
                idx = frame["idx"]

                if not has_labels(data_dir, self.sequence, idx):
                    continue

                labeled_pos_idxs = [f for f in frame["positive_idxs"] if has_labels(data_dir, self.sequence, f)]
                if not labeled_pos_idxs:
                    continue

                loop_labeled_gt.append({"idx": idx, "positive_idxs": labeled_pos_idxs})

            self.loop_gt = loop_labeled_gt

        self.have_matches = [frame["idx"] for frame in self.loop_gt]

        self.superclass_mapper = SemanticSuperclassMapper()

    def __len__(self):
        return len(self.frames_with_gt)

    def get_velo(self, idx, prefix):
        return get_velo(idx=idx, data_dir=self.dir, sequence=self.sequence, pose=None,
                        prefix=prefix, without_ground=self.without_ground,
                        use_semantic=self.use_semantic, use_panoptic=self.use_panoptic, use_logits=self.use_logits,
                        superclass_mapper=self.superclass_mapper, jitter=self.jitter)

    def __getitem__(self, idx):
        frame_idx = self.frames_with_gt[idx]
        sample = self.get_velo(idx=frame_idx, prefix=SamplePrefix.Anchor)

        if self.train:
            anchor_pose = self.poses[idx, :3, 3]

            indices = list(range(len(self.poses)))
            cont = 0
            positive_idx = frame_idx
            negative_idx = frame_idx
            while cont < 2:
                i = random.choice(indices)
                possible_match_idx = self.frames_with_gt[i]
                possible_match_pose = self.poses[i, :3, 3]
                distance = torch.norm(anchor_pose - possible_match_pose)
                if distance <= 5 and frame_idx == positive_idx and (
                        (self.use_panoptic or self.use_semantic) and
                        has_labels(self.dir, self.sequence, possible_match_idx)
                ):
                    positive_idx = possible_match_idx
                    cont += 1
                elif distance > 25 and frame_idx == negative_idx and (
                        (self.use_panoptic or self.use_semantic) and
                        has_labels(self.dir, self.sequence, possible_match_idx)
                ):  # 1.5 < dist < 2.5 -> unknown
                    negative_idx = possible_match_idx
                    cont += 1

            pos_sample = self.get_velo(idx=positive_idx, prefix=SamplePrefix.Positive)
            sample.update(pos_sample)

            neg_sample = self.get_velo(idx=negative_idx, prefix=SamplePrefix.Negative)
            sample.update(neg_sample)

        if self.use_panoptic or self.use_semantic:
            sample.update(self.superclass_mapper.one_hot_maps)

        sample["sequence"] = self.sequence_int

        return sample


class KITTI3603DDictPairs(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, data_dir, sequence, without_ground=False, loop_file='loop_GT', jitter=False,
                 use_semantic=False, use_panoptic=False, use_logits=False,
                 **_):
        """
        :param dataset: directory where dataset is located
        :param sequence: KITTI sequence
        :param poses: csv with data poses
        """

        super(KITTI3603DDictPairs, self).__init__()

        self.jitter = jitter
        self.dir = data_dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        self.use_logits = use_logits
        calib_file = os.path.join(data_dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        pose_path = os.path.join(data_dir, 'data_poses', sequence, 'cam0_to_world.txt')
        loaded_poses = np.loadtxt(pose_path)

        self.frames_with_gt = loaded_poses[:, 0].astype(int)
        self.frame_to_idx = {f: i for i, f in enumerate(self.frames_with_gt)}

        poses = np.zeros((loaded_poses.shape[0], 4, 4))
        poses[:, :3, :] = loaded_poses[:, 1:13].reshape((-1, 3, 4))
        poses[:, 3, 3] = 1.0
        poses = torch.from_numpy(poses) @ cam0_to_velo.inverse()
        self.poses = poses

        self.without_ground = without_ground
        gt_file = os.path.join(data_dir, 'data_poses', sequence, f'{loop_file}.pickle')

        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            self.loop_gt = [{"idx": elem["idx"], "positive_idxs": elem["positive_idxs"]} for elem in temp]

        if self.use_panoptic or self.use_semantic:
            # Filter not only if they form a loop, but also if they have panoptic labels
            loop_labeled_gt = []

            for frame in self.loop_gt:
                idx = frame["idx"]

                if not has_labels(data_dir, self.sequence, idx):
                    continue

                labeled_pos_idxs = [f for f in frame["positive_idxs"] if has_labels(data_dir, self.sequence, f)]
                if not labeled_pos_idxs:
                    continue

                loop_labeled_gt.append({"idx": idx, "positive_idxs": labeled_pos_idxs})

            self.loop_gt = loop_labeled_gt

        self.have_matches = [frame["idx"] for frame in self.loop_gt]

        if self.use_panoptic or self.use_semantic:
            self.have_matches = [f for f in self.have_matches
                                 if os.path.exists(os.path.join(data_dir, "labels", sequence, f"{f:010d}.bin"))]

        self.superclass_mapper = SemanticSuperclassMapper()

    def __len__(self):
        return len(self.loop_gt)

    def get_velo(self, frame_id, prefix):
        return get_velo(idx=frame_id, data_dir=self.dir, sequence=self.sequence,
                        pose=self.poses[self.frame_to_idx[frame_id]],
                        prefix=prefix, without_ground=self.without_ground,
                        use_semantic=self.use_semantic, use_panoptic=self.use_panoptic, use_logits=self.use_logits,
                        superclass_mapper=self.superclass_mapper, jitter=self.jitter)

    def __getitem__(self, idx):
        anc_frame = self.loop_gt[idx]
        frame_idx = anc_frame['idx']
        if frame_idx not in self.frames_with_gt:
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        sample = self.get_velo(frame_id=frame_idx, prefix=SamplePrefix.Anchor)

        positive_idx = np.random.choice(anc_frame['positive_idxs'])

        if positive_idx not in self.frames_with_gt:
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")

        pos_sample = self.get_velo(frame_id=positive_idx, prefix=SamplePrefix.Positive)
        sample.update(pos_sample)

        if self.use_panoptic or self.use_semantic:
            sample.update(self.superclass_mapper.one_hot_maps)

        sample["sequence"] = self.sequence_int

        return sample


class KITTI3603DDictTriplets(Dataset):
    """KITTI ODOMETRY DATASET"""

    def __init__(self, data_dir, sequence, without_ground=False,
                 loop_file='loop_GT', hard_negative=False, jitter=False,
                 use_semantic=False, use_panoptic=False, use_logits=False,
                 **_):
        """
        :param data_dir: directory where dataset is located
        :param sequence: KITTI sequence
        """

        super(KITTI3603DDictTriplets, self).__init__()

        self.jitter = jitter
        self.dir = data_dir
        self.sequence = sequence
        self.sequence_int = int(sequence[-8:-5])
        self.hard_negative = hard_negative
        self.use_semantic = use_semantic
        self.use_panoptic = use_panoptic
        self.use_logits = use_logits
        calib_file = os.path.join(data_dir, 'calibration', 'calib_cam_to_velo.txt')
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                data = np.array([float(x) for x in line.split()])

        cam0_to_velo = np.reshape(data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        cam0_to_velo = torch.tensor(cam0_to_velo)

        pose_path = os.path.join(data_dir, 'data_poses', sequence, 'cam0_to_world.txt')
        loaded_poses = np.loadtxt(pose_path)

        self.frames_with_gt = loaded_poses[:, 0].astype(int)
        self.frame_to_idx = {f: i for i, f in enumerate(self.frames_with_gt)}

        poses = np.zeros((loaded_poses.shape[0], 4, 4))
        poses[:, :3, :] = loaded_poses[:, 1:13].reshape((-1, 3, 4))
        poses[:, 3, 3] = 1.0
        poses = torch.from_numpy(poses) @ cam0_to_velo.inverse()
        self.poses = poses

        self.without_ground = without_ground
        gt_file = os.path.join(data_dir, 'data_poses', sequence, f'{loop_file}.pickle')

        with open(gt_file, 'rb') as f:
            temp = pickle.load(f)
            self.loop_gt = [{"idx": elem["idx"], "positive_idxs": elem["positive_idxs"]} for elem in temp]

        if self.use_panoptic or self.use_semantic:
            # Filter not only if they form a loop, but also if they have panoptic labels
            loop_labeled_gt = []

            for frame in self.loop_gt:
                idx = frame["idx"]

                if not has_labels(data_dir, self.sequence, idx):
                    continue

                labeled_pos_idxs = [f for f in frame["positive_idxs"] if has_labels(data_dir, self.sequence, f)]
                if not labeled_pos_idxs:
                    continue

                # TODO: Also verify the negative samples

                loop_labeled_gt.append({"idx": idx, "positive_idxs": labeled_pos_idxs})

            self.loop_gt = loop_labeled_gt

        self.have_matches = [frame["idx"] for frame in self.loop_gt]

        self.superclass_mapper = SemanticSuperclassMapper()

    def __len__(self):
        return len(self.loop_gt)

    def get_velo(self, frame_id, prefix):
        return get_velo(idx=frame_id, data_dir=self.dir, sequence=self.sequence,
                        pose=self.poses[self.frame_to_idx[frame_id]],
                        prefix=prefix, without_ground=self.without_ground,
                        use_semantic=self.use_semantic, use_panoptic=self.use_panoptic, use_logits=self.use_logits,
                        superclass_mapper=self.superclass_mapper, jitter=self.jitter)

    def __getitem__(self, idx):
        anc_frame = self.loop_gt[idx]
        frame_idx = anc_frame['idx']
        if frame_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, frame idx {frame_idx} ")

        sample = self.get_velo(frame_id=frame_idx, prefix=SamplePrefix.Anchor)

        positive_idx = np.random.choice(anc_frame['positive_idxs'])
        if positive_idx not in self.poses:
            print(f"ERRORE: sequence {self.sequence}, positive idx {positive_idx} ")
        pos_sample = self.get_velo(frame_id=positive_idx, prefix=SamplePrefix.Positive)
        sample.update(pos_sample)

        if self.hard_negative and len(anc_frame['hard_idxs']) > 0:
            negative_idx = np.random.choice(list(anc_frame['hard_idxs']))
        else:
            negative_idx = np.random.choice(anc_frame['negative_idxs'])

        neg_sample = self.get_velo(frame_id=negative_idx, prefix=SamplePrefix.Negative)
        sample.update(neg_sample)

        if self.use_panoptic or self.use_semantic:
            sample.update(self.superclass_mapper.one_hot_maps)

        sample["sequence"] = self.sequence_int

        return sample

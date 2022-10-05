import argparse
import os
import json
import pickle

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from scipy.stats import circmean
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

from evaluation_comparison.metrics.registration import batch_icp_registration, batch_ransac_registration,\
    get_ransac_features
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from models.get_models import load_model
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from utils.data import merge_inputs
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy, get_rt_matrix
from utils.rotation_conversion import npto_XYZRPY
from utils.tools import set_seed


torch.backends.cudnn.benchmark = True


def generate_pairs(dataset, positive_distance=4.):
    test_pair_idxs = []
    index = faiss.IndexFlatL2(3)
    poses = dataset.poses
    if isinstance(poses, list):
        poses = np.stack(dataset.poses).copy()
    elif isinstance(poses, torch.Tensor):
        poses = poses.detach().cpu().numpy()
    # Faiss Index only takes Float32
    poses = poses.astype(np.float32)
    index.add(poses[:50, :3, 3].copy())
    num_frames_with_loop = 0
    num_frames_with_reverse_loop = 0
    for i in tqdm(range(100, len(dataset.poses))):
        current_pose = poses[i:i + 1, :3, 3].copy()
        index.add(poses[i - 50:i - 49, :3, 3].copy())
        lims, D, I = index.range_search(current_pose, positive_distance ** 2)
        for j in range(lims[0], lims[1]):
            if j == 0:
                num_frames_with_loop += 1
                yaw_diff = npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
                yaw_diff = yaw_diff % (2 * np.pi)
                if 0.79 <= yaw_diff <= 5.5:
                    num_frames_with_reverse_loop += 1

            test_pair_idxs.append([I[j], i])
    test_pair_idxs = np.array(test_pair_idxs)

    return poses, test_pair_idxs


def get_dataset(dataset, data, sequence, device, exp_cfg):
    if dataset == 'kitti':
        dataset_for_recall = KITTILoader3DPoses(data, sequence,
                                                os.path.join(data, 'sequences',
                                                             sequence, 'poses.txt'),
                                                exp_cfg['num_points'], device, train=False,
                                                without_ground=exp_cfg['without_ground'],
                                                loop_file=exp_cfg['loop_file'])
    else:
        dataset_for_recall = KITTI3603DPoses(data, sequence,
                                             train=False,
                                             without_ground=exp_cfg['without_ground'], loop_file='loop_GT_4m_noneg')

    return dataset_for_recall


class BatchSamplePairs(BatchSampler):

    def __init__(self, data_source, pairs, batch_size):
        # super(BatchSamplePairs, self).__init__(batch_size, True)
        self.pairs = pairs
        self.batch_size = batch_size
        self.count = 0
        batches_pairs = [pairs[i:i+batch_size//2] for i in range(0, len(pairs), batch_size//2)]
        self.batches_idxs = [[p[i] for i in [0, 1] for p in batch_pair] for batch_pair in batches_pairs]

    def __len__(self):
        return len(self.batches_idxs)

    def __iter__(self):
        for b in self.batches_idxs:
            yield b


def preprocess_sample(sample, model, exp_cfg, device):
    with torch.no_grad():

        anchor_list = []
        for i in range(len(sample['anchor'])):
            anchor = sample['anchor'][i].to(device)

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(),
                                                   exp_cfg['num_points'])
                a = anchor_set[0, :].long()
                anchor_i = anchor[a]
                anchor_list.append(anchor_i[:, :3].unsqueeze(0))
            else:
                anchor_i = anchor
                anchor_list.append(model.backbone.prepare_input(anchor_i))

        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor = torch.cat(anchor_list)
            model_in = anchor
            # Normalize between [-1, 1], more or less
            # model_in = model_in / 100.
            if exp_cfg['3D_net'] == 'RandLANet':
                model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
        else:
            model_in = KittiDataset.collate_batch(anchor_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

    return model_in


def get_tra_rot(transformation):
    tra_rot = [mat2xyzrpy(t) for t in transformation]
    tra = [torch.linalg.norm(t[:3]).detach().cpu().item() for t in tra_rot]
    yaw = [torch.rad2deg(t[-1]).detach().cpu().item() for t in tra_rot]
    return tra, yaw


def eval_batch(model, exp_cfg, sample, batch_gt_tf_p2a, device, do_ransac=False, do_icp=False):
    model.eval()
    with torch.no_grad():

        model_in = preprocess_sample(sample, model, exp_cfg, device)

        torch.cuda.synchronize()
        batch_dict = model(model_in, metric_head=True)
        torch.cuda.synchronize()

        if not exp_cfg['rot_representation'].startswith('6dof'):
            raise NotImplementedError

        pred_tf_a2p = batch_dict['transformation']
        homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(pred_tf_a2p.shape[0], 1, 1).to(pred_tf_a2p.device)
        pred_tf_a2p = torch.cat((pred_tf_a2p, homogeneous), dim=1)

        coords = batch_dict["point_coords"].view(batch_dict['batch_size'], -1, 4)

        if do_ransac:
            feats = get_ransac_features(batch_dict, model=model)

            pred_tf_a2p = batch_ransac_registration(batch_coords=coords, batch_feats=feats,
                                                    batch_size=batch_dict["batch_size"]).to(device=device,
                                                                                            dtype=torch.float32)

        if do_icp:
            pred_tf_a2p = batch_icp_registration(batch_coords=coords, batch_size=batch_dict["batch_size"],
                                                 initial_transformations=pred_tf_a2p)

        pred_tra, pred_yaw = get_tra_rot(pred_tf_a2p)

        rel_pose_err = pred_tf_a2p @ batch_gt_tf_p2a
        err_tra, err_yaw = get_tra_rot(rel_pose_err)

        batch_size = batch_dict["batch_size"]

        stats = dict(
            anc_desc=batch_dict["out_embedding"][:batch_size//2].detach().cpu().numpy(),
            pos_desc=batch_dict["out_embedding"][batch_size//2:].detach().cpu().numpy(),
            pair_pred_yaw_deg=pred_yaw,
            pair_err_yaw_deg=err_yaw,
            pair_pred_tra=pred_tra,
            pair_err_tra=err_tra,
        )

        return stats


def make_serializable(value):

    if isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_serializable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def save_json(stats, path):
    if path is None:
        return

    if not path:
        return

    print("Serializing and Saving to ", path)
    save_dict = make_serializable(stats)

    with open(path, "w") as f:
        json.dump(save_dict, f)


def save_pickle(stats, path):
    if path is None:
        return

    if not path:
        return

    print("Saving to ", path)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def save_stats(stats, path):
    if path is None:
        return

    if not path:
        return

    final_ext = path.split(".")[-1].lower()

    ext_save = {
        "json": save_json,
        "pickle": save_pickle,
    }

    if final_ext not in ext_save:
        raise TypeError(f"Unsupported file extension {final_ext}. Supported types: {ext_save.keys()}.")

    ext_save[final_ext](stats, path)


def main_process(gpu, weights_path, dataset, data, batch_size=8, sequence=None, loop_file=None,
                 seed=0,
                 do_ransac=False, do_icp=False,
                 save_path=None):

    set_seed(seed)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    override_cfg = dict(
        batch_size=batch_size,
    )
    if loop_file is not None:
        override_cfg["loop_file"] = loop_file

    if sequence is None:
        default_sequences = {
            "kitti": "08",
            "kitti360": "2013_05_28_drive_0009_sync"
        }

        sequence = default_sequences[dataset]

        override_cfg['test_sequence'] = sequence
        override_cfg['sinkhorn_iter'] = 5

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg, is_training=False)

    if not (exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.):
        print("No losses configured for the pose. Exiting")
        return

    dataset_for_recall = get_dataset(dataset, data=data, sequence=sequence, device=device, exp_cfg=exp_cfg)

    print("Generating Loop-Closure pairs")
    poses, test_pair_idxs = generate_pairs(dataset_for_recall, positive_distance=4.0)

    batch_sampler = BatchSamplePairs(dataset_for_recall, test_pair_idxs, batch_size)

    recall_loader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                                # batch_size=exp_cfg['batch_size'],
                                                num_workers=2,
                                                # sampler=sampler,
                                                batch_sampler=batch_sampler,
                                                # worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                pin_memory=True)

    model = model.to(device)

    yaw_deg_values = list(range(0, 200, 20))

    # For debugging purposes
    # yaw_deg_values = [0, 20, 40]

    stats = dict(
        anc_idxs=[],
        pos_idxs=[],
        pair_pred_yaw_deg={a: [] for a in yaw_deg_values},
        pair_err_yaw_deg={a: [] for a in yaw_deg_values},
        pair_pred_tra={a: [] for a in yaw_deg_values},
        pair_err_tra={a: [] for a in yaw_deg_values},
        anc_desc={a: [] for a in yaw_deg_values},
        pos_desc={a: [] for a in yaw_deg_values},
    )

    print("\nEvaluating Model on loop-closure pairs.")
    for batch_idx, (sample, frames) in enumerate(tqdm(zip(recall_loader, batch_sampler.batches_idxs),
                                                      total=len(recall_loader))):
        # For debugging purposes
        # if batch_idx == 2:
        #     break

        batch_poses = poses[frames]

        stats["anc_idxs"].extend(frames[:batch_size//2])
        stats["pos_idxs"].extend(frames[batch_size//2:])

        gt_tf_p2a = np.linalg.inv(batch_poses[:batch_size//2]) @ batch_poses[batch_size//2:]
        gt_tf_p2a = torch.from_numpy(gt_tf_p2a).float().to(device)

        for yaw_deg in yaw_deg_values:

            aug_tra_yaw = torch.tensor([0., 0., 0.]).cuda()
            aug_rot_yaw = torch.tensor([0., 0., np.deg2rad(-yaw_deg)]).cuda()
            aug_tf_yaw = get_rt_matrix(transl=aug_tra_yaw, rot=aug_rot_yaw, rot_parmas="xyz").float().to(device)

            new_gt_tf_p2a = torch.linalg.inv(aug_tf_yaw)
            new_gt_tf_p2a = new_gt_tf_p2a.repeat(batch_size // 2, 1, 1)
            new_gt_tf_p2a[:, :3, 3] = gt_tf_p2a[:, :3, 3]

            gt_rot_p2a = gt_tf_p2a.clone()
            gt_rot_p2a[:, :3, 3] = 0
            aug_tf = aug_tf_yaw @ gt_rot_p2a

            rotated_sample = {
                "anchor": [s.to(device) for s in sample["anchor"][:batch_size//2]]
            }
            # Transform each Positive Point Cloud
            for tf, pc in zip(aug_tf, sample["anchor"][batch_size//2:]):
                homogeneous_pc = pc.clone().to(device)
                homogeneous_pc[:, 3] = 1.
                homogeneous_pc = tf @ homogeneous_pc.T
                homogeneous_pc = homogeneous_pc.T
                homogeneous_pc[:, 3] = pc[:, 3]
                rotated_sample["anchor"].append(homogeneous_pc)

            batch_angle_stats = eval_batch(model, exp_cfg, rotated_sample, new_gt_tf_p2a, device,
                                           do_ransac=do_ransac, do_icp=do_icp)

            for k, v in batch_angle_stats.items():
                stats[k][yaw_deg].extend(v)

    print(weights_path)
    print(exp_cfg['test_sequence'])

    stats["pair_mean_abs_err_yaw_deg"] = {a: circmean(np.abs(v), high=360)
                                          for a, v in stats["pair_err_yaw_deg"].items()}
    stats["pair_mean_abs_err_tra"] = {a: np.mean(np.abs(v)) for a, v in stats["pair_err_tra"].items()}
    stats["desc_ancpos_l2_dist"] = {a: np.linalg.norm(np.stack(stats["anc_desc"][a]) - np.stack(stats["pos_desc"][a]),
                                                      axis=1) for a in yaw_deg_values}

    stats["mean_desc_ancpos_l2_dist"] = {a: v.mean() for a, v in stats["desc_ancpos_l2_dist"].items()}

    stats["desc_pos0posth_l2_dist"] = {a: np.linalg.norm(
        np.stack(stats["pos_desc"][0]) - np.stack(stats["pos_desc"][a]), axis=1) for a in yaw_deg_values[1:]}

    stats["mean_desc_pos0posth_l2_dist"] = {a: v.mean() for a, v in stats["desc_pos0posth_l2_dist"].items()}

    save_pickle(stats, save_path)

    return stats


def cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="/home/cattaneo/Datasets/KITTI",
                        help="dataset directory")
    parser.add_argument("--weights_path", default="/home/cattaneo/checkpoints/deep_lcd")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="kitti")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop_file", type=str, default=None)
    parser.add_argument("--do_ransac", action="store_true")
    parser.add_argument("--do_icp", action="store_true")
    parser.add_argument("--save_path", default="")

    args = parser.parse_args()

    return vars(args)


if __name__ == '__main__':
    main_process(0, **cli_args())

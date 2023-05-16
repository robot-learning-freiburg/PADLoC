#! /usr/bin/env python

import argparse
from datasets.NCLTDataset import NCLTDataset
import os
from pathlib import Path
import json
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from sklearn.neighbors import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.Freiburg import FreiburgDataset
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses

from evaluation_comparison.metrics.detection import compute_pr_fp, compute_pr_fn, compute_pr_pairs,\
    compute_ap_from_pr, compute_ep_from_pr, compute_f1_from_pr,\
    generate_pairs, load_pairs_file
from models.get_models import load_model
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from utils.data import merge_inputs, Timer
from utils.tools import set_seed
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

torch.backends.cudnn.benchmark = True


def prepare_input(model, samples, exp_cfg, device):
    anchor_list = []
    for point_cloud in samples:
        if exp_cfg["3D_net"] != "PVRCNN":
            anchor_set = furthest_point_sample(point_cloud[:, 0:3].unsqueeze(0).contiguous(), exp_cfg["num_points"])
            a = anchor_set[0, :].long()
            anchor_i = point_cloud[a]
        else:
            anchor_i = point_cloud

        if exp_cfg["3D_net"] != "PVRCNN":
            anchor_list.append(anchor_i[:, :3].unsqueeze(0))
        else:
            anchor_list.append(model.backbone.prepare_input(anchor_i))
            del anchor_i

    if exp_cfg["3D_net"] != "PVRCNN":
        point_cloud = torch.cat(tuple(anchor_list), 0)
        model_in = point_cloud

        if exp_cfg["3D_net"] == "RandLANet":
            model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
    else:
        model_in = KittiDataset.collate_batch(anchor_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)
    return model_in


def geometric_verification(model, dataset, id_query, id_candidate, device, exp_cfg):
    model.eval()
    with torch.no_grad():

        sample_query = dataset.__getitem__(id_query)
        sample_candidate = dataset.__getitem__(id_candidate)
        query_pc = sample_query["anchor"].to(device)
        candidate_pc = sample_candidate["anchor"].to(device)

        model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

        transformation = batch_dict["transformation"]
        homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
        transformation = torch.cat((transformation, homogeneous), dim=1)

        query_intensity = query_pc[:, -1].clone()
        query_pc = query_pc.clone()
        query_pc[:, -1] = 1.
        transformed_query_pc = (transformation[0] @ query_pc.T).T
        transformed_query_pc[:, -1] = query_intensity

        model_in = prepare_input(model, [transformed_query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=False, compute_embeddings=True)

        emb = batch_dict["out_embedding"]
        if exp_cfg["norm_embeddings"]:
            emb = emb / emb.norm(dim=1, keepdim=True)

    return (emb[0] - emb[1]).norm().detach().cpu()


def geometric_verification2(model, dataset, id_query, id_candidate, device, exp_cfg):
    model.eval()
    with torch.no_grad():

        sample_query = dataset.__getitem__(id_query)
        sample_candidate = dataset.__getitem__(id_candidate)
        query_pc = sample_query["anchor"].to(device)
        candidate_pc = sample_candidate["anchor"].to(device)

        model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

    return batch_dict["transport"].sum(-1).sum().detach().cpu()


class SamplePairs(Sampler):

    def __init__(self, data_source, pairs):
        super(SamplePairs, self).__init__(data_source)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return [self.pairs[i, 0] for i in range(len(self.pairs))]


class BatchSamplePairs(BatchSampler):

    def __init__(self, data_source, pairs, batch_size):
        # super(BatchSamplePairs, self).__init__(batch_size, True)
        self.data_source = data_source
        self.pairs = pairs
        self.batch_size = batch_size
        self.count = 0

    def __len__(self):
        return 2*len(self.pairs)

    def __iter__(self):
        self.count = 0
        while 2*self.count + self.batch_size < 2*len(self.pairs):
            current_batch = []
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch
            self.count += self.batch_size//2
        if 2*self.count < 2*len(self.pairs):
            diff = 2*len(self.pairs)-2*self.count
            current_batch = []
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch


def load_poses(
        dataset: str,
        data: str,
        sequence: str,
        device,
        num_points: int,
        use_semantic: bool,
        use_panoptic: bool,
        without_ground: bool,
        loop_file: str,
        z_offset: float,
):
    if dataset == "kitti":
        dataset_for_recall = KITTILoader3DPoses(data, sequence,
                                                os.path.join(data, "sequences", sequence, "poses.txt"),
                                                num_points, device, train=False,
                                                use_semantic=use_semantic,
                                                use_panoptic=use_panoptic,
                                                without_ground=without_ground,
                                                loop_file=loop_file)
    elif dataset == "kitti360":
        dataset_for_recall = KITTI3603DPoses(data, sequence,
                                             train=False,
                                             without_ground=without_ground, loop_file="loop_GT_4m_noneg")
    elif dataset == "freiburg":
        dataset_for_recall = FreiburgDataset(data, without_ground=without_ground, z_offset=z_offset)
    elif dataset == "nclt":
        dataset_for_recall = NCLTDataset(data, sequence)
    else:
        raise ValueError(f"Invalid dataset {dataset}.")

    map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])

    return dataset_for_recall, map_tree_poses


def do_inference(
        device,
        weights_path: str,
        dataset: str,
        data: str,
        time_net: Timer,
        batch_size: int = 6,
        num_iters: int = 1,
        sequence: Optional[str] = None,
        save_path: Optional[str] = None,
        loop_file: Optional[str] = None,
        z_offset: float = 0.283,
):

    override_cfg = dict(
        batch_size=batch_size,
    )

    if loop_file is not None:
        override_cfg["loop_file"] = loop_file

    if sequence is None:
        if dataset == "kitti":
            override_cfg["test_sequence"] = "08"

        elif dataset == "nclt":
            override_cfg["test_sequence"] = "2013-04-05"

        elif dataset == "kitti360":
            override_cfg["test_sequence"] = "2013_05_28_drive_0002_sync"

        if dataset != "freiburg":
            sequences_validation = [override_cfg["test_sequence"]]
            sequence = sequences_validation[0]

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg)

    dataset_for_recall, map_tree_poses = load_poses(dataset, data, sequence, device,
                                                    exp_cfg["num_points"],
                                                    exp_cfg["use_semantic"], exp_cfg["use_panoptic"],
                                                    exp_cfg["without_ground"], exp_cfg["loop_file"], z_offset)

    map_loader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                             batch_size=exp_cfg["batch_size"],
                                             num_workers=2,
                                             shuffle=False,
                                             collate_fn=merge_inputs,
                                             pin_memory=True)

    model.train()
    model = model.to(device)

    emb_list_map = []
    rot_errors = []
    transl_errors = []

    for i in range(num_iters):
        rot_errors.append([])
        transl_errors.append([])

    for batch_idx, sample in enumerate(tqdm(map_loader)):

        model.eval()
        torch.cuda.synchronize()
        time_net.tic()
        with torch.no_grad():

            anchor_list = []
            for i in range(len(sample["anchor"])):
                anchor = sample["anchor"][i].to(device)

                if exp_cfg["3D_net"] != "PVRCNN":
                    anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg["num_points"])
                    a = anchor_set[0, :].long()
                    anchor_i = anchor[a]
                else:
                    anchor_i = anchor

                if exp_cfg["3D_net"] != "PVRCNN":
                    anchor_list.append(anchor_i[:, :3].unsqueeze(0))
                else:
                    if exp_cfg["use_semantic"] or exp_cfg["use_panoptic"]:
                        anchor_i = torch.cat((anchor_i, sample["anchor_logits"][i].to(device)), dim=1)
                    anchor_list.append(model.backbone.prepare_input(anchor_i))
                    del anchor_i

            if exp_cfg["3D_net"] != "PVRCNN":
                anchor = torch.cat(anchor_list)
                model_in = anchor
                if exp_cfg["3D_net"] == "RandLANet":
                    model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
            else:
                model_in = KittiDataset.collate_batch(anchor_list)
                for key, val in model_in.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

            emb = batch_dict["out_embedding"]
            emb_list_map.append(emb)

        torch.cuda.synchronize()
        time_net.toc(call_inc=batch_dict["batch_size"] // 2)

    emb_list_map = torch.cat(emb_list_map).cpu().numpy()

    emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
    pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)
    if save_path:
        print(f"Saving Pairwise Distances to {save_path}")
        np.savez(save_path, pair_dist)

    return dataset_for_recall, map_tree_poses, pair_dist


def main_process(
        gpu: int,
        weights_path: str,
        dataset: str,
        data: str,
        seed: int = 0,
        batch_size: int = 6,
        num_iters: int = 1,
        sequence: Optional[str] = None,
        save_path: Optional[str] = None,
        stats_save_path: Optional[str] = None,
        loop_file: Optional[str] = None,
        z_offset: float = 0.283,
        save_times_path: Optional[str] = None,
        force_inference: bool = False,
        pos_distance: float = 4.,
        neg_frames: int = 50,
        start_frame: int = 100,
        not_distance: bool = False,
        ignore_last: bool = False
) -> None:

    set_seed(seed)

    time_net = Timer()

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    run_inference = True
    if save_path is not None and not force_inference:
        if os.path.isfile(save_path):
            run_inference = False

    if run_inference:
        print("Running inference")
        dataset_for_recall, map_tree_poses, pair_dist = do_inference(device=device, weights_path=weights_path,
                                                                     dataset=dataset, data=data, time_net=time_net,
                                                                     batch_size=batch_size, num_iters=num_iters,
                                                                     sequence=sequence, save_path=save_path,
                                                                     loop_file=loop_file, z_offset=z_offset)
    else:
        print(f"Loading pairwise descriptor distances from {save_path}")
        dataset_for_recall, map_tree_poses = load_poses(dataset=dataset, data=data, sequence=sequence, device=device,
                                                        num_points=0, use_semantic=False, use_panoptic=False,
                                                        without_ground=False, loop_file=loop_file, z_offset=z_offset)
        pair_dist = load_pairs_file(save_path)

    poses = np.stack(dataset_for_recall.poses)
    pairs = generate_pairs(pair_dist=pair_dist, poses=poses, map_tree_poses=map_tree_poses,
                           positive_distance=pos_distance,
                           negative_frames=neg_frames, start_frame=start_frame,
                           ignore_last=ignore_last, is_distance=not_distance)
    pr_fp = compute_pr_fp(pairs=pairs, positive_distance=pos_distance)
    ap_ours_fp = compute_ap_from_pr(pr_fp)
    f1_ours_fp = compute_f1_from_pr(pr_fp)
    ep_ours_fp = compute_ep_from_pr(pr_fp)
    f1_max_ours_fp = np.nanmax(f1_ours_fp)
    pr_fn = compute_pr_fn(pairs=pairs, positive_distance=pos_distance)
    ap_ours_fn = compute_ap_from_pr(pr_fn)
    f1_ours_fn = compute_f1_from_pr(pr_fn)
    ep_ours_fn = compute_ep_from_pr(pr_fn)
    f1_max_ours_fn = np.nanmax(f1_ours_fn)
    print(weights_path)
    print(sequence)
    print("AP FP: ", ap_ours_fp)
    print("F1 Max FP: ", f1_max_ours_fp)
    print("EP FP: ", ep_ours_fp)
    print("AP FN: ", ap_ours_fn)
    print("F1 Max FN: ", f1_max_ours_fn)
    print("EP FN: ", ep_ours_fn)
    pr_pair_ours = compute_pr_pairs(pair_dist=pair_dist, poses=poses,
                                    is_distance=not_distance, ignore_last=ignore_last,
                                    positive_distance=pos_distance,
                                    negative_frames=neg_frames, start_frame=start_frame)
    f1_ours_pair = compute_f1_from_pr(pr_pair_ours)
    ep_ours_pair = compute_ep_from_pr(pr_pair_ours)
    f1_max_ours_pair = np.nanmax(f1_ours_pair)
    ap_ours_pair = compute_ap_from_pr(pr_pair_ours)
    print("AP Pairs: ", ap_ours_pair)
    print("F1 Max Pairs: ", f1_max_ours_pair)
    print("EP Pairs: ", ep_ours_pair)

    if stats_save_path:
        save_dict = dict(
            AP_FP=ap_ours_fp,
            AP_FN=ap_ours_fn,
            AP_Pairs=ap_ours_pair,
            F1_max_FP=f1_max_ours_fp,
            F1_max_FN=f1_max_ours_fn,
            F1_max_Pairs=f1_max_ours_pair,
            EP_FP=ep_ours_fp,
            EP_FN=ep_ours_fn,
            EP_Pairs=ep_ours_pair,
        )

        print(f"Saving Stats file to {stats_save_path}.")
        with open(stats_save_path, "w") as f:
            json.dump(save_dict, f, indent=2)

    if save_times_path:
        save_times_path = Path(save_times_path)
        save_times_path.mkdir(parents=True, exist_ok=True)
        time_net.save_json(save_times_path / "times_model_inference.json")


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/KITTI",
                        help="Path to the dataset directory.")
    parser.add_argument("--weights_path",
                        help="Path to the model's checkpoint.")
    parser.add_argument("--num_iters", type=int, default=1,
                        help="Number of iterations to run inference.")
    parser.add_argument("--dataset", type=str, default="kitti",
                        help="Dataset on which to perform inference and evaluate the model's loop closure detection"
                             " performance.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed.")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Dataset sequence on which to evaluate the model. If set to None, a default sequence will"
                             " be used for the selected dataset.")
    parser.add_argument("--loop_file", type=str, default=None,
                        help="Override the configured GT loop file.")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Number of pairs to perform inference on at a time.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to the numpy pairwise distance file. If the file does not exist, or the "
                             "--force_inference flag is set, the script will run  inference, compute the pairwise "
                             "distance over all frames and save them to this path. If the file exists and"
                             "--force_inference is not set, no inference will be run and the distances will be read "
                             "from this file. If None, inference will be performed, but the pairwise distances will "
                             "not be saved.")
    parser.add_argument("--stats_save_path", type=str, default=None,
                        help="Path to the JSON file where the evaluation metrics will be saved. If None, no stats will "
                             "be saved.")
    parser.add_argument("--save_times_path", type=str, default=None,
                        help="Path to the JSON file where the timing stats will be saved. If None, no stats will be "
                             "saved.")
    parser.add_argument("--z_offset", type=float, default=0.283,
                        help="Offset the point cloud along the Z axis by this ammount. Used for aligning ground "
                             "planes.")
    parser.add_argument("--force_inference", action="store_true",
                        help="Runs inference. If the PR file already exists, it is overwritten.")
    parser.add_argument("--pos_distance", type=float, default=4.,
                        help="Positive distance at which loops closures are detected.")
    parser.add_argument("--neg_frames", type=int, default=50,
                        help="Minimum number of frames between two scans for them to be considered loop closure"
                             " candidates.")
    parser.add_argument("--start_frame", type=int, default=100,
                        help="Start considering loop closures after this number of frames.")
    parser.add_argument("--not_distance", action="store_false",
                        help="Indicates that the values computed/stored are not distances, but rather"
                             "some form of a similarity metric.")
    parser.add_argument("--ignore_last", action="store_true",
                        help="If set, the last frame will not be considered for loop closure.")

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main_process(0, **cli_args())

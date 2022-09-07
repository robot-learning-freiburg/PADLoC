from argparse import ArgumentParser
import os
from pathlib import Path
from time import time

import faiss
import numpy as np
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from datasets.haomo import HaomoLoader
from evaluation_comparison.inference_placerecognition_mulran import compute_PR_mulran
from evaluation_comparison.plot_PR_curve import compute_AP
from models.get_models import load_model
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from utils.data import merge_inputs
from utils.tools import set_seed


def compute_emb_map(dataloader, model, device, world_size, rank, n_samples, seq):

    emb_list_map = []
    time_descriptors = []

    if rank == 0:
        print(f"Generating descriptors of Seq {seq}")

    for batch_id, sample in enumerate(tqdm(dataloader)):
        model.eval()
        time1 = time()

        with torch.no_grad():

            anchor_list = []
            for anchor in sample["anchor"]:
                anchor = torch.from_numpy(anchor).float().to(device)
                # Remove points with x = y = z = 0
                non_valid_idxs = torch.logical_and(anchor[:, 0] == 0, anchor[:, 1] == 0)
                non_valid_idxs = torch.logical_and(non_valid_idxs, anchor[:, 2] == 0)
                anchor_i = anchor[torch.logical_not(non_valid_idxs)]
                anchor_list.append(model.module.backbone.prepare_input(anchor_i))

            model_in = KittiDataset.collate_batch(anchor_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

            emb = batch_dict["out_embedding"]
            dist.barrier()
            out_emb = [torch.zeros_like(emb) for _ in range(world_size)]
            dist.all_gather(out_emb, emb)

            if rank == 0:
                interleaved_out = torch.empty((emb.shape[0] * world_size, emb.shape[1]),
                                              device=device, dtype=emb.dtype)

                for r in range(world_size):
                    interleaved_out[r::world_size] = out_emb[r]
                emb_list_map.append(interleaved_out.detach().clone())

        time2 = time()
        time_descriptors.append(time2 - time1)

    if rank == 0:
        emb_list_map = torch.cat(emb_list_map)
        emb_list_map = emb_list_map[:n_samples].cpu().numpy()

    dist.barrier()

    return emb_list_map, time_descriptors


def main(gpu, weights_path, seed, world_size, dataset_path,
         seq1=None, seq2=None, stride=10,
         batch_size=15,
         pr_filename=None,
         stats_filename=None):

    set_seed(seed)

    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # Load model
    # saved_params = torch.load(weights_path, map_location='cpu')
    # exp_cfg = saved_params['config']
    override_cfg = dict(
        batch_size=batch_size,
    )

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg)

    # model = get_model(exp_cfg, is_training=False)
    # renamed_dict = OrderedDict()
    # for key in saved_params['state_dict']:
    #     if not key.startswith('module'):
    #         renamed_dict = saved_params['state_dict']
    #         break
    #     else:
    #         renamed_dict[key[7:]] = saved_params['state_dict'][key]
    #
    # res = model.load_state_dict(renamed_dict, strict=False)
    # if len(res[0]) > 0:
    #     print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model.train()
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
                                    find_unused_parameters=True)

    seq1 = seq1 if seq1 is not None else 2
    seq2 = seq2 if seq2 is not None else 3

    dataset_for_recall1 = HaomoLoader(dataset_path, seq1, stride=stride)
    dataset_for_recall3 = HaomoLoader(dataset_path, seq2, stride=stride)

    dataset1_sampler = DistributedSampler(
        dataset_for_recall1,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False
    )
    dataset3_sampler = DistributedSampler(
        dataset_for_recall3,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False
    )

    dataloader1 = DataLoader(
        dataset=dataset_for_recall1,
        sampler=dataset1_sampler,
        batch_size=exp_cfg["batch_size"],
        num_workers=2,
        collate_fn=merge_inputs,
        pin_memory=True
    )
    dataloader3 = DataLoader(
        dataset=dataset_for_recall3,
        sampler=dataset3_sampler,
        batch_size=exp_cfg["batch_size"],
        num_workers=2,
        collate_fn=merge_inputs,
        pin_memory=True
    )

    emb_list_map, time_descriptors = compute_emb_map(dataloader1, model, device, world_size, rank,
                                                     len(dataset_for_recall1), seq1)
    emb_list_map3, time_descriptors3 = compute_emb_map(dataloader3, model, device, world_size, rank,
                                                       len(dataset_for_recall3), seq2)

    if rank == 0:
        emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
        emb_list_map_norm3 = emb_list_map3 / np.linalg.norm(emb_list_map3, axis=1, keepdims=True)

        pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm3)

        if pr_filename:
            print(f"Saving pairwise distances to {pr_filename}.")
            np.savez(pr_filename, pair_dist)

        precision_fn, recall_fn, precision_fp, recall_fp = compute_PR_mulran(pair_dist,
                                                                             dataset_for_recall1.poses,
                                                                             dataset_for_recall3.poses)
        ap_ours_fp = compute_AP(precision_fp, recall_fp)
        ap_ours_fn = compute_AP(precision_fn, recall_fn)

        print(weights_path)
        print(exp_cfg['test_sequence'])
        print("AP FP: ", ap_ours_fp)
        print("AP FN: ", ap_ours_fn)

        if stats_filename:
            save_dict = {
                "AP FP": ap_ours_fp,
                "AP FN": ap_ours_fn,
                # "AP Pairs": ap_ours_pair
            }

            print(f"Saving Stats to {stats_filename}.")
            with open(stats_filename, "wb") as f:
                pickle.dump(save_dict, f)

        print("Done")


if __name__ == "__main__":
    def_gpu_count = torch.cuda.device_count()

    parser = ArgumentParser()

    parser.add_argument("--data", default="/home/arceyd/MT/dat/haomo/sequences/", type=Path,
                        help="dataset directory")
    parser.add_argument("--weights_path", default="/home/arceyd/MT/cp/3D/")
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--seq1', type=str, default=None)
    parser.add_argument('--seq2', type=str, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_count", type=int, default=def_gpu_count)
    parser.add_argument("--pr_filename", type=str, default=None)
    parser.add_argument("--stats_filename", type=str, default=None)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8989'

    args.gpu_count = torch.cuda.device_count()

    mp.spawn(main, nprocs=args.gpu_count, args=(
        args.weights_path, args.seed, args.gpu_count, args.data, args.seq1, args.seq2, args.stride,
        args.batch_size, args.pr_filename, args.stats_filename
    ))

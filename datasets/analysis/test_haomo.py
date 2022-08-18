from argparse import ArgumentParser
from collections import OrderedDict
import os
from pathlib import Path

import open3d as o3d
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from datasets.haomo import HaomoLoader
from datasets.analysis.plot_utils import plot_2d_trajectory
from models.get_models import get_model


def plot_haomo_paths(data_path: Path, stride: int = 10, label_steps: int = 50):

    for seq in HaomoLoader.SEQUENCES.keys():
        dataset = HaomoLoader(data_path, seq, stride=stride)

        plot_2d_trajectory(dataset.poses[:, :2, 3], labels=dataset.idx, label_steps=label_steps,
                        save_path=data_path / f"haomo_seq{dataset.sequence_label}_str{stride:03d}.pdf",
                        title=f"Haomo SEQ {dataset.sequence_label} Stride {stride}")


def plot_voxel_pc(data_path: Path, weights_path: Path, seq: int, frame, save_path: Path):

    rank = 0
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8989'

    init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=1,
        rank=rank
    )

    torch.cuda.set_device(rank)
    device = torch.device(rank)

    dataset = HaomoLoader(data_path, seq)

    saved_params = torch.load(weights_path, map_location="cpu")
    exp_cfg = saved_params["config"]

    model = get_model(exp_cfg, is_training=False)
    renamed_dict = OrderedDict()
    for key in saved_params["state_dict"]:
        if not key.startswith("module"):
            renamed_dict = saved_params["state_dict"]
            break
        else:
            renamed_dict[key[7:]] = saved_params["state_dict"][key]

    res = model.load_state_dict(renamed_dict, strict=False)
    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model.eval()
    model.to(device)
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
                                    find_unused_parameters=True)

    with torch.no_grad():
        pc = dataset[frame]["anchor"]
        pc_torch = torch.from_numpy(pc).float().to(device)
        anchor_list = [model.module.backbone.prepare_input(pc_torch)]
        model_in = KittiDataset.collate_batch(anchor_list)

    if save_path:
        vx_pc = model_in["voxel_coords"]

        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(vx_pc[:, 1:])
        o3d.io.write_point_cloud("/home/arceyd/MT/dat/haomo/sequences/voxel_pc_1.ply", pc1)
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(pc[:, :3])
        o3d.io.write_point_cloud("/home/arceyd/MT/dat/haomo/sequences/pc_1.ply", pc2)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=Path, default="/home/arceyd/MT/dat/haomo/sequences/")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--label_steps", type=int, default=50)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    lcl_args = cli_args()
    # plot_haomo_paths(**lcl_args)
    plot_voxel_pc(lcl_args["data_path"], Path("/home/arceyd/MT/cp/3D/27-05-2022_19-10-54/best_model_so_far_auc.tar"),
                  1, 100, Path("/home/arceyd/MT/dat/haomo/sequences/seq1-1_frame100_voxel.pdf"))

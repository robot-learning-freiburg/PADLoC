from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch

from datasets.KITTI_data_loader import KITTILoader3DDictTriplets
from datasets.KITTIClasses import KITTI_COLORS
from utils.geometry import get_rt_matrix


def iso_transform(device):
    zeros = torch.zeros(3).to(device)
    rot_z = torch.Tensor([0., 0., np.pi / 4]).to(device)
    rot_x = torch.Tensor([np.arcsin(np.tan(30 * np.pi / 180)), 0., 0.]).to(device)
    rot_z_matrix = get_rt_matrix(zeros, rot_z, rot_parmas="xyz")
    rot_x_matrix = get_rt_matrix(zeros, rot_x, rot_parmas="xyz")
    rot_iso = rot_x_matrix @ rot_z_matrix
    return rot_iso


def transform_vertices(*, vertices, transform_matrix, device):
    # Convert to homogeneous coordinates
    n_vertices, vertex_dim = vertices.shape
    homo_vertices = torch.ones((n_vertices, vertex_dim + 1), device=device)
    homo_vertices[:, :vertex_dim] = vertices[:, :vertex_dim].clone()
    # Apply transformation matrix
    tf_vertices = (transform_matrix @ homo_vertices.T).T
    # Return to euclidean coordinates
    tf_vertices = tf_vertices[:, :vertex_dim] / tf_vertices[:, vertex_dim:]
    return tf_vertices


def plot_pc_on_ax(ax, pc, xlim=None, ylim=None, semantic_labels=None, s=0.3, **kwargs):

    scatter_style = {}
    if semantic_labels is not None:
        scatter_style = dict(c=KITTI_COLORS[semantic_labels], s=s, marker="o", lw=0)

    ax.scatter(pc[:, 0], pc[:, 1], **scatter_style, **kwargs)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return ax


def plot_pc(pc, xlim, ylim, fig_size=(11.6, 8.2), dpi=400, semantic_labels=None, save_path=None,
            frame_on=True, edge_color="none", facecolor="white"):
    fig = plt.figure(dpi=dpi, figsize=fig_size, frameon=frame_on, edgecolor=edge_color, facecolor=facecolor)
    ax = fig.add_subplot(facecolor="white")

    ax.axis('off')
    ax.set_aspect('equal')
    ax.patch.set_facecolor("white")
    # ax.patch.set_edgecolor("none")

    ax = plot_pc_on_ax(ax, pc, xlim, ylim, semantic_labels=semantic_labels)

    ax.patch.set_facecolor("white")
    ax.patch.set_edgecolor("none")

    if save_path:
        print(f"Saving figure to {save_path}")
        plt.savefig(save_path)
    else:
        plt.show(fig)

    plt.close(fig)


def plot_single_frame(pc, tf, device, semantics, save_path):
    pc = transform_vertices(vertices=pc[:, :3], transform_matrix=tf, device=device)

    pc = pc.cpu().detach().numpy()
    semantics = semantics.cpu().detach().numpy().flatten().astype(int)

    xlim = (pc[:, 0].min(), pc[:, 0].max())
    ylim = (pc[:, 1].min(), pc[:, 1].max())

    plot_pc(pc, xlim, ylim, save_path=save_path, semantic_labels=semantics)


def plot_seq_frames(kitti_dir, seq, frames, panoptic=True, without_ground=True):
    if isinstance(seq, int):
        seq = f"{seq:02d}"

    torch.cuda.set_device(0)
    device = torch.device(0)

    poses = kitti_dir / "sequences" / seq / "poses.txt"
    dataset = KITTILoader3DDictTriplets(kitti_dir, seq, poses, npoints=200000, device=device, loop_file="loop_GT_4m",
                                        use_panoptic=panoptic, without_ground=without_ground)

    frame_idx = {val["idx"]: key for key, val in enumerate(dataset.loop_gt)}
    frames = [frame_idx[f] for f in frames]

    iso_tf = iso_transform(device)

    subframes = ["anchor", "positive", "negative"]

    for frame in frames:
        sample = dataset[frame]

        for subframe in subframes:
            if subframe not in sample:
                continue

            pc = sample[subframe]
            pc[:, 3] *= -1
            semantics = sample[f"{subframe}_semantic"]
            pc_frame = sample[f"{subframe}_idx"]

            img_dir = Path("/home/arceyd/")
            img_name = f"PanPC_{subframe}"
            ext = "png"
            save_path = img_dir / f"{img_name}_{seq}_{pc_frame}.{ext}"

            plot_single_frame(pc, iso_tf, device, semantics, save_path)


def main(kitti_dir, seq, frames, without_ground=False):
    plot_seq_frames(kitti_dir, seq, frames, panoptic=True, without_ground=without_ground)


def frame_list(arg):
    if arg is None:
        return [800]

    return [int(frame) for frame in arg.split(",")]


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--kitti_dir", type=Path, default="/data/arceyd/kitti/dataset")
    parser.add_argument("--without_ground", action="store_true")
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--frames", type=frame_list, default=[800])

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

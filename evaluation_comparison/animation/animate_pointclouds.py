from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import torch

from datasets.KITTI_data_loader import KITTILoader3DPoses
from evaluation_comparison.plot.plot_pan_pc import plot_pc_on_ax, transform_vertices
from utils.geometry import mat2xyzrpy

Tranformation = namedtuple("Transformation", ["label", "transforms", "per_frame", "plot_car", "tf_car",
                                              "x_lim", "y_lim", "z_lim", "aspect_ratio", "s"])


def look_at(camera_position, camera_target, up_vector):
    fwd = camera_position - camera_target
    fwd = fwd / np.linalg.norm(fwd)

    rgt = np.cross(up_vector, fwd)
    rgt = rgt / np.linalg.norm(rgt)

    up = np.cross(fwd, rgt)

    view = np.eye(4)

    view[0, :3] = rgt
    view[1, :3] = up
    view[2, :3] = fwd

    eye = np.eye(4)
    eye[:3, 3] = - camera_position

    view = view @ eye
    return view


def create_perspective_projection_from_bounds(
        left,
        right,
        bottom,
        top,
        near,
        far,
        dtype=None):

    a = (right + left) / (right - left)
    b = (top + bottom) / (top - bottom)
    c = -(far + near) / (far - near)
    d = -2. * far * near / (far - near)
    e = 2. * near / (right - left)
    f = 2. * near / (top - bottom)

    tf = np.array((
        ( e, 0.,   a,  0.),
        (0.,  f,   b,  0.),
        (0., 0.,   c,   d),
        (0., 0., -1.,  0.),
    ), dtype=dtype)

    return tf


def create_perspective_projection(fovy, aspect, near, far, degrees=False, dtype=None):
    if degrees:
        fovy = np.deg2rad(fovy)
    ymax = near * np.tan(fovy / 2)
    xmax = ymax * aspect
    return create_perspective_projection_from_bounds(-xmax, xmax, -ymax, ymax, near, far, dtype=dtype)


def main(dataset_path, sequence, save_path, pose_file="poses.txt", loop_file="loog_GT_4m", gpu=0,
         transform_keys="car_centric", start_frame=0, end_frame=None):

    transform_keys = list(filter(None, transform_keys.split(",")))

    car_file = Path(save_path) / "car_top.png"
    car_img = plt.imread(car_file)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    video_size = np.array([1920, 1080])

    y_lim = 30
    x_lim = y_lim * video_size[0] / video_size[1]
    x_lim = np.array([-1., 1.]) * x_lim
    y_lim = np.array([-1., 1.]) * y_lim

    resolution = 96
    fig_size = video_size / resolution

    dataset = KITTILoader3DPoses(dataset_path, sequence=sequence,
                                 poses=dataset_path / "sequences" / sequence / pose_file, npoints=None, device=None,
                                 loop_file=loop_file, use_semantic=True, use_logits=False, train=False)

    poses = np.stack(dataset.poses)
    fixed_north_tf = np.eye(4)
    fixed_north_tf = np.tile(fixed_north_tf, (poses.shape[0], 1, 1))
    fixed_north_tf[:, :3, :3] = poses[:, :3, :3]
    fixed_north_tf = torch.from_numpy(fixed_north_tf).to(device=device, dtype=torch.float32)

    camera_pos = np.array([0., 0., 0.])
    camera_tgt = np.array([1., 0., 0.])
    camera_up = np.array([0., 0., 1.])
    driver_view = look_at(camera_position=camera_pos, camera_target=camera_tgt, up_vector=camera_up)

    near = 1.  # Near (projection) plane
    far = 70.  # Far plane
    th_v = 60.  # Vertical FOV (deg)
    aspect_ratio = video_size[0] / video_size[1]

    driver_projection = create_perspective_projection(fovy=th_v, aspect=aspect_ratio, near=near, far=far, degrees=True)

    driver = driver_projection @ driver_view
    driver = torch.from_numpy(driver).to(device=device, dtype=torch.float32)

    transformations = [
        Tranformation("car_centric", None, per_frame=False, plot_car=True, tf_car=False,
                      x_lim=x_lim, y_lim=y_lim, z_lim=None, aspect_ratio="equal", s=0.3),
        Tranformation("fixed_north", fixed_north_tf, per_frame=True, plot_car=True, tf_car=True,
                      x_lim=x_lim, y_lim=y_lim, z_lim=None, aspect_ratio="equal", s=0.3),
        Tranformation("pov", driver, per_frame=False, plot_car=False, tf_car=False,
                      x_lim=np.array([-1., 1.]), y_lim=np.array([-1., 1.]), z_lim=np.array([-1., 1.]),
                      aspect_ratio=1/aspect_ratio, s=1.),
    ]

    selected_transformations = [t for t in transformations if t.label in transform_keys]

    for t in selected_transformations:
        save_dir = save_path / t.label
        save_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(dataset)

    for i in range(start_frame, end_frame):
        frame = dataset[i]
        pc = frame["anchor"]
        sem = frame["anchor_semantic"]

        for t in selected_transformations:
            tf_pc = pc.clone().to(device)
            if t.transforms is not None:
                tf = t.transforms
                if t.per_frame:
                    tf = tf[i]
                tf_pc = transform_vertices(vertices=tf_pc[:, :3], transform_matrix=tf, device=device)

            frame_save_path = save_path / t.label / f"{i:06d}.png"

            fig = plt.figure(dpi=resolution, figsize=fig_size, frameon=False, edgecolor="none", facecolor="none")
            ax = fig.add_subplot(facecolor="white")

            ax.axis('off')
            ax.set_aspect(t.aspect_ratio)

            if t.z_lim is not None:
                z_mask = torch.logical_and(tf_pc[:, 2] >= t.z_lim[0], tf_pc[:, 2] <= t.z_lim[1])
                tf_pc = tf_pc[z_mask]
                sem = sem[z_mask]

            plot_pc_on_ax(ax, tf_pc.detach().cpu().numpy(), xlim=t.x_lim, ylim=t.y_lim,
                          semantic_labels=sem.detach().cpu().numpy().astype(int), s=t.s)

            if t.plot_car:
                if t.transforms is not None and t.tf_car:
                    yaw = mat2xyzrpy(tf)[-1]
                    car_tf = transforms.Affine2D().rotate(yaw)

                    ax.imshow(car_img, transform=car_tf + ax.transData, extent=[-2.15, 2.15, -1., 1.])
                else:
                    ax.imshow(car_img, extent=[-2.15, 2.15, -1., 1.])

            fig.tight_layout(pad=0.05)

            if frame_save_path:
                print(f"Saving figure to {frame_save_path}")
                plt.savefig(frame_save_path)
            else:
                plt.show(fig)

            plt.close(fig)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", type=Path, default=Path("/home/arceyd/MT/dat/kitti/dataset/"))
    parser.add_argument("--save_path", type=Path, default=Path("/home/arceyd/MT/res/animations/pc/"))
    parser.add_argument("--sequence", default="08")

    parser.add_argument("--pose_file", type=str, default="poses.txt")
    parser.add_argument("--loop_file", type=str, default="loop_GT_4m")
    parser.add_argument("--transform_keys", type=str, default="car_centric")

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)

    # parser.add_argument("--gpu", "-g", type=int, default=0)
    #
    # parser.add_argument("--style", default="cw")
    #
    # parser.add_argument("--image_paths", "-f", type=path_list_arg)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

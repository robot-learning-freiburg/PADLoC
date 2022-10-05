from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import torch.cuda

from datasets.KITTI_data_loader import KITTILoader3DDictPairs
from evaluation_comparison.plot.plot_matches import collate_samples
from evaluation_comparison.plot.plot_pan_pc import transform_vertices, plot_pc_on_ax
from evaluation_comparison.plot_styles import Color
from models.get_models import load_model


class TransformLerp:
    def __init__(self, init_translation: np.ndarray, init_rotation_mat: np.ndarray,
                 last_translation: np.ndarray, last_rotation_mat: np.ndarray,
                 init_time: float, last_time: float,
                 device):

        self.init_translation = init_translation
        self.last_translation = last_translation
        self.delta_translation = self.last_translation - self.init_translation
        self.init_time = init_time
        self.last_time = last_time
        self.delta_time = self.last_time - self.init_time

        self.device = device

        rotations = np.stack([init_rotation_mat, last_rotation_mat])
        rotations = Rotation.from_matrix(rotations)

        self.slerp = Slerp(times=[init_time, last_time], rotations=rotations)

    def lerp(self, t):
        translation = self.init_translation + ((t - self.init_time) / self.delta_time) * self.delta_translation

        rotation = self.slerp(t).as_matrix()

        transformation = torch.eye(4, device=self.device)
        transformation[:3, 3] = torch.from_numpy(translation).to(self.device)
        transformation[:3, :3] = torch.from_numpy(rotation).to(self.device)

        return transformation


def main(kitti_path, seq, frame, weights_path, save_path: Path, loop_file="loop_GT_4m", gpu=0):

    dataset = KITTILoader3DDictPairs(dir=kitti_path, sequence=seq, poses=kitti_path / "sequences" / seq / "poses.txt",
                                     loop_file=loop_file, npoints=None, device=None,
                                     use_panoptic=True)

    save_path.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model, exp_cfg = load_model(weights_path=weights_path)
    model.eval()
    model.to(device)

    frame_ids = {k["idx"]: i for i, k in enumerate(dataset.loop_gt)}
    pairs = dataset[frame_ids[frame]]
    model_in = collate_samples(pairs, model, device)

    batch_dict = model(model_in)

    anc_pc = batch_dict["anchor"][0][:, :3].to(device)
    pos_pc = batch_dict["positive"][0][:, :3].to(device)

    pred_pos_pc = batch_dict["sinkhorn_matches"][0][:, :3]

    anc_samp_ids = batch_dict["keypoint_idxs"][0]
    # pos_samp_ids = batch_dict["keypoint_idxs"][1]

    p2a_transf = torch.eye(4, device=device)
    p2a_transf[:3, :] = batch_dict["transformation"][0]  # Prediction

    geometric_accuracy = anc_pc[anc_samp_ids] - transform_vertices(vertices=pred_pos_pc, transform_matrix=p2a_transf,
                                                                   device=device)
    geometric_accuracy = geometric_accuracy.square().sum(dim=1).sqrt().detach().cpu().numpy()
    good_matches = geometric_accuracy < 6.

    anc_semantic = batch_dict["anchor_semantic"][0].detach().cpu().numpy().flatten().astype(int)
    pos_semantic = batch_dict["positive_semantic"][0].detach().cpu().numpy().flatten().astype(int)

    gt_p2a_transf = batch_dict["p2a_transform"][0]

    p2a_transf = gt_p2a_transf  # Set to GT

    # matches = batch_dict["transport"]
    match_weights = batch_dict["conf_weights"][0]

    clip_duration = 6.
    fps = 30
    video_size = np.array([1920, 1080])
    resolution = 96
    fig_size = video_size / resolution

    frames = int(clip_duration * fps)
    # frames = 10  # For debugging

    separation = 60

    x_lim = 60
    y_lim = 60
    x_lim = np.array([-1., 1.]) * (x_lim + separation)
    y_lim = np.array([-1., 1.]) * y_lim

    anc_tra_s = np.array([-separation, 0., 0.])
    anc_rot_s = Rotation.from_euler(seq="xyz", angles=[0., 0., 0.], degrees=False).as_matrix()
    anc_tra_e = np.array([0., 0., 0.])
    anc_rot_e = Rotation.from_euler(seq="xyz", angles=[0., 0., 0.], degrees=False).as_matrix()
    anc_lerp = TransformLerp(init_translation=anc_tra_s, init_rotation_mat=anc_rot_s,
                             last_translation=anc_tra_e, last_rotation_mat=anc_rot_e,
                             init_time=0., last_time=frames - 1,
                             device=device)

    pos_tra_s = np.array([separation, 0., 0.])
    pos_rot_s = Rotation.from_euler(seq="xyz", angles=[0., 0., 0.], degrees=False).as_matrix()
    pos_tra_e = p2a_transf[:3, 3].detach().cpu().numpy()
    pos_rot_e = p2a_transf[:3, :3].detach().cpu().numpy()
    pos_lerp = TransformLerp(init_translation=pos_tra_s, init_rotation_mat=pos_rot_s,
                             last_translation=pos_tra_e, last_rotation_mat=pos_rot_e,
                             init_time=0., last_time=frames - 1,
                             device=device)

    style = Color()

    semantic_coloring = False
    acceleration = True
    accel_spread = 20

    def accel_curve(x, span=1, spread=1):
        # Sigmoid function for simulating time dilation
        return span / (1 + np.exp(-(x - (span / 2)) / spread))

    for i in range(frames):
        fig, ax = plt.subplots(figsize=fig_size, dpi=resolution, facecolor=None, edgecolor=None, frameon=False)

        t = accel_curve(i, span=frames - 1, spread=accel_spread) if acceleration else i

        anc_transf = anc_lerp.lerp(t)
        anc_frame = transform_vertices(vertices=anc_pc, transform_matrix=anc_transf, device=device)

        pos_transf = pos_lerp.lerp(t)
        pos_frame = transform_vertices(vertices=pos_pc, transform_matrix=pos_transf, device=device)

        pred_pos_frame = transform_vertices(vertices=pred_pos_pc, transform_matrix=pos_transf, device=device)

        anc_semantic_labels = anc_semantic if semantic_coloring else None
        pos_semantic_labels = pos_semantic if semantic_coloring else None

        plot_pc_on_ax(ax, pc=anc_frame.detach().cpu().numpy(), xlim=x_lim, ylim=y_lim,
                      semantic_labels=anc_semantic_labels,
                      # **style.src_point_cloud
                      s=0.1, c="#00FF00", marker="o", lw=0,
                      )

        match_segments = np.concatenate([anc_frame[anc_samp_ids][:, :2].reshape(-1, 1, 2).detach().cpu().numpy(),
                                         pred_pos_frame[:, :2].reshape(-1, 1, 2).detach().cpu().numpy()], axis=1)

        match_style = style.match_lines_styles(match_weights[good_matches],
                                               geometric_accuracy=np.ones(good_matches.sum(), dtype=bool)
                                               )
        match_style["colors"][:, :3] *= 0
        match_style["linewidths"] *= 3

        match_lines = LineCollection(match_segments[good_matches],
                                     **match_style
                                     )
        ax.add_collection(match_lines)

        plot_pc_on_ax(ax, pc=pos_frame.detach().cpu().numpy(), xlim=x_lim, ylim=y_lim,
                      semantic_labels=pos_semantic_labels,
                      # **style.tgt_point_cloud
                      s=0.1, c="#FF0000", marker="o", lw=0,
                      )

        fig.tight_layout(pad=0.05)

        filename = f"{i:05d}.png"
        print(f"Saving frame {filename}.")
        file_path = save_path / filename
        fig.savefig(file_path)
        plt.close()


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--kitti_path", "-k", type=Path, default=Path("/home/arceyd/MT/dat/kitti/dataset/"))
    parser.add_argument("--seq", "-s", default="08")
    parser.add_argument("--frame", type=int, default=800)

    parser.add_argument("--weights_path", type=Path,
                        default=Path("/home/arceyd/MT/cp/3D/27-05-2022_19-10-54/checkpoint_last_iter.tar"))

    parser.add_argument("--save_path", type=Path,
                        default=Path("/home/arceyd/MT/res/animations/matches"))

    # parser.add_argument("--gpu", "-g", type=int, default=0)
    #
    # parser.add_argument("--style", default="cw")
    #
    # parser.add_argument("--image_paths", "-f", type=path_list_arg)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

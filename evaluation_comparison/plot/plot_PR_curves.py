from collections import namedtuple
from pathlib import Path
from typing import List, NoReturn, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from datasets.KITTI_data_loader import KITTILoader3DPoses
from evaluation_comparison.metrics.detection import generate_pairs, compute_pr_fp

Dataset = namedtuple("Dataset", ["label", "path", "sequence", "poses_func", "pose_file",
                                 "positive_dist", "negative_frames", "start_frame", "pair_files"])
Poses = namedtuple("Poses", ["poses", "pose_tree"])
PairFile = namedtuple("PairFile", ["label", "path",
                                   "is_dist", "ignore_last",
                                   "marker"])
PRPlot = namedtuple("PRPlot", ["title", "pr_func"])

PRCurve = namedtuple("PRCurve", ["label", "precision", "recall", "marker"])


def plot_pr_curve(ax, precision, recall, label, marker, mark_every: float = 0.03, smooth_alpha=0.5):

    if smooth_alpha:
        precision_df = pd.DataFrame({"pre": precision})
        precision = precision_df.ewm(alpha=smooth_alpha).mean()
        precision = precision["pre"].to_numpy()

    ax.plot(recall, precision, label=label, marker=marker, markevery=mark_every)


def plot_pr_curves(plot_title: str, pr_curves: List[PRCurve],
                   x_label: str = "Recall [%]", y_label: str = "Precision [%]",
                   save_path: Optional[Path] = None, show: bool = False,
                   x_ticks: Optional[List[float]] = None, x_tick_labels: Optional[List[str]] = None,
                   y_ticks: Optional[List[float]] = None, y_tick_labels: Optional[List[str]] = None,
                   mark_every=0.03, dpi=300
                   ) -> NoReturn:
    fig, ax = plt.subplots(dpi=dpi)

    for pr_curve in pr_curves:
        plot_pr_curve(ax, pr_curve.precision, pr_curve.recall, pr_curve.label, pr_curve.marker, mark_every=mark_every)

    if x_ticks is None:
        x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    if x_tick_labels is None:
        x_tick_labels = ["0", "20", "40", "60", "80", "100"]

    if y_ticks is None:
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    if y_tick_labels is None:
        y_tick_labels = ["0", "20", "40", "60", "80", "100"]

    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_title(plot_title)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.legend(loc="lower left")

    if save_path is not None:
        print(f"Saving plot to file {save_path}.")
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def compute_and_plot_pr(*, pair_files: List[Dataset],
                        pr_plots: List[PRPlot],
                        save_path: Optional[Path] = None,
                        extension: str = "pdf"
                        ) -> NoReturn:

    for dataset in pair_files:
        poses = load_poses(dataset)

        seq_str = f", s{dataset.sequence}" if dataset.sequence is not None else ""
        dataset_str = f" ({dataset.label}{seq_str})"
        precision_recall_curves = {p.title: [] for p in pr_plots}

        for pair_file in dataset.pair_files:
            pair_dist = np.load(pair_file.path)["arr_0"]

            print(f"\nGenerating pairs for {pair_file.label}")
            pairs = generate_pairs(pair_dist=pair_dist, poses=poses.poses, map_tree_poses=poses.pose_tree,
                                   positive_distance=dataset.positive_dist, negative_frames=dataset.negative_frames,
                                   start_frame=dataset.start_frame,
                                   ignore_last=pair_file.ignore_last, is_distance=pair_file.is_dist)

            for pr_plot in pr_plots:
                pr = pr_plot.pr_func(detected_loop=pairs.detected_loops, real_loop=pairs.real_loops,
                                     distances=pairs.distances, positive_distance=dataset.positive_dist)
                pr_curve = PRCurve(label=pair_file.label, precision=pr.precision, recall=pr.recall,
                                   marker=pair_file.marker)
                precision_recall_curves[pr_plot.title].append(pr_curve)

        for title, pr_curves in precision_recall_curves.items():
            plot_title = f"{title}{dataset_str}"
            plot_save_path = "_".join(plot_title.lower().split(" ")) + "." + extension
            plot_save_path = save_path / plot_save_path
            plot_pr_curves(plot_title=plot_title, pr_curves=pr_curves, save_path=plot_save_path)


def load_poses(dataset_cfg: Dataset) -> Poses:
    return dataset_cfg.poses_func(dataset_cfg.path, seq=dataset_cfg.sequence, pose_file=dataset_cfg.pose_file)


def kitti_poses(dataset_path, seq, pose_file) -> Poses:
    dataset = KITTILoader3DPoses(dataset_path, seq, pose_file, device=None, npoints=None, loop_file="loop_GT_4m")

    poses = np.stack(dataset.poses)
    pose_tree = KDTree(poses[:, :3, 3])
    return Poses(poses, pose_tree)


def ford_poses():
    raise NotImplementedError


def frbg_poses():
    raise NotImplementedError


def main():

    ds_path = Path("/work/dlclarge2/arceyd-padloc/dat/")
    pair_path = Path("/work/dlclarge2/arceyd-padloc/res/final_models/")
    bl_path = pair_path / "baselines"
    save_path = Path("/work/dlclarge2/arceyd-padloc/res/img/pr_curves/")
    extension = "pdf"

    save_path.mkdir(parents=True, exist_ok=True)

    pair_files = [
        Dataset(label="KITTI", path=ds_path / "KITTI", sequence="08", poses_func=kitti_poses,
                pose_file=ds_path / "KITTI" / "sequences" / "08" / "poses.txt",
                positive_dist=4.0, negative_frames=50, start_frame=100,
                pair_files=[
                    # Baselines
                    # PairFile(label="M2DP", path=bl_path / "m2dp" / "kitt" / "lcd" / "pairs_dist_m2dp_08.npz",
                    #          is_dist=True, ignore_last=False, marker="x"),
                    # PairFile(label="Scan Context", path=bl_path / "sc" / "kitt" / "lcd" / "pairs_dist_sc_08.npz",
                    #          pose_cfg="kitti", is_dist=True, ignore_last=False, marker="x"),
                    # PairFile(label="ISC", path=bl_path / "isc" / "kitt" / "lcd" / "pairs_dist_isc_08.npz",
                    #          is_dist=False, ignore_last=False, marker="x"),
                    # PairFile(label="LiDaR-IRIS", path=bl_path / "iris2" / "kitt" / "lcd" / "pairs_dist_iris2_08.npz",
                    #          is_dist=True, ignore_last=False, marker="x"),
                    PairFile(label="OverlapNet", path=bl_path / "overlap" / "kitt" / "lcd" / "overlap_pairs_08.npz",
                             is_dist=False, ignore_last=False, marker="x"),
                    # PairFile(label="SG_PR", path="",
                    #          is_dist=True, ignore_last=False, marker="x"),

                    # Deep-Learning Methods
                    PairFile(label="LCDNet", path=pair_path / "lcdnet_210916000234" / "lastiter" / "kitt" / "lcd" /
                             "eval_lcd_lcdnet_210916000234_lastiter_kitt_seq_08.npz",
                             is_dist=True, ignore_last=False, marker="x"),
                    # PairFile(label="DCP", path=pair_path / "dcp_220404183414" / "lastiter" / "kitt" / "lcd" /
                    #          "eval_lcd_dcp_220404183414_lastiter_kitt_seq_08.npz",
                    #          is_dist=True, ignore_last=False, marker="x"),
                    PairFile(label="PADLoC", path=pair_path / "padloc_220527191054" / "lastiter" / "kitt" / "lcd" /
                             "eval_lcd_padloc_220527191054_lastiter_kitt_seq_08.npz",
                             is_dist=True, ignore_last=False, marker="x"),
                ]),
    ]

    pr_plots = [
        PRPlot(title="Precision-Recall (FP)", pr_func=compute_pr_fp),
    ]

    compute_and_plot_pr(pair_files=pair_files, pr_plots=pr_plots,
                        save_path=save_path, extension=extension)


if __name__ == "__main__":
    main()

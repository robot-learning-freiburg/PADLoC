from argparse import ArgumentParser
import numpy as np
import os
import pickle
from scipy.stats import circmean
from scipy.spatial.transform import Rotation

from datasets.KITTI360Dataset import KITTI3603DPoses


def normalize_angles(angles, low=0., high=2*np.pi):
    diff = high - low
    angles[angles < low] += diff
    angles[angles > high] -= diff
    return angles


def get_stats(data_path, seq, loop_file="loop_GT_4m_noneg", needs_labels=False):
    dataset = KITTI3603DPoses(data_path, seq, train=False, loop_file=loop_file, use_panoptic=needs_labels)

    scan_dir = os.path.join(data_path, "data_3d_raw", seq, "velodyne_points", "data")

    num_frames = len([scan for scan in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, scan))])
    num_loops = len(dataset.loop_gt)

    frames_with_loops = np.array(dataset.have_matches)
    pair_tra = []
    pair_yaw = []
    avg_pair_tra = []
    avg_pair_yaw = []

    for frame in dataset.loop_gt:
        anc_frame = frame["idx"]
        anc_idx = dataset.frame_to_idx[anc_frame]

        anc_pose = dataset.poses[anc_idx]
        pos_poses = [dataset.poses[dataset.frame_to_idx[f]] for f in frame["positive_idxs"]]
        pos_poses = np.stack(pos_poses)

        rel_poses = np.linalg.inv(anc_pose) @ pos_poses
        rel_trans = rel_poses[:, :3, 3]
        rel_trans = np.linalg.norm(rel_trans, axis=1)
        pair_tra.append(rel_trans)
        avg_rel_trans = np.mean(rel_trans)
        avg_pair_tra.append(avg_rel_trans)

        rel_yaws = Rotation.from_matrix(rel_poses[:, :3, :3])
        rel_yaws = rel_yaws.as_euler("xyz")
        rel_yaws = rel_yaws[:, 2]
        rel_yaws = normalize_angles(rel_yaws)
        avg_rel_yaw = circmean(rel_yaws)
        pair_yaw.append(rel_yaws)
        avg_pair_yaw.append(avg_rel_yaw)

    avg_pair_tra = np.stack(avg_pair_tra)
    avg_pair_yaw = np.stack(avg_pair_yaw)
    norm_avg_pair_yaw = normalize_angles(avg_pair_yaw, low=np.pi / 4, high=np.pi * 9 / 4)
    crosses = np.logical_or(
        np.logical_and(norm_avg_pair_yaw >= np.pi * 1 / 4, norm_avg_pair_yaw < np.pi * 3 / 4),
        np.logical_and(norm_avg_pair_yaw >= np.pi * 5 / 4, norm_avg_pair_yaw < np.pi * 7 / 4),
    )
    num_crs_loops = np.sum(crosses)
    pc_crs_loops = 100 * num_crs_loops / num_loops
    rev_loops = np.logical_and(norm_avg_pair_yaw >= np.pi * 3 / 4, norm_avg_pair_yaw < np.pi * 5 / 4)
    num_rev_loops = np.sum(rev_loops)
    pc_rev_loops = 100 * num_rev_loops / num_loops
    fwd_loops = np.logical_and(norm_avg_pair_yaw >= np.pi * 7 / 4, norm_avg_pair_yaw < np.pi * 9 / 4)
    num_fwd_loops = np.sum(fwd_loops)
    pc_fwd_loops = 100 * num_fwd_loops / num_loops

    avg_tra = np.mean(avg_pair_tra)
    avg_yaw = circmean(avg_pair_yaw)

    stats = dict(
        frames=num_frames,

        frames_with_loops=frames_with_loops,
        num_loops=num_loops,

        pair_tra=pair_tra,
        avg_pair_tra=avg_pair_tra,
        avg_tra=avg_tra,

        pair_yaw=pair_yaw,
        avg_pair_yaw=avg_pair_yaw,
        avg_yaw_deg=avg_yaw * 180 / np.pi,

        # Loop Stats
        crs_loops=frames_with_loops[crosses],
        num_crs_loops=num_crs_loops,
        pc_crs_loops=pc_crs_loops,
        fwd_loops=frames_with_loops[fwd_loops],
        num_fwd_loops=num_fwd_loops,
        pc_fwd_loops=pc_fwd_loops,
        rev_loops=frames_with_loops[rev_loops],
        num_rev_loops=num_rev_loops,
        pc_rev_loops=pc_rev_loops
    )

    return stats


def main(data_path, save_path,
         loop_file="loop_GT_4m_noneg"):
    sequences = [
        # Training
        "2013_05_28_drive_0000_sync",
        "2013_05_28_drive_0004_sync",
        "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0006_sync",
        # Validation
        "2013_05_28_drive_0002_sync",
        "2013_05_28_drive_0009_sync"
    ]

    gt_types = [
        ("needs_poses", False),
        ("needs_labels", True),
    ]

    def stats(seq, needs_labels):
        return get_stats(data_path, seq, loop_file, needs_labels)

    stat_dict = {seq: {label: stats(seq, w_labels) for label, w_labels in gt_types} for seq in sequences}

    if save_path is not None:
        print(f"Saving stats to {save_path}.")
        with open(save_path, "wb") as f:
            pickle.dump(stat_dict, f)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--loop_file", type=str, default="loop_GT_4m_noneg")

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

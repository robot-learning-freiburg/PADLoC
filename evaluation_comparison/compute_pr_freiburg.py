from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import pickle
from sklearn.neighbors import KDTree

from datasets.Freiburg import FreiburgDataset
from evaluation_comparison.metrics.detection import generate_pairs, compute_pr_fp, compute_pr_fn, load_pairs_file


# def compute_pr(pair_dist, poses, map_tree_poses, is_distance=True, ignore_last=False,
#                positive_distance=10., negative_frames=200):
#     real_loop = []
#     detected_loop = []
#     distances = []
#     last = poses.shape[0]
#     if ignore_last:
#         last = last-1
#
#     print("Generating Positive Pairs.")
#     for i in tqdm(range(negative_frames + 1, last)):
#         min_range = max(0, i-negative_frames)  # Scan Context
#         current_pose = poses[i][:3, 3]
#         indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), positive_distance)
#         valid_idxs = list(set(indices[0]) - set(range(min_range, last)))
#         if len(valid_idxs) > 0:
#             real_loop.append(1)
#         else:
#             real_loop.append(0)
#
#         if is_distance:
#             candidate = pair_dist[i, :i-negative_frames].argmin()
#             detected_loop.append(-pair_dist[i, candidate])
#         else:
#             candidate = pair_dist[i, :i-negative_frames].argmax()
#             detected_loop.append(pair_dist[i, candidate])
#         candidate_pose = poses[candidate][:3, 3]
#         distances.append(np.linalg.norm(candidate_pose-current_pose))
#
#     distances = np.array(distances)
#     detected_loop = -np.array(detected_loop)
#     real_loop = np.array(real_loop)
#     precision_fn = []
#     recall_fn = []
#     for thr in np.unique(detected_loop):
#         asd = detected_loop <= thr
#         asd = asd & real_loop
#         asd = asd & (distances <= positive_distance)
#         tp = asd.sum()
#         fn = (detected_loop <= thr) & (distances > positive_distance) & real_loop
#         fn2 = (detected_loop > thr) & real_loop
#         fn = fn.sum() + fn2.sum()
#         fp = (detected_loop <= thr) & (distances > positive_distance) & (1 - real_loop)
#         fp = fp.sum()
#         # fp = (detected_loop<=thr).sum() - tp
#         # fn = (real_loop.sum()) - tp
#         if (tp+fp) > 0:
#             precision_fn.append(tp/(tp+fp))
#         else:
#             precision_fn.append(1.)
#         recall_fn.append(tp/(tp+fn))
#     precision_fp = []
#     recall_fp = []
#     for thr in np.unique(detected_loop):
#         asd = detected_loop<=thr
#         asd = asd & real_loop
#         asd = asd & (distances <= positive_distance)
#         tp = asd.sum()
#         fp = (detected_loop <= thr) & (distances > positive_distance)
#         fp = fp.sum()
#         fn = (detected_loop > thr) & real_loop
#         fn = fn.sum()
#         if (tp+fp) > 0:
#             precision_fp.append(tp/(tp+fp))
#         else:
#             precision_fp.append(1.)
#         recall_fp.append(tp/(tp+fn))
#
#     return np.array(precision_fn), np.array(recall_fn), np.array(precision_fp), np.array(recall_fp)


def compute_ap(precision, recall):
    ap = np.diff(recall)
    ap = np.dot(ap, precision[1:])
    return ap


def load_poses(dataset_path, without_ground=False):
    dataset = FreiburgDataset(dataset_path, without_ground=without_ground)
    poses = np.stack(dataset.poses)
    return poses


def main(pair_distances_path, dataset_path, stats_save_path=None, without_ground=False,
         positive_distance=10., negative_frames=200):

    pair_distances = load_pairs_file(pair_distances_path)

    poses = load_poses(dataset_path, without_ground)
    poses_tree = KDTree(poses[:, :3, 3])

    pairs = generate_pairs(pair_dist=pair_distances, poses=poses,
                           map_tree_poses=poses_tree,
                           positive_distance=positive_distance,
                           negative_frames=negative_frames,
                           start_frame=negative_frames + 1)

    pre_fn, rec_fn = compute_pr_fn(detected_loop=pairs.detected_loops, real_loop=pairs.real_loops,
                                   distances=pairs.distances, positive_distance=positive_distance)
    pre_fp, rec_fp = compute_pr_fp(detected_loop=pairs.detected_loops, real_loop=pairs.real_loops,
                                   distances=pairs.distances, positive_distance=positive_distance)

    ap_fp = compute_ap(pre_fp, rec_fp)
    ap_fn = compute_ap(pre_fn, rec_fn)

    print("AP FP: ", ap_fp)
    print("AP FN: ", ap_fn)

    if stats_save_path:
        save_dict = {
            "AP_FP": ap_fp,
            "AP_FN": ap_fn,
        }

        print(f"Saving Stats file to {stats_save_path}.")
        with open(stats_save_path, "wb") as f:
            pickle.dump(save_dict, f)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--pair_distances_path", type=Path)
    parser.add_argument("--dataset_path", type=Path, default=Path("/home/cattaneo/Datasets/Freiburg/"))
    parser.add_argument("--stats_save_path", type=Path, default=None)
    parser.add_argument("--without_ground", type=bool, default=False)
    parser.add_argument("--positive_distance", type=float, default=10.)
    parser.add_argument("--negative_frames", type=int, default=200)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

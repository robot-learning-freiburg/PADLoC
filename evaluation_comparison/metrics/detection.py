from collections import namedtuple

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

Pairs = namedtuple("Pairs", ["distances", "real_loops", "detected_loops"])
PR = namedtuple("PR", ["precision", "recall"])


def generate_pairs(pair_dist: np.ndarray, poses: np.ndarray, map_tree_poses: KDTree,
                   positive_distance: float = 4.0, negative_frames: int = 50, start_frame: int = 100,
                   ignore_last: bool = False, is_distance: bool = True
                   ) -> Pairs:
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last - 1

    for i in tqdm(range(start_frame, last)):
        min_range = max(0, i - negative_frames)
        current_pose = poses[i][:3, 3]
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), positive_distance)
        valid_idxs = list(set(indices[0]) - set(range(min_range, last)))
        if len(valid_idxs) > 0:
            real_loop.append(1)
        else:
            real_loop.append(0)

        if is_distance:
            candidate = pair_dist[i, :i - negative_frames].argmin()
            detected_loop.append(-pair_dist[i, candidate])
        else:
            candidate = pair_dist[i, :i - negative_frames].argmax()
            detected_loop.append(pair_dist[i, candidate])
        candidate_pose = poses[candidate][:3, 3]
        distances.append(np.linalg.norm(candidate_pose - current_pose))

    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)

    return Pairs(distances=distances, real_loops=real_loop, detected_loops=detected_loop)


def compute_pr_fn(detected_loop: np.ndarray, real_loop: np.ndarray, distances: np.ndarray,
                  positive_distance: float = 4.0
                  ) -> PR:

    precision = []
    recall = []
    for thr in np.unique(detected_loop):
        asd = detected_loop <= thr
        asd = asd & real_loop
        asd = asd & (distances <= positive_distance)
        tp = asd.sum()
        fn = (detected_loop <= thr) & (distances > positive_distance) & real_loop
        fn2 = (detected_loop > thr) & real_loop
        fn = fn.sum() + fn2.sum()
        fp = (detected_loop <= thr) & (distances > positive_distance) & (1 - real_loop)
        fp = fp.sum()
        # fp = (detected_loop<=thr).sum() - tp
        # fn = (real_loop.sum()) - tp
        if (tp + fp) > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(1.)
        recall.append(tp / (tp + fn))

    return PR(precision=np.array(precision), recall=np.array(recall))


def compute_pr_fp(detected_loop: np.ndarray, real_loop: np.ndarray, distances: np.ndarray,
                  positive_distance: float = 4.0
                  ) -> PR:
    precision = []
    recall = []
    for thr in np.unique(detected_loop):
        asd = detected_loop <= thr
        asd = asd & real_loop
        asd = asd & (distances <= positive_distance)
        tp = asd.sum()
        fp = (detected_loop <= thr) & (distances > positive_distance)
        fp = fp.sum()
        fn = (detected_loop > thr) & real_loop
        fn = fn.sum()
        if (tp + fp) > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(1.)
        recall.append(tp / (tp + fn))

    return PR(precision=np.array(precision), recall=np.array(recall))

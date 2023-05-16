from collections import namedtuple

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KDTree
from scipy.io import loadmat
from tqdm import tqdm

Pairs = namedtuple("Pairs", ["distances", "real_loops", "detected_loops"])
PR = namedtuple("PR", ["precision", "recall"])


def generate_pairs(
        *,
        pair_dist: np.ndarray,
        poses: np.ndarray,
        map_tree_poses: KDTree,
        positive_distance: float = 4.0,
        negative_frames: int = 50,
        start_frame: int = 100,
        ignore_last: bool = False,
        is_distance: bool = True
) -> Pairs:
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last - 1

    if pair_dist.shape[0] < last:
        last = pair_dist.shape[0]

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


def sort_pr(
        pr: PR
) -> PR:
    recall_order = np.argsort(pr.recall)

    return PR(precision=pr.precision[recall_order], recall=pr.recall[recall_order])


def compute_precision(
        *,
        tp,
        fp,
):
    den = tp + fp
    if den > 0.:
        return tp / den
    return 1.


def compute_recall(
        *,
        tp,
        fn
):
    return tp / (tp + fn)


def compute_pr_fn_thr(
        *,
        pairs: Pairs,
        thr: float,
        positive_distance: float,
):
    asd = pairs.detected_loops <= thr
    asd = asd & pairs.real_loops
    asd = asd & (pairs.distances <= positive_distance)
    tp = asd.sum()
    fn = (pairs.detected_loops <= thr) & (pairs.distances > positive_distance) & pairs.real_loops
    fn2 = (pairs.detected_loops > thr) & pairs.real_loops
    fn = fn.sum() + fn2.sum()
    fp = (pairs.detected_loops <= thr) & (pairs.distances > positive_distance) & (1 - pairs.real_loops)
    fp = fp.sum()

    return compute_precision(tp=tp, fp=fp), compute_recall(tp=tp, fn=fn)


def compute_pr_fp_thr(
        *,
        pairs: Pairs,
        thr: float,
        positive_distance: float,
):
    asd = pairs.detected_loops <= thr
    asd = asd & pairs.real_loops
    asd = asd & (pairs.distances <= positive_distance)
    tp = asd.sum()
    fp = (pairs.detected_loops <= thr) & (pairs.distances > positive_distance)
    fp = fp.sum()
    fn = (pairs.detected_loops > thr) & pairs.real_loops
    fn = fn.sum()

    return compute_precision(tp=tp, fp=fp), compute_recall(tp=tp, fn=fn)


def compute_pr_fn(
        *,
        pairs: Pairs,
        positive_distance: float = 4.0,
        sort: bool = True
) -> PR:
    pr = np.array([list(compute_pr_fn_thr(pairs=pairs, thr=thr, positive_distance=positive_distance))
                   for thr in np.unique(pairs.detected_loops)])

    pr = PR(precision=pr[:, 0], recall=pr[:, 1])

    if sort:
        pr = sort_pr(pr)

    return pr


def compute_pr_fp(
        *,
        pairs: Pairs,
        positive_distance: float = 4.0,
        sort: bool = True,
) -> PR:
    pr = np.array([list(compute_pr_fp_thr(pairs=pairs, thr=thr, positive_distance=positive_distance))
                   for thr in np.unique(pairs.detected_loops)])

    pr = PR(precision=pr[:, 0], recall=pr[:, 1])

    if sort:
        pr = sort_pr(pr)

    return pr


def compute_pr_pairs(
        *,
        pair_dist: np.ndarray,
        poses: np.ndarray,
        is_distance: bool = True,
        ignore_last: bool = False,
        positive_distance: float = 4.,
        negative_frames: int = 50,
        start_frame: int = 100,
        ignore_below: int = -1,
        sort: bool = True
) -> PR:
    real_loop = []
    detected_loop = []
    last = poses.shape[0]
    if ignore_last:
        last = last - 1

    for i in tqdm(range(start_frame, last)):
        current_pose = poses[i][:3, 3]
        for j in range(i - negative_frames):
            candidate_pose = poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose - current_pose)
            if dist_pose <= positive_distance:
                real_loop.append(1)
            elif dist_pose <= ignore_below:
                continue
            else:
                real_loop.append(0)
            if is_distance:
                detected_loop.append(-pair_dist[i, j])
            else:
                detected_loop.append(pair_dist[i, j])
    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)

    pr = PR(precision=precision, recall=recall)

    if sort:
        pr = sort_pr(pr)

    return pr


def compute_f1_from_pr(
        precision_recall: PR
) -> np.ndarray:

    return _compute_f1(precision=precision_recall.precision, recall=precision_recall.recall)


def _compute_f1(
        *,
        precision: np.ndarray,
        recall: np.ndarray
) -> np.ndarray:

    f1 = 2 * (precision * recall) / \
         (precision + recall)

    return f1


def compute_ap_from_pr(
        precision_recall: PR,
) -> float:
    return _compute_ap(precision=precision_recall.precision, recall=precision_recall.recall)


def _compute_ap(
        *,
        precision: np.ndarray,
        recall: np.ndarray
) -> float:
    ap = np.diff(recall)
    ap = np.dot(ap, precision[1:]).item()
    return ap


def compute_ep_from_pr(
    precision_recall: PR,
) -> float:
    """ Computes the Extended Precision (EP) as defined in:

        B.Ferrarini, M. Waheed, S. Waheed, S. Ehsan, M.J. Milford, K.D. McDonald-Maier.
        Exploring Performance Bounds of Visual Place Recognition Using Extended Precision.
        RAL 2020.
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8968579

    Args:
        precision_recall:

    Returns:

    """

    return compute_ep(precision=precision_recall.precision, recall=precision_recall.recall)


def compute_ep(
        *,
        precision: np.ndarray,
        recall: np.ndarray
) -> float:
    """ Computes the Extended Precision (EP) as defined in:

        B.Ferrarini, M. Waheed, S. Waheed, S. Ehsan, M.J. Milford, K.D. McDonald-Maier.
        Exploring Performance Bounds of Visual Place Recognition Using Extended Precision.
        RAL 2020.
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8968579

    Args:
        precision:
        recall:

    Returns:

    """

    r0 = np.argmin(recall)
    p_r0 = precision[r0]

    p100 = precision == 1.0
    if np.any(p100):
        r_p100 = np.max(recall[p100])
    else:
        r_p100 = 0

    ep = (p_r0 + r_p100) / 2

    return ep


def load_pairs_file(
        path: str
) -> np.ndarray:
    ext = path.split(".")[-1].lower()

    if ext == "mat":
        return loadmat(path)["pairs_dist"]

    if ext in ["npy", "npz"]:
        pairs = np.load(path)
        if ext.lower() == "npz":
            pairs = pairs["arr_0"]

        return pairs

    raise NotImplementedError(f"Invalid extension *.{ext}.")


def unflatten_pairs_file(
        path_in: str,
        out_ext: str = "npz",
):
    pairs_dist = load_pairs_file(path_in)

    pair_h = np.sqrt(pairs_dist.shape[0])
    pair_h = int(pair_h)

    pairs_dist = pairs_dist.reshape((pair_h, pair_h))

    out_path = ".".join(path_in.split(".")[:-2]) + "." + out_ext
    np.savez(out_path, pairs_dist)


if __name__ == "__main__":
    pair_path = "~/"

    pair_files = [
        "pair_dist_iris2_freiburg.npy",
        "pair_dist_iris2_freiburg_bias.npy",
        "pairs_dist_sc_ford_seq1.npy",
        "pairs_dist_sc_ford_seq1_bias.npy"
    ]

    for pair_file in pair_files:
        unflatten_pairs_file(pair_path + pair_file)

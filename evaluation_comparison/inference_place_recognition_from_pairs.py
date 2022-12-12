from collections import namedtuple

from evaluation_comparison.metrics.detection import load_pairs_file
from evaluation_comparison.inference_placerecognition_general import load_poses as load_poses_general
from evaluation_comparison.inference_placerecognition_ford import load_poses as load_poses_ford


PairFile = namedtuple("PairFile", ["path", "method", "dataset"])
Method = namedtuple("Method", ["method", "is_distance"])
Dataset = namedtuple("Dataset", ["dataset", "path", "pose_loader", "positive_distance", "negative_frames", "start_frame", ])

ds_kitti = Dataset(dataset="kitti", path=ds_path + "kitti", pose_loader=load_poses_general)

mtd_m2dp = Method(method="M2DP", is_distance=True)

_pair_files = [
    PairFile(path=pair_root, method= )
]

pair_files = {p.path: p for p in _pair_files}


def eval_pair_file(
        pair_file: PairFile
):

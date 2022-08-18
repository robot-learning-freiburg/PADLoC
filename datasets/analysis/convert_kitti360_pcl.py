from argparse import ArgumentParser
from pathlib import Path

import open3d as o3d

from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DPoses
from datasets.KITTI360Classes import color_points


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=Path, default=Path("/home/arceyd/MT/dat/kitti360/"))
    parser.add_argument("--sequence", type=str, default="2013_05_28_drive_0000_sync")
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--loop_file", type=str, default="loop_GT_4m_noneg")
    parser.add_argument("--output_path", type=Path)

    args = parser.parse_args()

    return vars(args)


def export_pcl(data_path, frame, output_path, sequence, loop_file):
    dataset = KITTI3603DDictPairs(data_path, sequence=sequence, loop_file=loop_file, use_panoptic=True)
    d2 = KITTI3603DPoses(data_path, sequence=sequence, loop_file=loop_file, use_panoptic=True)

    pcl = dataset[frame]

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(pcl["anchor"][:, :3])
    pc1.colors = o3d.utility.Vector3dVector(color_points(pcl["anchor_semantic"]))

    print(f"Saving Semantic PointCloud to {output_path}.")
    o3d.io.write_point_cloud(str(output_path), pc1)


if __name__ == "__main__":
    export_pcl(**cli_args())

from argparse import ArgumentParser
from pathlib import Path

import open3d as o3d

from datasets.Freiburg import FreiburgDataset


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=Path, default="/home/arceyd/MT/dat/haomo/sequences/")
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--output_path", type=Path)

    args = parser.parse_args()

    return vars(args)


def export_pcl(data_path, frame, output_path):
    dataset = FreiburgDataset(data_path)

    pcl = dataset[frame]

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(pcl["anchor"][:, :3])
    o3d.io.write_point_cloud(str(output_path), pc1)


if __name__ == "__main__":
    export_pcl(**cli_args())

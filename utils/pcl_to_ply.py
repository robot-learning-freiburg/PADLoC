#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import open3d as o3d
import numpy as np


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("in_file", type=Path)
    parser.add_argument("out_file", type=Path)

    args = parser.parse_args()

    return vars(args)


def main(in_file, out_file):

    scan = np.fromfile(in_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    scan = scan[:, :3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(scan)
    o3d.io.write_point_cloud(str(out_file), pc)


if __name__ == "__main__":
    main(**cli_args())

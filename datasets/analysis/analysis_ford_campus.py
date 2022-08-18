from argparse import ArgumentParser
from pathlib import Path

from datasets.analysis.plot_utils import plot_2_5d_trajectory, plot_2d_trajectory
from datasets.FordCampus import FordCampusDataset


def main(dataset_path, without_ground=False):

    sequences = FordCampusDataset.SEQUENCES.keys()
    sequences = ["1"]

    for seq in sequences:
        dataset = FordCampusDataset(str(dataset_path), seq, without_ground=without_ground)

        positions = dataset.poses[:, :3, 3]

        plot_2d_trajectory(positions, save_path=dataset_path / f"ford_traj2d_seq{seq}_all.pdf")
        plot_2_5d_trajectory(positions, save_path=dataset_path / f"ford_traj2.5d_seq{seq}_all.pdf")


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", type=Path, default="/home/arceyd/Documents/Datasets/Ford Campus")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

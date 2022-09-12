from argparse import ArgumentParser

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def subplot_loop_angles(ax, angles, n_bins=32, density=True, color=None):
    bin_width = 2 * np.pi / n_bins

    range_start = - bin_width / 2
    range_end = range_start + 2 * np.pi

    angles[angles < range_start] += 2 * np.pi
    angles[angles > range_end] -= 2 * np.pi

    bins = np.linspace(range_start, range_end, n_bins + 1)

    a, b = np.histogram(angles, bins=bins)
    centers = bins[:-1] + bin_width / 2

    a = a / a.sum()

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        # Calculate corresponding bin radius
        a = (a / np.pi) ** 0.5
    # Otherwise plot frequency proportional to radius
    else:
        a = a

    ax.bar(centers, a, width=bin_width, bottom=0.0, align="center", color=color)


def plot_loop_angles(angles, n_bins=32, density=True, save_path=None):

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="polar")

    rough_colors = [mcolors.TABLEAU_COLORS[c] for c in ["tab:green", "tab:purple", "tab:orange", "tab:purple"]]
    fine_color = mcolors.TABLEAU_COLORS["tab:blue"]

    subplot_loop_angles(ax, angles, n_bins=4, density=density, color=rough_colors)
    subplot_loop_angles(ax, angles, n_bins=n_bins, density=density, color=fine_color)

    if density:
        ticks = np.linspace(0.1, 1.0, 10)
        tick_labels = [f"{t:.1f}" for t in ticks]
        ax.set_rgrids((ticks / np.pi) ** 0.5, labels=tick_labels, angle=90)

    if save_path is not None:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--stats_path", type=str,
                        default="/home/arceyd/Documents/Datasets/KITTI360/analysis/kitti360stats.pickle")

    args = parser.parse_args()

    return vars(args)


def main(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    save_dir = "/".join(stats_path.split("/")[:-1])
    for seq, seq_dict in stats.items():
        for gt, stats_dict in seq_dict.items():

            save_path = os.path.join(save_dir, f"loop_angles_dist_{gt}_{seq}.pdf")

            plot_loop_angles(stats_dict["avg_pair_yaw"], save_path=save_path)


if __name__ == "__main__":
    main(**cli_args())

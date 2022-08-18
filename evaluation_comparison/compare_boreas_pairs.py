from argparse import ArgumentParser
from collections import namedtuple
from itertools import chain
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pickle

from evaluation_comparison.boreas_utils import SEQUENCES, SEQUENCE_PAIRS, yaw_stats_filename, lcd_stats_filename


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--save_path", type=Path,
                        default="/home/arceyd/Documents/Projects/PADLoC/res/boreas_comparison/")
    parser.add_argument("--do_plot", action="store_true")
    parser.add_argument("--n_best", type=int, default=5)

    args = parser.parse_args()

    return vars(args)


def plot_matrix(arr, title, cb_title, ticks, save_path, cmap="RdBu", autorange=True):
    fig, ax = plt.subplots()

    vmin, vmax = None, None
    if not autorange:
        vrange = max(np.abs(np.nanmin(arr)), np.abs(np.nanmax(arr)))
        vmin, vmax = -vrange, vrange

    cmap = mpl.cm.get_cmap(cmap).copy()
    cmap.set_bad(color="0.75")

    plot = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    yticks = [f"({i:02d}) {s}" for i, s in enumerate(SEQUENCES)]
    xticks = [f"({i:02d})" for i in range(len(SEQUENCES))]
    ax.set_xticks(ticks, xticks)
    ax.set_yticks(ticks, yticks)

    # Minor ticks
    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)

    ax.xaxis.tick_top()
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='0.5', linestyle='-', linewidth=0.5)

    cb = fig.colorbar(plot, ax=ax)
    cb.ax.set_title(cb_title)

    plt.setp(ax.get_xticklabels(), rotation=90)  # , ha="right",# rotation_mode="anchor")
    ax.tick_params(labelsize=6)

    print(f"Saving fig {save_path}")
    plt.savefig(save_path)
    plt.close(fig)


def load_stats_file(filename, seq1, seq2, model_name, file_type):
    stats = {}

    try:
        if filename.exists():
            print(f"Reading {model_name}'s {file_type} stats file for seq {seq1} and {seq2}.")
            with open(filename, "rb") as f1:
                stats = pickle.load(f1)

    except EOFError:
        print(f"ERROR: Exception occurred while reading file {filename}!")

    return stats


def fill_arrays_in(array_cfg_list, lcd_stats, pad_stats, seq1i, seq2i):
    for arr_cfg in array_cfg_list:
        if lcd_stats:
            arr_cfg.lcd_array[seq1i, seq2i] = lcd_stats[arr_cfg.stats_key]
        if pad_stats:
            arr_cfg.pad_array[seq1i, seq2i] = pad_stats[arr_cfg.stats_key]
        if lcd_stats and pad_stats:
            arr_cfg.diff_array[seq1i, seq2i] = lcd_stats[arr_cfg.stats_key] - pad_stats[arr_cfg.stats_key]
            if arr_cfg.bigger_is_better:
                arr_cfg.diff_array[seq1i, seq2i] = - arr_cfg.diff_array[seq1i, seq2i]


def main(save_path, do_plot=False, n_best=5):
    seq_idx = {s: i for i, s in enumerate(SEQUENCES)}
    n_seqs = len(SEQUENCES)
    shape = (n_seqs, n_seqs)
    ticks = np.arange(n_seqs)

    DiffArray = namedtuple("diffarray",
                           ["label", "diff_array", "lcd_array", "pad_array", "stats_key", "units",
                            "bigger_is_better", "weight_diff", "weight_lcd", "weight_pad",
                            "bias_diff", "bias_lcd", "bias_pad"])

    diff_arr_cfgs = [
        DiffArray("Rot Mean", np.empty(shape), np.empty(shape), np.empty(shape), "Mean rotation error", "deg",
                  # False, 1./5., -1./3., 0., 0.5, 5./3., 0.),
                  False, 0., 0., 0., 0., 0., 0.),
        DiffArray("Rot Median", np.empty(shape), np.empty(shape), np.empty(shape), "Median rotation error", "deg",
                  False, 0., 0., 0., 0., 0., 0.),
        DiffArray("Rot Std", np.empty(shape), np.empty(shape), np.empty(shape), "STD rotation error", "deg",
                  False, 0., 0., 0., 0., 0., 0.),
        DiffArray("Tra Mean", np.empty(shape), np.empty(shape), np.empty(shape), "Mean translation error", "m",
                  False, 1./3., -1./(4.6 - 3.), 0., 0.5, 4.6 / (4.6 - 3.), 0.,),
        DiffArray("Tra Median", np.empty(shape), np.empty(shape), np.empty(shape), "Median translation error", "m",
                  False, 0., 0., 0., 0., 0., 0.),
        DiffArray("Tra Std", np.empty(shape), np.empty(shape), np.empty(shape), "STD translation error", "m",
                  False, 0., 0., 0., 0., 0., 0.),
    ]

    diff_ap_arr_cfgs = [
        DiffArray("AP FP", np.empty(shape), np.empty(shape), np.empty(shape), "AP_FP", "%",
                  True, 1./.6, 1., 0., 0.5, 0., 0.),
        DiffArray("AP FN", np.empty(shape), np.empty(shape), np.empty(shape), "AP_FN", "%",
                  True, 0., 0., 0., 0., 0., 0.),
    ]

    for arr_cfg in chain(diff_arr_cfgs, diff_ap_arr_cfgs):
        arr_cfg.diff_array[:] = np.nan
        arr_cfg.lcd_array[:] = np.nan
        arr_cfg.pad_array[:] = np.nan

    print("\n")
    print("=" * 80)
    print("Reading Stats Files")
    print("=" * 80 + "\n")
    for seq1, seq2 in SEQUENCE_PAIRS:
        seq1i, seq2i = seq_idx[seq1], seq_idx[seq2]

        yaw_file_lcdnet = yaw_stats_filename(seq1, seq2, "lcdnet", save_path / "reg_stats")
        yaw_file_padloc = yaw_stats_filename(seq1, seq2, "padloc", save_path / "reg_stats")

        lcdnet_yaw_stats = load_stats_file(yaw_file_lcdnet, seq1, seq2, "LCDNet", "REG")
        padloc_yaw_stats = load_stats_file(yaw_file_padloc, seq1, seq2, "PADLoC", "REG")

        fill_arrays_in(diff_arr_cfgs, lcdnet_yaw_stats, padloc_yaw_stats, seq1i, seq2i)

        pr_file_lcdnet = lcd_stats_filename(seq1, seq2, "lcdnet", save_path / "pr_stats")
        pr_file_padloc = lcd_stats_filename(seq1, seq2, "padloc", save_path / "pr_stats")

        lcdnet_pr_stats = load_stats_file(pr_file_lcdnet, seq1, seq2, "LCDNet", "PR")
        padloc_pr_stats = load_stats_file(pr_file_padloc, seq1, seq2, "PADLoC", "PR")

        fill_arrays_in(diff_ap_arr_cfgs, lcdnet_pr_stats, padloc_pr_stats, seq1i, seq2i)

    if do_plot:
        print("\n\n")
        print("=" * 80)
        print("Saving Plots")
        print("=" * 80 + "\n")
    scores = np.zeros(shape)
    for arr_cfg in chain(diff_arr_cfgs, diff_ap_arr_cfgs):
        if do_plot:
            plot_matrix(arr_cfg.diff_array, f"{arr_cfg.label} Diff", f"PADLoC\nBetter\n[{arr_cfg.units}]", ticks,
                        save_path / f"{arr_cfg.label} diff.pdf", "RdBu", autorange=False)
            plot_matrix(arr_cfg.lcd_array, f"{arr_cfg.label} LCDNet", f"[{arr_cfg.units}]", ticks,
                        save_path / f"{arr_cfg.label} lcdnet.pdf", "plasma", autorange=True)
            plot_matrix(arr_cfg.pad_array, f"{arr_cfg.label} PADLoC", f"[{arr_cfg.units}]", ticks,
                        save_path / f"{arr_cfg.label} padloc.pdf", "plasma", autorange=True)

        if not n_best:
            continue

        # Compute weighted score
        if arr_cfg.weight_diff or arr_cfg.bias_diff:
            scores += arr_cfg.weight_diff * arr_cfg.diff_array + arr_cfg.bias_diff

        if arr_cfg.weight_lcd or arr_cfg.bias_lcd:
            scores += arr_cfg.weight_lcd * arr_cfg.lcd_array + arr_cfg.bias_lcd

        if arr_cfg.weight_pad or arr_cfg.bias_pad:
            scores += arr_cfg.weight_pad * arr_cfg.pad_array + arr_cfg.bias_pad

    if not n_best:
        return

    # Maximize score and return best n pairs
    print("\n\n")
    print("="*80)
    print("Computing best pairs")
    print("=" * 80 + "\n")

    flat_scores = scores.flatten()
    best_n_idx = np.argsort(-flat_scores)[:n_best]
    best_n_idx = np.unravel_index(best_n_idx, shape)

    for i in range(n_best):
        idx1 = best_n_idx[0][i]
        idx2 = best_n_idx[1][i]

        seq1 = SEQUENCES[idx1]
        seq2 = SEQUENCES[idx2]

        curr_score = scores[idx1, idx2]

        txt = f"{i:02d}, IDX: ({idx1:02d}, {idx2:02d}), SEQS:{seq1} and {seq2}, Score {curr_score:.3f}"

        for arr_cfg in chain(diff_arr_cfgs, diff_ap_arr_cfgs):
            txt2 = ""
            if arr_cfg.weight_diff or arr_cfg.bias_diff:
                txt2 += f" Diff {arr_cfg.diff_array[idx1, idx2]:.3f}"

            if arr_cfg.weight_lcd or arr_cfg.bias_lcd:
                txt2 += f" LCDNet {arr_cfg.lcd_array[idx1, idx2]:.3f}"

            if arr_cfg.weight_pad or arr_cfg.bias_pad:
                txt2 += f" PADLoC {arr_cfg.pad_array[idx1, idx2]:.3f}"

            if txt2:
                txt += f", {arr_cfg.label} {txt2}"

        print(txt)


if __name__ == "__main__":
    main(**cli_args())

from argparse import ArgumentParser
from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pickle

from evaluation_comparison.inference_boreas_paper import SEQUENCES, SEQUENCE_PAIRS, MODELS, yaw_stats_filename, pr_filename


def cli_args():
	parser = ArgumentParser()

	parser.add_argument("--save_path", type=Path, default="/home/arceyd/MT/res/boreas_comparison/")

	args = parser.parse_args()

	return vars(args)




def main(save_path):

	seq_idx = {s: i for i, s in enumerate(SEQUENCES)}
	n_seqs = len(SEQUENCES)
	shape = (n_seqs, n_seqs)
	ticks = np.arange(n_seqs)

	rot_mea_diff = np.zeros(shape)
	rot_std_diff = np.zeros(shape)
	tra_mea_diff = np.zeros(shape)
	tra_std_diff = np.zeros(shape)
	lcd_ap_diff = np.zeros(shape)

	DiffArray = namedtuple("diffarray", ["label", "array", "stats_key", "reversed"])

	diff_arr_cfgs = [
		DiffArray("Rot Mean", rot_mea_diff, "Mean rotation error", False),
		DiffArray("Rot Std",  rot_std_diff, "STD rotation error", False),
		DiffArray("Tra Mean", tra_mea_diff, "Mean translation error", False),
		DiffArray("Tra Std",  tra_std_diff, "STD translation error", False),
	]

	for seq1, seq2 in SEQUENCE_PAIRS:

		seq1i, seq2i = seq_idx[seq1], seq_idx[seq2]

		yaw_file_lcdnet = yaw_stats_filename(seq1, seq2, "lcdnet", save_path)
		yaw_file_padloc = yaw_stats_filename(seq1, seq2, "padloc", save_path)

		if not (yaw_file_lcdnet.exists() and yaw_file_padloc.exists()):
			continue

		with open(yaw_file_lcdnet, "rb") as f1:
			lcdnet_yaw_stats = pickle.load(f1)
		with open(yaw_file_padloc, "rb") as f2:
			padloc_yaw_stats = pickle.load(f2)

		for arr in diff_arr_cfgs:
			arr.array[seq1i, seq2i] = padloc_yaw_stats[arr.stats_key] - lcdnet_yaw_stats[arr.stats_key]
			if arr.reversed:
				arr.array[seq1i, seq2i] = - arr.array[seq1i, seq2i]

	for arr in diff_arr_cfgs:
		fig, ax = plt.subplots()

		vrange = max(np.abs(np.min(arr.array)), np.abs(np.max(arr.array)))

		ax.imshow(arr.array, cmap="RdBu", vmin=-vrange, vmax=vrange)

		ax.set_title(arr.label)
		ax.set_xticks(ticks, SEQUENCES)
		ax.set_yticks(ticks, SEQUENCES)

		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				 rotation_mode="anchor")

		print(f"Saving fig {arr.label}")
		plt.savefig(save_path / f"{arr.label}.pdf")
		plt.close(fig)


if __name__ == "__main__":
	main(**cli_args())

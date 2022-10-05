from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
import torch

from evaluation_comparison.plot_path import compute_PR
from evaluation_comparison.plot_styles import Style
from datasets.KITTI_data_loader import KITTILoader3DPoses
from datasets.KITTI360Dataset import KITTI3603DPoses


def plot_legends(handles, legends, filename, expand=None, ncol=4):

	tmp_fig, tmp_ax = plt.subplots()
	tmp_ax.axis("off")

	legend = plt.legend(handles, legends, loc=3, framealpha=1, frameon=True, ncol=ncol)

	if expand is None:
		expand = [-5, -5, 5, 5]

	fig = legend.figure
	fig.canvas.draw()
	bbox = legend.get_window_extent()
	bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
	bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

	legend_dir = filename.parent
	legend_name = filename.stem + "_legend"
	legend_sfxs = ".pdf"
	legend_path = legend_dir / f"{legend_name}{legend_sfxs}"

	print(f"Saving plot legend to file {legend_path}.")
	fig.savefig(legend_path, dpi="figure", bbox_inches=bbox)
	plt.close(fig)


def plot_path(poses, tp, fp, fn, style, save_path=None, save_stats=True, do_plot_legends=False, legend_cols=4):
	fig = plt.figure(dpi=style.path_dpi, figsize=(5.8, 4.1))
	ax = fig.add_subplot()
	ax.axis("off")
	ax.set_aspect("equal")
	# ax.plot(poses[100:, 0, 3], poses[100:, 1, 3], **style.tn)
	h1 = ax.plot(poses[:, 0, 3], poses[:, 1, 3], zorder=1, **style.tn)
	h1 = h1[0]

	fp_indexes = np.nonzero(fp)[0] + 100
	fn_indexes = np.nonzero(fn)[0] + 100
	tp_indexes = np.nonzero(tp)[0] + 100

	h2 = ax.scatter(poses[fp_indexes, 0, 3], poses[fp_indexes, 1, 3], zorder=4, **style.fp)
	h3 = ax.scatter(poses[fn_indexes, 0, 3], poses[fn_indexes, 1, 3], zorder=3, **style.fn)
	h4 = ax.scatter(poses[tp_indexes, 0, 3], poses[tp_indexes, 1, 3], zorder=2, **style.tp)

	if save_stats:
		stats = dict(TP=len(tp_indexes), FP=len(fp_indexes), FN=len(fn_indexes))
		stats_dir = save_path.parent / f"{save_path.stem}_stats.json"
		with open(stats_dir, "w") as f:
			json.dump(stats, f)

	if save_path:
		print(f"Saving plot to {save_path}.")
		fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
		plt.close(fig)

		if do_plot_legends:
			plot_legends([h1, h4, h3, h2], ["TN", "TP", "FN", "FP"], save_path, ncol=legend_cols)
	else:
		plt.show()


def arg_path(value):
	return Path(value).expanduser()


def arg_seq(value):
	if isinstance(value, str):
		return value

	return f"{value:02d}"


def arg_style_list(value):
	if value is None:
		return value

	style_keys = value.split(",")
	styles = [Style(s, use_latex=False) for s in style_keys]
	return styles


def img_style_filename(style, path):
	img_dir = path.parent
	img_file = path.stem.split("_")
	img_file.insert(1, style.pfx)
	img_file = "_".join(img_file)
	img_ext = "".join(path.suffixes)
	img_style_path = img_dir / f"{img_file}{img_ext}"

	return img_style_path


def parse_args():
	parser = ArgumentParser()

	parser.add_argument("--pairs_file", "-p", type=arg_path)
	parser.add_argument("--dataset", "-t", type=str, default="kitti")
	parser.add_argument("--dataset_path", "-d", type=arg_path, default=Path("/data/arceyd/kitti"))
	parser.add_argument("--sequence", "-s", type=arg_seq, default="08")
	parser.add_argument("--styles", "-v", type=arg_style_list, default="bw")
	parser.add_argument("--save_path", "-f", type=arg_path, default=None)
	parser.add_argument("--do_plot_legends", "-l", action="store_true")

	args = parser.parse_args()

	return vars(args)


def main(pairs_file, dataset, dataset_path, sequence, styles, save_path=None, save_stats=True, do_plot_legends=False,
		 is_distance=True):
	# sequence = "00"
	# dataset_path = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY'
	device = torch.device("cuda:0")

	pd_ours = np.load(pairs_file)
	pd_ours = pd_ours["arr_0"]

	if dataset == 'kitti':
		dataset_for_recall = KITTILoader3DPoses(dataset_path, sequence,
												dataset_path / "sequences" / sequence / "poses.txt",
												4096, device, train=False,
												without_ground=False, loop_file="loop_GT_4m")
	elif dataset == "kitti360":
		dataset_for_recall = KITTI3603DPoses(dataset_path, sequence,
											 train=False,
											 without_ground=False, loop_file="loop_GT_4m_noneg")
	else:
		raise NotImplementedError(f"Invalid dataset {dataset}")

	poses = np.stack(dataset_for_recall.poses)
	map_tree_poses = KDTree(poses[:, :3, 3])

	tp, fp, fn = compute_PR(pd_ours, poses, map_tree_poses, is_distance=is_distance)

	for style in styles:
		plot_path(poses, tp, fp, fn, style, img_style_filename(style, save_path),
				  save_stats=save_stats, do_plot_legends=do_plot_legends)


if __name__ == "__main__":
	main(**parse_args())

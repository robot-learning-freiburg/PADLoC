from collections import namedtuple
from pathlib import Path

from evaluation_comparison.plot_path_from_pairs import main as plot_pairs
from evaluation_comparison.plot_styles import Style


PairFile = namedtuple("pair_file", ["file", "dataset", "seq"])
Dataset = namedtuple("dataset", ["path"])


def main():

	# dataset_path = Path("/data/arceyd/kitti/")
	home_dir = Path("~/MT/res/place_recognition/").expanduser()
	save_dir = home_dir

	pair_file_ext = ".npz"
	fig_ext = ".png"

	pair_files = [
		# Kitti
		# PairFile("dcp_kitti_seq00", "kitti", "00"),
		# PairFile("dcp_kitti_seq08", "kitti", "08"),
		# PairFile("lcdnet_kitti_seq00", "kitti", "00"),
		# PairFile("lcdnet_kitti_seq08", "kitti", "08"),
		# PairFile("padloc_kitti_seq00", "kitti", "00"),
		# PairFile("padloc_kitti_seq08", "kitti", "08"),
		# PairFile("tf_kitti_seq00", "kitti", "00"),
		# PairFile("tf_kitti_seq08", "kitti", "08"),
		# Kitti360
		PairFile("dcp_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
		PairFile("lcdnet_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
		PairFile("padloc_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
		PairFile("tf_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
	]

	datasets = {
		"kitti": Path("/data/arceyd/kitti/"),
		"kitti360": Path("/home/arceyd/MT/dat/kitti360/")
	}

	styles = [
		Style("bw"),
		Style("color")
	]

	for pair_file in pair_files:
		file_path = home_dir / f"{pair_file.file}{pair_file_ext}"
		save_path = save_dir / f"fig_lcd_{pair_file.file}{fig_ext}"
		plot_pairs(file_path,
				   dataset=pair_file.dataset, dataset_path=datasets[pair_file.dataset], sequence=pair_file.seq,
				   styles=styles, save_path=save_path)


if __name__ == "__main__":
	main()

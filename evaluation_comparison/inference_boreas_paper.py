import os
from argparse import ArgumentParser
from collections import namedtuple
from itertools import combinations
from pathlib import Path
import torch
import torch.multiprocessing as mp

from evaluation_comparison.inference_yaw_general_boreas import main_process as inference_yaw
from evaluation_comparison.inference_placerecognition_mulran import main_process as inference_lcd

Model = namedtuple("model", ["dir", "cp", "label", "batch_size"])

SEQUENCES = [
	"2020-11-26-13-58",	 # Overcast, Snow
	"2020-12-01-13-26",	 # Overcast, Snow, Snowing
	"2020-12-04-14-00",	 # Overcast, Snow
	"2020-12-18-13-44",	 # Sun, Snow
	"2021-01-15-12-17",	 # Sun, Clouds, Snow
	"2021-01-19-15-08",	 # Clouds, Snow
	"2021-01-26-10-59",	 # Overcast, Snow, Snowing
	"2021-01-26-11-22",	 # Overcast, Snow, Snowing
	"2021-02-02-14-07",	 # Overcast, Snow
	"2021-02-09-12-55",	 # Sun, Clouds, Snow
	"2021-03-02-13-38",	 # Sun, Clouds, Snow
	"2021-03-09-14-23",	 # Sun
	"2021-03-23-12-43",	 # Overcast, Construction
	"2021-03-30-14-23",	 # Sun, Clouds, Construction
	"2021-04-08-12-44",	 # Sun
	"2021-04-13-14-49",	 # Sun, Clouds, Construction
	"2021-04-15-18-55",	 # Clouds, Construction
	"2021-04-20-14-11",	 # Clouds, Construction
	"2021-04-22-15-00",	 # Clouds, Snowing, Construction
	"2021-04-29-15-55",	 # Overcast, Rain
	"2021-05-06-13-19",	 # Sun, Clouds
	"2021-05-13-16-11",	 # Sun, Clouds
	"2021-06-03-16-00",	 # Sun, Clouds
	"2021-06-17-17-52",	 # Sun
	"2021-06-29-18-53",	 # Overcast, Rain
	"2021-06-29-20-43",	 # Sun, Clouds, Dusk
	"2021-07-20-17-33",	 # Clouds, Rain
	"2021-07-27-14-43",	 # Clouds
	"2021-08-05-13-34",	 # Sun, Clouds
	"2021-09-02-11-42",	 # Sun
	"2021-09-07-09-35",	 # Sun
	"2021-09-08-21-00",	 # Night
	"2021-09-09-15-28",	 # Sun, Clouds, Alternate, Construction
	"2021-09-14-20-00",	 # Night
	"2021-10-05-15-35",	 # Overcast
	"2021-10-15-12-35",	 # Clouds
	"2021-10-22-11-36",	 # Clouds
	"2021-10-26-12-35",	 # Overcast, Rain
	"2021-11-02-11-16",	 # Sun, Clouds
	"2021-11-06-18-55",	 # Night
	"2021-11-14-09-47",	 # Overcast
	"2021-11-16-14-10",	 # Clouds
	"2021-11-23-14-27",	 # Sun, Clouds
	"2021-11-28-09-18",  # Overcast, Snow, Snowing
]

SEQUENCE_PAIRS = list(combinations(SEQUENCES, 2))

MODELS = [
	Model(dir="16-09-2021_00-02-34", cp="best_model_so_far_rot.tar", label="lcdnet", batch_size=15),
	Model(dir="27-05-2022_19-10-54", cp="best_model_so_far_rot.tar", label="padloc", batch_size=15)
]


def out_filename(seq1, seq2, model_name, path, proc, dataset="boreas", ext="pickle"):
	return path / f"{model_name}_{dataset}_{proc}_seqs_{seq1}_{seq2}.{ext}"


def yaw_stats_filename(seq1, seq2, model_name, path, dataset="boreas", ext="pickle"):
	return out_filename(seq1, seq2, model_name, path, "inference_yaw_stats", dataset, ext)


def pr_filename(seq1, seq2, model_name, path, dataset="boreas", ext="npz"):
	return out_filename(seq1, seq2, model_name, path, "inference_pr", dataset, ext)


def seq_pair_downloaded(seq1, seq2, path: Path):
	tmp_path1 = path / seq1
	tmp_path2 = path / seq2
	return tmp_path1.exists() and tmp_path2.exists()


# def multi_eval(func, output_naming_func, sequence_pairs,
# dataset_path, models, cp_path, save_path, gpus, batch_size=1):
# 	for seq1, seq2 in sequence_pairs:
# 		seq1_dir = f"boreas-{seq1}"
# 		seq2_dir = f"boreas-{seq2}"
#
# 		if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
# 			print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
# 			continue
#
# 		for model in models:
# 			output_file = output_naming_func(seq1, seq2, model.label, save_path)
#
# 			if output_file.exists():
# 				print(f"{func} already executed on pair {seq1} - {seq2}. Skipping.")
# 				continue
#
# 			inference_yaw(gpus[0], cp_path / model.dir / model.cp,
# 						  dataset_path=dataset_path, seq1=seq1_dir, seq2=seq2_dir,
# 						  save_path=yaw_stats_file, batch_size=batch_size)
#
# 			pr_file = pr_filename(seq1, seq2, model.label, save_path)
# 			if not pr_file.exists():
# 				os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[1:])
# 				mp.spawn(inference_lcd, nprocs=len(gpus) - 1, )


def eval_reg(dataset_path, models, cp_path, save_path):
	for seq1, seq2 in SEQUENCE_PAIRS:
		seq1_dir = f"boreas-{seq1}"
		seq2_dir = f"boreas-{seq2}"

		if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
			print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
			continue

		for model in models:
			yaw_stats_file = yaw_stats_filename(seq1, seq2, model.label, save_path)

			if yaw_stats_file.exists():
				print(f"Registration Eval already executed on pair {seq1} - {seq2} (see {yaw_stats_file.stem}). Skipping.")
				continue

			inference_yaw(0, cp_path / model.dir / model.cp,
							  dataset_path=dataset_path, seq1=seq1_dir, seq2=seq2_dir,
							  save_path=yaw_stats_file, batch_size=model.batch_size)


def eval_lcd(dataset_path, models, cp_path, save_path):

	gpus = torch.cuda.device_count()

	for seq1, seq2 in SEQUENCE_PAIRS:
		seq1_dir = f"boreas-{seq1}"
		seq2_dir = f"boreas-{seq2}"

		if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
			print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
			continue

		for model in models:
			pr_file = pr_filename(seq1, seq2, model.label, save_path)

			if pr_file.exists():
				print(f"Registration Eval already executed on pair {seq1} - {seq2} (see {pr_filename.stem}). Skipping.")
				continue

			mp.spawn(inference_lcd, nprocs=gpus, args=(cp_path / model.dir / model.cp, 42, gpus,
													   dataset_path, "boreas", seq1_dir, seq2_dir, 1,
													   model.batch_size))


def cli_args():
	parser = ArgumentParser()

	parser.add_argument("--do_eval_lcd", action="store_true")

	parser.add_argument("--dataset_path", type=Path, default="/home/cattaneo/Datasets/boreas/")
	parser.add_argument("--save_path", type=Path, default="/home/arceyd/MT/res/boreas_comparison/")
	parser.add_argument("--cp_path", type=Path, default="/home/arceyd/MT/cp/3D/")

	args = parser.parse_args()

	return vars(args)


def main_process(do_eval_lcd, dataset_path, save_path, cp_path):

	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '8989'

	if do_eval_lcd:
		return eval_lcd(dataset_path, MODELS, cp_path, save_path)
	else:
		return eval_reg(dataset_path, MODELS, cp_path, save_path)


if __name__ == "__main__":
	main_process(**cli_args())

import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path
import torch
import torch.multiprocessing as mp

from evaluation_comparison.inference_yaw_general_boreas import main_process as inference_yaw
from evaluation_comparison.inference_placerecognition_mulran import main_process as inference_lcd
from evaluation_comparison.boreas_utils import SEQUENCE_PAIRS, seq_pair_downloaded, yaw_stats_filename,pr_filename, lcd_stats_filename

Model = namedtuple("model", ["dir", "cp", "label", "batch_size"])

MODELS_YAW = [
	Model(dir="16-09-2021_00-02-34", cp="best_model_so_far_rot.tar", label="lcdnet", batch_size=20),
	Model(dir="27-05-2022_19-10-54", cp="best_model_so_far_rot.tar", label="padloc", batch_size=20)
]

MODELS_LCD = [
	Model(dir="16-09-2021_00-02-34", cp="best_model_so_far_auc.tar", label="lcdnet", batch_size=20),
	Model(dir="27-05-2022_19-10-54", cp="best_model_so_far_auc.tar", label="padloc", batch_size=20)
]


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

			stdout = sys.stdout
			null = open(os.devnull, "w")
			sys.stdout = null
			try:
				inference_yaw(0, cp_path / model.dir / model.cp,
							  dataset_path=dataset_path, seq1=seq1_dir, seq2=seq2_dir,
							  save_path=yaw_stats_file, batch_size=model.batch_size)
			except Exception as e:
				print(f"Exception when evaluating pair {seq1} - {seq2}. Skipping. \n{e}")
			sys.stdout = stdout


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
			lcd_stats_file = lcd_stats_filename(seq1, seq2, model.label, save_path)

			if pr_file.exists():
				print(f"Registration Eval already executed on pair {seq1} - {seq2} (see {pr_file.stem}). Skipping.")
				continue

			mp.spawn(inference_lcd, nprocs=gpus, args=(cp_path / model.dir / model.cp, 42, gpus,
													   dataset_path, "boreas", seq1_dir, seq2_dir, 1,
													   model.batch_size, pr_file, lcd_stats_file), join=True)


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
		return eval_lcd(dataset_path, MODELS_LCD, cp_path, save_path)
	else:
		return eval_reg(dataset_path, MODELS_YAW, cp_path, save_path)


if __name__ == "__main__":
	main_process(**cli_args())

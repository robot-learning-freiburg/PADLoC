from collections import namedtuple
import os
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle

from evaluation_comparison.inference_yaw_general import main_process as eval_registration


Run = namedtuple("run", ["name", "label"])


def save_stats(file_path, eval_stats, key):
	if file_path.exists():
		print(f"Appending stats for {key} to {file_path}.")
		with open(file_path, "rb") as f:
			stats = pickle.load(f)

		os.remove(file_path)
	else:
		print(f"Saving stats for {key} in {file_path}.")
		stats = {}

	stats[key] = eval_stats

	with open(file_path, "wb") as f:
		pickle.dump(stats, f)


def main(data_path="/data/arceyd/kitti/", save_path="~/MT/res/",
		 model_path="~/MT/cp/3D/", checkpoint="best_model_so_far_rot.tar",
		 keys=None,
		 num_iters=1, dataset="kitti", do_ransac=False, do_icp=False):
	gpu = 0

	cfg_dict = {
		"ablation_losses": [
			Run("16-09-2021_00-02-34", "LCDNet"),
			Run("18-05-2022_14-57-01", "LCDNet+Rev"),
			Run("20-05-2022_02-20-25", "LCDNet+Rev+Sem"),
			Run("23-05-2022_05-26-27", "LCDNet+Rev+Mse"),
			Run("21-05-2022_15-55-37", "LCDNet+Rev+Pan"),
			Run("17-05-2022_00-20-38", "LCDNet+Rev+Pan+Sem+Mse"),
			Run("12-05-2022_10-38-29", "TF"),
			Run("15-05-2022_10-59-41", "TF+Rev"),
			Run("13-05-2022_18-02-51", "TF+Rev+Sem"),
			Run("08-05-2022_23-35-45", "TF+Rev+Mse"),
			Run("01-05-2022_11-33-12", "TF+Rev+Pan"),
			Run("10-05-2022_13-24-39", "TF+Rev+Pan+Sem+Mse")
		],
		"ablation_weight": [
			Run("24-05-2022_23-08-23", "Uniform"),
			Run("25-05-2022_10-51-25", "Column Sum"),
			Run("25-05-2022_23-04-41", "Shannon"),
			Run("26-05-2022_09-55-19", "Hill, order=2"),
			Run("26-05-2022_20-52-10", "Hill, order=4"),
			Run("27-05-2022_07-46-12", "Berger-Parker")
		]
	}

	if keys:
		cfg_dict = {k: cfg_dict[k] for k in keys}

	data_path = Path(data_path).expanduser()
	save_root = Path(save_path).expanduser()
	model_root = Path(model_path).expanduser()
	ext = "pickle"

	file_sfx = ""
	file_sfx += "_icp" if do_icp else ""
	file_sfx += "_ransac" if do_ransac else ""

	for filename, cfg in cfg_dict.items():
		for model in cfg:
			model_path = model_root / model.name / checkpoint

			args_dict = dict(
				data=data_path,
				dataset=dataset,
				weights_path=model_path,
				num_iters=num_iters,
				ransac=do_ransac,
				icp=do_icp,
				save=False
			)
			args = Namespace(**args_dict)

			print(f"\nEvaluating Model {model_path}.")
			eval_stats = eval_registration(gpu, model_path, args)

			savefile_path = save_root / f"{filename}{file_sfx}_{dataset}.{ext}"

			# Save after every evaluation, in case something breaks at the middle,
			# to at least not loose all previous evaluations
			save_stats(savefile_path, eval_stats, model.label)


def parse_path(value):
	return Path(value).expanduser()


def parse_keys(value):
	if value is None:
		return value

	return value.split(",")


def parse_args():
	parser = ArgumentParser()

	parser.add_argument("--data_path", "-p", type=parse_path, default="/data/arceyd/kitti/")
	parser.add_argument("--save_path", "-s", type=parse_path, default="~/MT/res/ablations/")
	parser.add_argument("--model_path", "-m", type=parse_path, default="~/MT/cp/3D/")

	parser.add_argument("--checkpoint", "-c", default="best_model_so_far_rot.tar")
	parser.add_argument("--keys", "-k", type=parse_keys, default=None)

	parser.add_argument("--num_iters", "-n", type=int, default=1)

	parser.add_argument("--dataset", "-d", default="kitti")

	parser.add_argument("--do_ransac", "-r", action="store_true")
	parser.add_argument("--do_icp", "-i", action="store_true")

	args = parser.parse_args()

	return vars(args)


if __name__ == "__main__":
	main(**parse_args())

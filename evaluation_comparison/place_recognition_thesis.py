from collections import namedtuple
from pathlib import Path

from evaluation_comparison.inference_placerecognition_general import main_process as place_rec

Model = namedtuple("model", ["label", "checkpoint_dir", "checkpoint_name"])


def main():

	gpu = 0
	# dataset = "kitti"
	dataset = "kitti360"
	num_iters = 1

	# dataset_path = Path("/data/arceyd/kitti/")
	# dataset_path = Path("/home/arceyd/MT/dat/kitti360/")
	home_path = Path("~/MT/").expanduser()
	cp_path = home_path / "cp" / "3D"
	save_path = home_path / "res" / "place_recognition"
	save_ext = ".npz"

	models = [
		Model("lcdnet", "16-09-2021_00-02-34", "checkpoint_133_auc_0.859.tar"),
		Model("dcp",    "04-04-2022_18-34-14", "best_model_so_far_auc.tar"),
		Model("tf",     "12-05-2022_10-38-29", "best_model_so_far_auc.tar"),
		Model("padloc", "27-05-2022_19-10-54", "best_model_so_far_auc.tar"),
	]

	datasets = {
		"kitti": dict(path=Path("/data/arceyd/kitti/"), sequences=["00", "08"]),
		"kitti360": dict(path=Path("/home/arceyd/MT/dat/kitti360/"), sequences=["2013_05_28_drive_0002_sync"])
	}

	for model in models:
		for sequence in datasets[dataset]["sequences"]:
			checkpoint = cp_path / model.checkpoint_dir / model.checkpoint_name
			res_path = save_path / f"{model.label}_{dataset}_seq{sequence}{save_ext}"
			place_rec(gpu, checkpoint, dataset, datasets[dataset]["path"], num_iters, res_path, sequence=sequence)


if __name__ == "__main__":
	main()

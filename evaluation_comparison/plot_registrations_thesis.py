import json
from pathlib import Path

from evaluation_comparison.plot_styles import Style
from evaluation_comparison.plot_registration import main as plot_registration


def main():

	gpu = 0

	# seq = 8
	# frames = [[2520, 3866]]

	home_path = Path("~").expanduser() / "MT"
	kitti_path = home_path / "dat" / "kitti" / "dataset"
	cp_path = home_path / "cp" / "3D"
	cp_file = "best_model_so_far_rot.tar"
	img_path = home_path / "res" / "registrations"
	img_ext = ".png"

	def get_cp(run):
		return cp_path / run / cp_file

	def get_img(img_name):
		return img_path / (img_name + img_ext)

	def get_cp_img(*, run, img_name, seq, frames, plot_individual=False, **_):
		seq = f"{seq:02d}"
		return dict(cp_path=get_cp(run), seq=seq, frames=frames,
					image_paths=[get_img(f"fig_{img_name}_{seq}_{frame[0]}_{frame[1]}") for frame in frames],
					plot_individual=plot_individual
					)

	styles = [
		Style("bw"),
		Style("color")
	]

	common_cfg = dict(kitti_path=kitti_path, gpu=gpu)

	# augment_ry180 = dict(ry=180, angle_units="deg")

	configs = [
		#          SEQ  , Frames     , Model Checkpoint         , Image Name
		#                     90º Loop      180º Loop     180º, 10m    180º, 20m     180º, 30m     180º, 40m
		dict(seq=8,
			 frames=[
				 # [2520, 3866],
				 # [ 750, 1460],
				 # [ 750, 1453],
				 [ 750, 1448],
				 # [ 750, 1446],
				 # [ 750, 1436],
				 # [ 750, 1425],
				 # [ 750, 1404]
			 ],
			 run="16-09-2021_00-02-34", img_name="lcdnet", plot_individual=True),
		dict(seq=8,
			 frames=[
				 # [2520, 3866],
				 # [ 750, 1460],
				 # [ 750, 1453],
				 [ 750, 1448],
				 # [ 750, 1446],
				 # [ 750, 1436],
				 # [ 750, 1425],
				 # [ 750, 1404]
			 ],
		 	 run="04-04-2022_18-34-14", img_name="dcp"),
		dict(seq=8,
			 frames=[
				# [2520, 3866],
				# [ 750, 1460],
				#  [ 750, 1453],
				 [ 750, 1448],
				# [ 750, 1446],
				# [ 750, 1436],
				# [ 750, 1425],
				# [ 750, 1404]
			 ],
			 run="12-05-2022_10-38-29", img_name="tf"),
		dict(seq=8,
			 frames=[
				 # [2520, 3866],
				 # [ 750, 1460],
				 # [ 750, 1453],
				 [ 750, 1448],
				 # [ 750, 1446],
				 # [ 750, 1436],
				 # [ 750, 1425],
				 # [ 750, 1404]
			 ],
			 run="13-05-2022_18-02-51", img_name="tf_rev_sem"),
		dict(seq=8,
			 frames=[
				 # [2520, 3866],
				 # [ 750, 1460],
				 # [ 750, 1453],
				 [ 750, 1448],
				 # [ 750, 1446],
				 # [ 750, 1436],
				 # [ 750, 1425],
				 # [ 750, 1404]
			 ],
			 run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem"),
		dict(seq=8,
			 frames=[
				 # [2520, 3866],
				 # [ 750, 1460],
				 # [ 750, 1453],
				 [ 750, 1448],
				 # [ 750, 1446],
				 # [ 750, 1436],
				 # [ 750, 1425],
				 # [ 750, 1404]
			 ],
			 run="27-05-2022_19-10-54", img_name="padloc"),
		# 0º Loop
		# dict(seq=9, frames=[[   0, 1575]], run="16-09-2021_00-02-34", img_name="lcdnet", plot_individual=True),
		# dict(seq=9, frames=[[   0, 1575]], run="04-04-2022_18-34-14", img_name="dcp"),
		# dict(seq=9, frames=[[   0, 1575]], run="12-05-2022_10-38-29", img_name="tf"),
		# dict(seq=9, frames=[[   0, 1575]], run="13-05-2022_18-02-51", img_name="tf_rev_sem"),
		# dict(seq=9, frames=[[   0, 1575]], run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem"),
		# dict(seq=9, frames=[[   0, 1575]], run="27-05-2022_19-10-54", img_name="padloc"),
	]

	all_evals = {}

	for cfg in configs:
		cfg_dict = get_cp_img(**cfg)
		evals = plot_registration(styles=styles, **common_cfg, **cfg_dict)
		all_evals.update(evals)

	with open(img_path / "errors.json", "w") as f:
		json.dump(all_evals, f)


if __name__ == "__main__":
	main()

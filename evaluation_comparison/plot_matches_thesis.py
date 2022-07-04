from pathlib import Path

from evaluation_comparison.plot_styles import Style
from evaluation_comparison.plot_matches import main as plot_matches


def main():

	gpu = 3

	# seq = 8
	# frames = [[2520, 3866]]

	home_path = Path("~").expanduser() / "MT"
	kitti_path = home_path / "dat" / "kitti" / "dataset"
	cp_path = home_path / "cp" / "3D"
	cp_file = "best_model_so_far_rot.tar"
	img_path = home_path / "res" / "matches"
	img_ext = ".png"

	def get_cp(run):
		return cp_path / run / cp_file

	def get_img(img_name):
		return img_path / (img_name + img_ext)

	def get_cp_img(*, run, img_name, seq, frames, augment_positive=None, **_):
		seq = f"{seq:02d}"
		return dict(cp_path=get_cp(run), seq=seq, frames=frames,
					image_paths=[get_img(f"fig_{img_name}_{seq}_{frames[0][0]}_{frames[0][1]}")],
					augment_positive=augment_positive)

	styles = [
		Style("bw"),
		Style("color")
	]

	common_cfg = dict(kitti_path=kitti_path, gpu=gpu)

	augment_ry180 = dict(ry=180, angle_units="deg")


	configs = [
		#          SEQ  , Frames     , Model Checkpoint         , Image Name
		# 90º Loop
		# dict(seq=8, frames=[[2520, 3866]], run="16-09-2021_00-02-34", img_name="lcdnet"),
		# dict(seq=8, frames=[[2520, 3866]], run="04-04-2022_18-34-14", img_name="dcp"),
		# dict(seq=8, frames=[[2520, 3866]], run="12-05-2022_10-38-29", img_name="tf"),
		# dict(seq=8, frames=[[2520, 3866]], run="13-05-2022_18-02-51", img_name="tf_rev_sem"),
		# dict(seq=8, frames=[[2520, 3866]], run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem"),
		dict(seq=8, frames=[[2520, 3866]], run="27-05-2022_19-10-54", img_name="padloc"),
		# 0º Loop
		# dict(seq=9, frames=[[   0, 1575]], run="16-09-2021_00-02-34", img_name="lcdnet"),
		# dict(seq=9, frames=[[   0, 1575]], run="04-04-2022_18-34-14", img_name="dcp"),
		# dict(seq=9, frames=[[   0, 1575]], run="12-05-2022_10-38-29", img_name="tf"),
		# dict(seq=9, frames=[[   0, 1575]], run="13-05-2022_18-02-51", img_name="tf_rev_sem"),
		# dict(seq=9, frames=[[   0, 1575]], run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem"),
		dict(seq=9, frames=[[   0, 1575]], run="27-05-2022_19-10-54", img_name="padloc"),
		# 180º Loop
		# dict(seq=8, frames=[[ 750, 1460]], run="16-09-2021_00-02-34", img_name="lcdnet"),
		# dict(seq=8, frames=[[ 750, 1460]], run="04-04-2022_18-34-14", img_name="dcp"),
		# dict(seq=8, frames=[[ 750, 1460]], run="12-05-2022_10-38-29", img_name="tf"),
		# dict(seq=8, frames=[[ 750, 1460]], run="13-05-2022_18-02-51", img_name="tf_rev_sem"),
		# dict(seq=8, frames=[[ 750, 1460]], run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem"),
		dict(seq=8, frames=[[ 750, 1460]], run="27-05-2022_19-10-54", img_name="padloc"),
		# Augmented 180º Loop
		# dict(seq=5, frames=[[ 700,  700]], run="16-09-2021_00-02-34", img_name="lcdnet_augm_ry180",
		# 	 augment_positive=augment_ry180),
		# dict(seq=5, frames=[[ 700,  700]], run="04-04-2022_18-34-14", img_name="dcp_augm_ry180",
		# 	 augment_positive=augment_ry180),
		# dict(seq=5, frames=[[ 700,  700]], run="12-05-2022_10-38-29", img_name="tf_augm_ry180",
		# 	 augment_positive=augment_ry180),
		# dict(seq=5, frames=[[ 700,  700]], run="13-05-2022_18-02-51", img_name="tf_rev_sem_augm_ry180",
		# 	 augment_positive=augment_ry180),
		# dict(seq=5, frames=[[ 700,  700]], run="10-05-2022_13-24-39", img_name="tf_rev_pan_mse_sem_augm_ry180",
		# 	 augment_positive=augment_ry180),
		dict(seq=5, frames=[[ 700,  700]], run="27-05-2022_19-10-54", img_name="padloc",
			 augment_positive=augment_ry180),
	]
	for style in styles:
		for cfg in configs:
			old_img_name = cfg.pop("img_name")
			img_name = style.pfx + "_" + old_img_name
			cfg_dict = get_cp_img(img_name=img_name, **cfg)
			plot_matches(style=style, **common_cfg, **cfg_dict)
			cfg["img_name"] = old_img_name


if __name__ == "__main__":
	main()

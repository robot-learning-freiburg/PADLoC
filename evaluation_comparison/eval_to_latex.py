import pickle


def eval2latex(file):
	with open(file, "rb") as f:
		stats = pickle.load(f)

	print("Run & AP & Yaw (Mean + Std) & Rot (Mean + Std) & Tra (Mean + Std)")
	for key, stat_dict in stats.items():

		if "tra_mean" not in stat_dict or "tra_std" not in stat_dict:
			tra_key = "transl" if "transl" in stat_dict else "tra"
			transl = stat_dict[tra_key]
			tra_mean = transl.mean()
			tra_std = transl.std()
		else:
			tra_mean = stat_dict["tra_mean"]
			tra_std = stat_dict["tra_std"]

		ap = f"\\({stat_dict['success_rate']*100:.2f}\\)"
		yaw = f"\\({stat_dict['yaw_mean']:.3f} \\pm {stat_dict['yaw_std']:.3f}\\)"
		rot = f"\\({stat_dict['rot_mean']:.3f} \\pm {stat_dict['rot_std']:.3f}\\)"
		tra = f"\\({tra_mean:.3f} \\pm {tra_std:.3f}\\)"

		row = [
			key,
			ap,
			# yaw, # Yaw no longer needed since we show the full rotational error
			rot,
			tra
		]

		line = " & ".join(row) + r" \\"

		print(line)


if __name__ == "__main__":
	eval2latex("/Users/Jose/Documents/Homeworks/06 M.Sc. Informatik/Master Thesis in RL Chair/res/ablations/ablation_weight_kitti.pickle")
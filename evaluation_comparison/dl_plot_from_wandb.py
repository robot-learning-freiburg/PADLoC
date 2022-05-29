from collections import namedtuple
from pathlib import Path

from evaluation_comparison.rot_tra_from_wandb import main as dl_from_wandb
from evaluation_comparison.plot_exp_log import plot_with_inset
from evaluation_comparison.plot_styles import Style

Run = namedtuple("run", ["name", "label"])
Curve = namedtuple("curve", ["title", "label", "units", "smooth_f", "x_lim", "do_inset",
							 "inset_x_lim", "inset_y_lim", "plot_runs",
							 "legend", "legend_cols"])


def main():
	project = "joseab10/deep_lcd"
	runs_labels = {
		# "ConfWAbl": [
		# 	("", "ColSum"),
		# 	("", "Shannon"),
		# 	("", "Hill"),
		# 	("", "Berger"),
		# ],
		"LossesAblLCDNet": {
			"runs": [
				Run("16/09/2021 00:02:34", "LCDNet"),
				Run("18/05/2022 14:57:01", "LCDNet+Rev"),
				Run("20/05/2022 02:20:25", "LCDNet+Rev+Sem"),
				Run("23/05/2022 05:26:27", "LCDNet+Rev+Mse"),
				Run("21/05/2022 15:55:37", "LCDNet+Rev+Pan"),
				Run("17/05/2022 00:20:38", "LCDNet+Rev+Pan+Sem+Mse"),
			],
			"curves": [
				Curve("Rotation Mean Error", "rme", "[deg]", 0.8, (1, 150), True, (80, 150), (1.15, 3.75),
					  ["LCDNet", "LCDNet+Rev", "LCDNet+Rev+Sem", "LCDNet+Rev+Pan+Sem+Mse"],
					  True, 4),
				Curve("Translation Error", "tme", "[m]", 0.8, (1, 150), True,  (80, 150), (0.9, 1.55),
					  ["LCDNet", "LCDNet+Rev", "LCDNet+Rev+Sem", "LCDNet+Rev+Pan+Sem+Mse"],
					  False, 4),
			]
		},
		"LossesAblTF": {
			"runs": [
				Run("12/05/2022 10:38:29", "TF"),
				Run("15/05/2022 10:59:41", "TF+Rev"),
				Run("13/05/2022 18:02:51", "TF+Rev+Sem"),
				Run("08/05/2022 23:35:45", "TF+Rev+Mse"),
				Run("01/05/2022 11:33:12", "TF+Rev+Pan"),
				Run("10/05/2022 13:24:39", "TF+Rev+Pan+Sem+Mse")
			],
			"curves": [
				Curve("Rotation Mean Error", "rme", "[deg]", 0.8, (1, 150), True,  (80, 150), (2.5, 3.75),
					  ["TF", "TF+Rev", "TF+Rev+Sem", "TF+Rev+Pan+Sem+Mse"],
					  True, 4),
				Curve("Translation Error", "tme", "[m]", 0.8, (1, 150), True,  (80, 150), (1.15, 1.7),
					  ["TF", "TF+Rev", "TF+Rev+Sem", "TF+Rev+Pan+Sem+Mse"],
					  False, 4),
			]
		},
		"WeightsAbl": {
			"runs": [
				Run("24/05/2022 23:08:23", "Uniform"),
				Run("25/05/2022 10:51:25", "Column Sum"),
				Run("25/05/2022 23:04:41", "Shannon"),
				Run("26/05/2022 09:55:19", "Hill, order=2"),
				Run("26/05/2022 20:52:10", "Hill, order=4"),
				Run("27/05/2022 07:46:12", "Berger-Parker")
			],
			"curves": [
				Curve("Rotation Mean Error", "rme", "[deg]", 0.5, (1, 50), False, None, None,
					  ["Uniform", "Column Sum", "Shannon", "Hill, order=2", "Hill, order=4", "Berger-Parker"],
					  True, 3),
				Curve("Translation Error", "tme", "[m]", 0.5, (1, 50), False, None, None,
					  ["Uniform", "Column Sum", "Shannon", "Hill, order=2", "Hill, order=4", "Berger-Parker"],
					  False, 3),

			]
		}
	}

	home_path = Path("~/Documents/Homeworks/06 M.Sc. Informatik/Master Thesis in RL Chair/Master Thesis/shared/")
	home_path = home_path.expanduser()
	data_path = home_path / "dat" / "res"
	img_path = home_path / "img" / "res"

	data_file_extension = "csv"
	plot_file_extension = "pdf"

	styles = [
		Style("bw"),
		Style("color")
	]

	for file_prefix, cfg in runs_labels.items():
		runs = cfg["runs"]
		curves_cfg = cfg["curves"]

		run_names = [rl.name for rl in runs]
		run_labels = [rl.label for rl in runs]

		for curve_cfg in curves_cfg:
			# Download data from WandB
			df = dl_from_wandb(project=project, path=data_path, runs=run_names, labels=run_labels,
							   curves=[curve_cfg.title], curve_labels=[curve_cfg.label],
							   file_prefix=file_prefix, file_extension=data_file_extension)
			df = df[curve_cfg.label]

			if curve_cfg.plot_runs:
				df = df[curve_cfg.plot_runs]

			if curve_cfg.smooth_f:
				df = df.ewm(alpha=1 - curve_cfg.smooth_f).mean()

			# Plot data from WandB
			for style in styles:
				plot_path = img_path / f"fig_{style.pfx}_{file_prefix}_{curve_cfg.label}.{plot_file_extension}"

				plot_with_inset(dataframe=df, xlim=curve_cfg.x_lim,
								xlabel="Epochs", ylabel=f"Mean Error {curve_cfg.units}",
								plot_title="", inset_pos=None, inset_mark_loc=(3, 1),
								do_inset=curve_cfg.do_inset,
								inset_xlim=curve_cfg.inset_x_lim, inset_ylim=curve_cfg.inset_y_lim,
								style=style, save_path=plot_path, show=False,
								plot_legend=curve_cfg.legend, legend_cols=curve_cfg.legend_cols)


if __name__ == "__main__":
	main()


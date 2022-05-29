# import matplotlib as mpl
# # Use the pgf backend (must be set before pyplot imported)
# mpl.use('pgf')

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
import numpy as np


from evaluation_comparison.plot_styles import Style


def plot_linlog(ax, dataframe, style, xlim, xlabel, ylabel, plot_title):
	ax.set_prop_cycle(style.cycler_lines)
	dataframe.plot(ax=ax, legend=None, **style.curves)

	# Main Plot Formatting
	ax.set_yscale("log")
	ax.yaxis.set_major_formatter(ScalarFormatter())
	ax.yaxis.set_minor_locator(LogLocator(subs=(2., 4., 6., 8.,)))
	ax.yaxis.set_minor_formatter(ScalarFormatter())
	ax.grid(visible=True, which="both", **style.grid)
	ax.set(xlim=xlim)
	ax.set(xlabel=xlabel, ylabel=ylabel)
	ax.set_title(plot_title)


def plot_inset(ax, dataframe, style, xlim, ylim):
	ax.set_prop_cycle(style.cycler_lines)
	ax.set(xlim=xlim, ylim=ylim)

	# Inset Plot
	dataframe.plot(ax=ax, legend=None, **style.curves)

	ax.grid(visible=True, which="both", **style.grid)
	ax.set(xlabel="", ylabel="")

	# Set tick labels to have a bounding box with a style (to mask the background)
	plt.setp(ax.get_xticklabels(), bbox=style.inset_label_bbox)
	plt.setp(ax.get_yticklabels(), bbox=style.inset_label_bbox)


def plot_legends(curve_list, filename, style, expand=None, ncol=4):

	tmp_fig, tmp_ax = plt.subplots()
	tmp_ax.set_prop_cycle(style.cycler_lines)

	handles = [tmp_ax.plot([], [], label=curve)[0] for curve in curve_list]

	legend = plt.legend(handles, curve_list, loc=3, framealpha=1, frameon=True, ncol=ncol)

	if expand is None:
		expand = [-5, -5, 5, 5]

	fig = legend.figure
	fig.canvas.draw()
	bbox = legend.get_window_extent()
	bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
	bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

	print(f"Saving plot to file {filename}.")
	fig.savefig(filename, dpi="figure", bbox_inches=bbox)
	plt.close(fig)


def plot_with_inset(*, dataframe,
					xlim=(1, 150),
					xlabel="Epochs",
					ylabel="Mean Error",
					plot_title="Mean Error",
					do_inset=True,
					inset_pos=None,
					inset_mark_loc=(3, 1),
					inset_xlim=(80, 150),
					inset_ylim=(2.3, 3.6),
					style=None,
					save_path=None,
					show=True,
					plot_legend=False,
					legend_cols=4,
					**_
					):

	if style is None:
		style = Style("bw")

	if inset_pos is None:
		inset_pos = [0.35, 0.45, 0.6, 0.5]

	fig, ax1 = plt.subplots(dpi=style.dpi)

	plot_linlog(ax1, dataframe, style, xlim, xlabel, ylabel, plot_title)

	# Zoomed-In Inset
	# Create a set of inset Axes: these should fill the bounding box allocated to
	# them.
	if do_inset:
		ax2 = plt.axes([0, 0, 1, 1])
		# Manually set the position and relative size of the inset axes within ax1
		ip = InsetPosition(ax1, inset_pos)  # Left, Bottom, Width, Height
		ax2.set_axes_locator(ip)
		# Mark the region corresponding to the inset axes on ax1 and draw lines
		# in grey linking the two axes.
		mark_inset(ax1, ax2, loc1=inset_mark_loc[0], loc2=inset_mark_loc[1], **style.inset)

		plot_inset(ax2, dataframe.iloc[inset_xlim[0] - 1:inset_xlim[1]], style, inset_xlim, inset_ylim)

	# ax2.set_prop_cycle(style.cycler_lines)
	# ax2.set(xlim=(inset_xlim[0], inset_xlim[1]), ylim=inset_ylim)
	#
	# # Inset Plot
	# df2.iloc[inset_xlim[0] - 1:inset_xlim[1]].plot(ax=ax2, marker=",", linewidth=0.8, legend=None)
	#
	# ax2.grid(visible=True, which="both", **style.grid)
	# ax2.set(xlabel="", ylabel="")
	#
	# # Set tick labels to have a bounding box with a style (to mask the background)
	# plt.setp(ax2.get_xticklabels(), bbox=style.inset_label_bbox)
	# plt.setp(ax2.get_yticklabels(), bbox=style.inset_label_bbox)

	# ax1.set_prop_cycle(style.cycler_lines)
	# df2.plot(ax=ax1, legend=None, **style.curves)
	#
	# # Main Plot Formatting
	# ax1.set_yscale("log")
	# ax1.yaxis.set_major_formatter(ScalarFormatter())
	# ax1.yaxis.set_minor_locator(LogLocator(subs=(2., 4., 6., 8.,)))
	# ax1.yaxis.set_minor_formatter(ScalarFormatter())
	# ax1.grid(visible=True, which="both", **style.grid)
	# ax1.set(xlim=xlim)
	# ax1.set(xlabel=xlabel, ylabel=ylabel)
	# ax1.set_title(plot_title)

	if save_path is not None:
		print(f"Saving plot to file {save_path}.")
		plt.savefig(save_path)

	if show:
		plt.show()

	plt.close(fig)

	if plot_legend and save_path is not None:
		legend_dir = save_path.parent
		legend_name = save_path.stem + "_legend"
		legend_sfxs = "".join(save_path.suffixes)
		legend_path = legend_dir / f"{legend_name}{legend_sfxs}"
		curve_list = dataframe.columns.tolist()
		if "Step" in curve_list:
			curve_list.delete("Step")

		plot_legends(curve_list=curve_list, filename=legend_path, style=style, ncol=legend_cols)


if __name__ == "__main__":

	home_dir = Path("~/Documents/Homeworks/06 M.Sc. Informatik/Master Thesis in RL Chair/Master Thesis/shared")
	home_dir = home_dir.expanduser()
	csv_dir = home_dir / "dat" / "res"
	save_dir = home_dir / "img" / "res"
	wandb_smoothing_factor = 0.8
	show_plot = False
	plot_style = Style("bw")

	mre_tf_cfg = dict(
		file=csv_dir / "ablationlossrme.csv",
		smoothing_factor=wandb_smoothing_factor,
		xlim=(1, 150),
		columns=["TF", "R", "RS", "RPMS"],
		xlabel="Epochs", ylabel="Mean Error [deg]", plot_title="Mean Rotation Error (Transformer)",
		inset_pos=[0.35, 0.45, 0.6, 0.5], inset_mark_loc=(3, 1), inset_xlim=(80, 150), inset_ylim=(2.3, 3.6),
		style=plot_style,
		save_path=save_dir / "fig_abl_loss_tf_rme.pdf", show=show_plot,
	)

	mte_tf_cfg = dict(
		file=csv_dir / "ablationlosstme.csv",
		smoothing_factor=wandb_smoothing_factor,
		xlim=(1, 150),
		columns=["TF", "R", "RS", "RPMS"],
		xlabel="Epochs", ylabel="Mean Error [m]", plot_title="Mean Translation Error (Transformer)",
		inset_pos=[0.35, 0.45, 0.6, 0.5], inset_mark_loc=(3, 1), inset_xlim=(80, 150), inset_ylim=(1.1, 1.45),
		style=plot_style,
		save_path=save_dir / "fig_abl_loss_tf_tme.pdf", show=show_plot,
	)

	plots_cfg = [
		mre_tf_cfg, mte_tf_cfg,
	]

	for plot_cfg in plots_cfg:
		df = pd.read_csv(plot_cfg.pop("file"), sep=",", index_col=0)
		df = df.ewm(alpha=1 - plot_cfg.pop("smoothing_factor")).mean()
		columns = plot_cfg.pop("columns")
		# Select only some of the columns
		if columns is not None:
			df = df[columns]

		plot_with_inset(dataframe=df, **plot_cfg)

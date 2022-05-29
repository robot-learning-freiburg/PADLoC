from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation_comparison.plot_styles import Style
from evaluation_comparison.plot_matches import (transform_vertices, iso_transform, get_samples,
												load_model, load_dataset, parse_args)
from evaluation_comparison.inference_yaw_general import rot2aa


def buffer_single_registration(points, scatter_style, xlim, ylim, fig_size=(5.8, 4.1), dpi=400):
	fig = plt.figure(dpi=dpi, figsize=fig_size, frameon=False, facecolor="none", edgecolor="none")
	ax = fig.add_subplot()
	ax.axis('off')
	ax.set_aspect('equal')
	ax.patch.set_facecolor("none")
	ax.patch.set_edgecolor("none")
	ax.scatter(points[:, 0], points[:, 1], **scatter_style)
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)
	ax.patch.set_facecolor("none")
	ax.patch.set_edgecolor("none")
	fig.canvas.draw()

	w, h = fig.canvas.get_width_height()
	img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy()

	ax.clear()
	fig.clf()
	plt.close(fig)
	return img


def plot_image(img, fig_size, style, image_path=None):
	fig = plt.figure(figsize=fig_size, dpi=style.point_cloud_dpi)
	ax = fig.add_subplot()
	ax.axis('off')
	ax.set_aspect('equal')
	ax.imshow(img)

	fig.tight_layout(pad=0.05)
	if image_path is None:
		fig.show()
		return

	print(f"Saving figure {image_path}.")
	fig.savefig(image_path)
	plt.close()


def plot_point_clouds(point_clouds, styles, dpi, fig_size, x_lim, y_lim, image_path=None):

	fig = plt.figure(dpi=dpi, figsize=fig_size, frameon=False)
	ax = fig.add_subplot()
	ax.axis('off')
	ax.set_aspect('equal')

	for pc, sty in zip(point_clouds, styles):
		ax.scatter(pc[:, 0], pc[:, 1], **sty)

	ax.set_xlim(*x_lim)
	ax.set_ylim(*y_lim)
	fig.tight_layout(pad=0.05)

	if image_path is None:
		fig.show()
		return

	print(f"Saving figure {image_path}.")
	fig.savefig(image_path)
	plt.close(fig)


def plot_registration(*, anc_pc, pos_pc, p2a_transform, device, style,
					  image_path: Optional[Path] = None, plot_individual=False):

	iso_tf = iso_transform(device)
	pos_tf = iso_tf @ p2a_transform

	proj_anc_pc = transform_vertices(vertices=anc_pc[:, :3], transform_matrix=iso_tf, device=device).cpu().numpy()
	proj_pos_pc = transform_vertices(vertices=pos_pc[:, :3], transform_matrix=pos_tf, device=device).cpu().numpy()

	# Compute the x and y limits so that both images have the same scaling
	anc_min = np.amin(proj_anc_pc, axis=0)
	anc_max = np.amax(proj_anc_pc, axis=0)
	pos_min = np.amin(proj_pos_pc, axis=0)
	pos_max = np.amax(proj_pos_pc, axis=0)

	xlim = (min(anc_min[0], pos_min[0]), max(anc_max[0], pos_max[0]))
	ylim = (min(anc_min[1], pos_min[1]), max(anc_max[1], pos_max[1]))

	fig_size = (5.8, 4.1)

	# Get the RGB-A buffers for each point cloud scatter plot
	anc_buffer = buffer_single_registration(proj_anc_pc, style.src_reg_point_cloud, xlim, ylim,
											fig_size=fig_size, dpi=style.point_cloud_dpi)
	pos_buffer = buffer_single_registration(proj_pos_pc, style.tgt_reg_point_cloud, xlim, ylim,
											fig_size=fig_size, dpi=style.point_cloud_dpi)

	if plot_individual:
		img_dir = image_path.parent
		img_name = image_path.stem
		img_sfxs = image_path.suffixes

		unreg_pos_pc = transform_vertices(vertices=pos_pc[:, :3], transform_matrix=iso_tf, device=device).cpu().numpy()
		upos_buffer = buffer_single_registration(unreg_pos_pc, style.tgt_reg_point_cloud, xlim, ylim,
												 fig_size=fig_size, dpi=style.point_cloud_dpi)
		plot_image(anc_buffer, fig_size, style, img_dir / (img_name + "_anc" + "".join(img_sfxs)))
		plot_image(upos_buffer, fig_size, style, img_dir / (img_name + "_pos" + "".join(img_sfxs)))

	merged_buffer = style.blend_registration_buffers(anc_buffer, pos_buffer)

	plot_image(merged_buffer, fig_size, style, image_path)


def eval_registration(gt_pose_anc, gt_pose_pos, pred_transform):

	homogeneous = torch.tensor([[0., 0., 0., 1.]]).to(pred_transform.device)
	pred_transform = torch.cat((pred_transform, homogeneous), dim=0)

	delta_pose = torch.linalg.inv(gt_pose_anc) @ gt_pose_pos
	rel_pose_err = delta_pose @ pred_transform
	rel_pose_tra_err = rel_pose_err[:3, 3]  # Just for comparison with transl_error
	rel_pose_tra_err = torch.norm(rel_pose_tra_err).cpu().detach().numpy().item()
	rel_pose_rot_err_mat = rel_pose_err[:3, :3]
	rel_pose_rot_err_ax, rel_pos_rot_err_ang = rot2aa(rel_pose_rot_err_mat.cpu().detach().numpy())
	rel_pose_rot_err_ang = np.abs(rel_pos_rot_err_ang) % (2 * np.pi)
	rel_pose_rot_err_ang *= 180 / np.pi

	return rel_pose_tra_err, rel_pose_rot_err_ang


def infer_and_plot(*, dataset, device, model, frames, styles, image_paths, plot_individual=False, **_):
	samples = get_samples(dataset, frames, model, device)

	with torch.no_grad():
		model(samples, metric_head=True, compute_embeddings=True,
			  compute_transl=False, compute_rotation=True, mode="pairs")

	batch_size = len(samples["anchor_idx"])
	if len(image_paths) != batch_size:
		print(f"Warning: Different number of image paths {len(image_paths)} than batches {batch_size}. Not saving.")
		image_paths = [None] * batch_size

	evals = {}

	for i, image_path in enumerate(image_paths):
		for style in styles:
			img_dir = image_path.parent
			img_file = image_path.stem.split("_")
			img_file.insert(1, style.pfx)
			img_file = "_".join(img_file)
			img_ext = "".join(image_path.suffixes)
			img_style_path = img_dir / f"{img_file}{img_ext}"
			plot_registration(anc_pc=samples["anchor"][i], pos_pc=samples["positive"][i],
							  p2a_transform=samples["p2a_transform"][i],
							  device=device, style=style,
							  image_path=img_style_path, plot_individual=plot_individual)

		tra_err, rot_err = eval_registration(samples["anchor_transform"][i], samples["positive_transform"][i], samples["transformation"][i])
		evals[image_path.stem] = {"tra_err_m": tra_err, "rot_err_deg": rot_err}

	return evals


def main(kitti_path, seq, cp_path, frames, image_paths, gpu, styles="bw", plot_individual=False):
	kitti_dataset = load_dataset(kitti_path=kitti_path, seq=seq)

	torch.cuda.set_device(gpu)
	cuda_device = torch.device(gpu)

	model = load_model(cp_file=cp_path, device=cuda_device)

	if isinstance(styles, str):
		styles = [Style(styles)]

	return infer_and_plot(dataset=kitti_dataset, device=cuda_device, frames=frames, model=model,
						   styles=styles, image_paths=image_paths, plot_individual=plot_individual)


if __name__ == "__main__":
	cli_args = parse_args()
	main(plot_individual=True, **cli_args)

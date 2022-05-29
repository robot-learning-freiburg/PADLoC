from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import torch.nn
import torch.nn.functional as f

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from datasets.KITTI_data_loader import KITTILoader3DDictSingle
from models.get_models import get_model
from utils.geometry import get_rt_matrix, mat2xyzrpy
from evaluation_comparison.plot_styles import Style


def create_orthogonal_projection(
		left,
		right,
		bottom,
		top,
		near,
		far,
):
	"""Creates an orthogonal projection matrix.

	:param float left: The left of the near plane relative to the plane's centre.
	:param float right: The right of the near plane relative to the plane's centre.
	:param float top: The top of the near plane relative to the plane's centre.
	:param float bottom: The bottom of the near plane relative to the plane's centre.
	:param float near: The distance of the near plane from the camera's origin.
		It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
		at close range.
	:param float far: The distance of the far plane from the camera's origin.
	:rtype: numpy.array
	:return: A projection matrix representing the specified orthogonal perspective.
	"""

	"""
	A 0 0 Tx
	0 B 0 Ty
	0 0 C Tz
	0 0 0 1

	A = 2 / (right - left)
	B = 2 / (top - bottom)
	C = -2 / (far - near)

	Tx = (right + left) / (right - left)
	Ty = (top + bottom) / (top - bottom)
	Tz = (far + near) / (far - near)
	"""
	rml = right - left
	tmb = top - bottom
	fmn = far - near

	a = 2. / rml
	b = 2. / tmb
	c = -2. / fmn
	tx = -(right + left) / rml
	ty = -(top + bottom) / tmb
	tz = -(far + near) / fmn

	return torch.Tensor((
		( a, 0., 0., 0.),
		(0.,  b, 0., 0.),
		(0., 0.,  c, 0.),
		(tx, ty, tz, 1.),
	))


def create_look_at(eye, target, up):
	"""Creates a look at matrix according to OpenGL standards.

	:param numpy.array eye: Position of the camera in world coordinates.
	:param numpy.array target: The position in world coordinates that the
		camera is looking at.
	:param numpy.array up: The up vector of the camera.
	:rtype: numpy.array
	:return: A look at matrix that can be used as a viewMatrix
	"""

	forward = f.normalize(target - eye, dim=0)
	side = f.normalize(torch.cross(forward, up), dim=0)
	up = f.normalize(torch.cross(side, forward), dim=0)

	p1 = - torch.dot(side, eye)
	p2 = - torch.dot(up, eye)
	p3 = torch.dot(forward, eye)

	return torch.Tensor((
		(side[0], up[0], -forward[0], 0.),
		(side[1], up[1], -forward[1], 0.),
		(side[2], up[2], -forward[2], 0.),
		(     p1,    p2,          p3, 1.)
	))


def load_model(cp_file, device):
	saved_params = torch.load(cp_file, map_location='cpu')
	config = saved_params["config"]

	model = get_model(config)

	model.load_state_dict(saved_params['state_dict'], strict=True)

	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model.eval()
	model.to(device)

	return model


def load_dataset(kitti_path, seq,
				 loop_file="loop_GT_4m", **_):

	poses = kitti_path / "sequences" / seq / "poses.txt"
	dataset = KITTILoader3DDictSingle(kitti_path, sequence=seq,
									  npoints=None, device=None,  # Mandatory arguments that are not used!
									  poses=poses, loop_file=loop_file,
									  use_semantid=True, use_panoptic=True, use_logits=False, filter_dynamic=False)

	return dataset


def move_to_sample(frame, sample_pfx, frame_pfx, device, sample=None, suffixes=None):
	if sample is None:
		sample = {}

	if suffixes is None:
		suffixes = ["idx", "", "pose", "rot", "semantic", "supersem", "panoptic", "instance"]

	for suffix in suffixes:
		frame_k = frame_pfx + ("_" if frame_pfx and suffix else "") + suffix
		sample_k = sample_pfx + ("_" if sample_pfx and suffix else "") + suffix
		frame_val = frame[frame_k]
		if isinstance(frame_val, torch.Tensor):
			frame_val = frame_val.to(device)

		sample[sample_k] = frame_val

	return sample


def get_augmented_transform(device,
							tx=0., ty=0., tz=0.,
							rr=0., rp=0., ry=0.,
							angle_units="rad",
							rot_params="xyz"
							):
	trans = torch.Tensor([tx, ty, tz]).to(device)
	rot = torch.Tensor([rr, rp, ry]).to(device)

	if angle_units != "rad":
		rot = rot * np.pi / 180

	return get_rt_matrix(trans, rot, rot_parmas=rot_params)


def augment_sample(device, sample, tf, key="anchor"):
	inv_tf = torch.linalg.inv(tf)

	pose = get_rt_matrix(sample[f"{key}_pose"].to(device), sample[f"{key}_rot"].to(device), rot_parmas="xyz")
	augmented_pose = pose @ tf

	augmented_tra_rot = mat2xyzrpy(augmented_pose)

	sample[f"{key}_pose"] = augmented_tra_rot[:3]
	sample[f"{key}_rot"] = augmented_tra_rot[3:]

	points = torch.ones_like(sample[key]).to(device)
	points[:, :3] = sample[key][:, :3]
	augmented_points = (inv_tf @ points.T).T
	augmented_points[:, 3] = sample[key][:, 3]
	sample[key] = augmented_points

	return sample


def get_sample(dataset, frames, device, augment_positive=None):

	# Hack for dealing with the indices of the KittiDataset class,
	# since they are not the ones from the dataset, but rather depend on the GT Loop file...
	# dataset_idx = {dataset.loop_gt[k]['idx']: k for k in range(len(dataset))}
	# frames_idx = [dataset_idx[frame] for frame in frames]

	# Get the data for the anchor and positive samples.
	# Not using the auto-chosen ones by the GT Loop file to allow an arbitrary selection of any two frames
	anc_frame = dataset[frames[0]]


	pos_frame = dataset[frames[1]]

	# We can generate the second sample by augmenting the second
	if augment_positive is not None:
		augmented_tf = get_augmented_transform(device, **augment_positive)
		pos_frame = augment_sample(device, pos_frame, augmented_tf, key="anchor")

	# Now recombine the two frames into a single sample dictionary
	sample = move_to_sample(frame=anc_frame, sample_pfx="", frame_pfx="",
							suffixes=["sequence", "class_one_hot_map", "superclass_one_hot_map"], device=device)
	sample = move_to_sample(frame=anc_frame, sample_pfx="anchor", frame_pfx="anchor", sample=sample, device=device)
	sample = move_to_sample(frame=pos_frame, sample_pfx="positive", frame_pfx="anchor", sample=sample, device=device)

	# Delete the loaded frames to release memory just in case
	del anc_frame, pos_frame
	return sample


def compute_samples_transforms(samples):
	keys = ["anchor", "positive", "negative"]

	for k in keys:
		if k + "_pose" in samples and k + "_transform" not in samples:
			poses = samples[k + "_pose"]
			rotat = samples[k + "_rot"]
			samples[k + "_transform"] = [get_rt_matrix(p, r, rot_parmas="xyz") for p, r in zip(poses, rotat)]

	if "p2a_transform" not in samples:
		samples["p2a_transform"] = [torch.linalg.inv(anc) @ pos
									for anc, pos in zip(samples["anchor_transform"], samples["positive_transform"])]
		samples["a2p_transform"] = [torch.linalg.inv(p2a) for p2a in samples["p2a_transform"]]

	return samples


def samples_np_to_torch(samples, device):
	for k, v in samples.items():
		if isinstance(v, np.ndarray):
			samples[k] = torch.from_numpy(v).float().to(device)


def collate_samples(samples, model, device):
	if not isinstance(samples, list):
		samples = [samples]

	sample_keys = samples[0].keys()
	collated_sample = {k: [s[k] for s in samples] for k in sample_keys}

	preproc_keys = ["anchor", "positive", "negative"]

	model_in = []
	for k in preproc_keys:
		if k in collated_sample:
			preproc_sample = [model.backbone.prepare_input(subsample) for subsample in collated_sample[k]]
			model_in += preproc_sample
			# del collated_sample[k]

	model_in = KittiDataset.collate_batch(model_in)
	samples_np_to_torch(model_in, device)
	model_in.update(collated_sample)
	compute_samples_transforms(model_in)

	return model_in


def get_samples(dataset, frame_list, model, device, augment_positive=None):
	samples = [get_sample(dataset, frames, device, augment_positive=augment_positive) for frames in frame_list]
	model_in = collate_samples(samples, model, device)

	return model_in


def iso_transform(device):
	zeros = torch.zeros(3).to(device)
	rot_z = torch.Tensor([0., 0., np.pi/4]).to(device)
	rot_x = torch.Tensor([np.arcsin(np.tan(30*np.pi/180)), 0., 0.]).to(device)
	rot_z_matrix = get_rt_matrix(zeros, rot_z, rot_parmas="xyz")
	rot_x_matrix = get_rt_matrix(zeros, rot_x, rot_parmas="xyz")
	rot_iso = rot_x_matrix @ rot_z_matrix
	return rot_iso


def transform_vertices(*, vertices, transform_matrix, device):
	# Convert to homogeneous coordinates
	n_vertices, vertex_dim = vertices.shape
	homo_vertices = torch.ones((n_vertices, vertex_dim + 1), device=device)
	homo_vertices[:, :vertex_dim] = vertices[:, :vertex_dim].clone()
	# Apply transformation matrix
	tf_vertices = (transform_matrix @ homo_vertices.T).T
	# Return to euclidean coordinates
	tf_vertices = tf_vertices[:, :vertex_dim] / tf_vertices[:, vertex_dim:]
	return tf_vertices


def plot_matches(*, anc_pc, anc_samp_ids,
				 pos_pc, pos_samp_ids, p2a_transform,
				 pred_pos_pc, match_weights,
				 device, style,
				 image_path=None,
				 z_distance=100.):

	# Anchor Point Cloud's Transformation (only the isometric projection)
	iso_tf = iso_transform(device)

	# Positive Point Cloud's Transformation
	# Move to anchor reference frame
	pos_rel_matrix = p2a_transform
	# Translate in z
	pos_traz_matrix = get_rt_matrix(torch.Tensor([0., 0., -z_distance]).to(device), torch.zeros(3).to(device),
									rot_parmas="xyz")
	pos_fin_matrix = pos_traz_matrix @ pos_rel_matrix
	# Isometric Projection
	pos_fin_matrix = iso_tf @ pos_fin_matrix

	# Transform Point Clouds
	proj_anc_pc = transform_vertices(vertices=anc_pc[:, :3], transform_matrix=iso_tf,
									 device=device).cpu().numpy()
	proj_pos_pc = transform_vertices(vertices=pos_pc[:, :3], transform_matrix=pos_fin_matrix,
									 device=device).cpu().numpy()
	proj_pred_pos_pc = transform_vertices(vertices=pred_pos_pc[:, :3], transform_matrix=pos_fin_matrix,
										  device=device).cpu().numpy()

	anc_samp_ids = anc_samp_ids.cpu().numpy()
	pos_samp_ids = pos_samp_ids.cpu().numpy()

	# Split point clouds into sampled and unsampled points
	proj_samp_anc_pc = proj_anc_pc[anc_samp_ids]
	proj_samp_pos_pc = proj_pos_pc[pos_samp_ids]
	proj_uns_anc_pc = np.delete(proj_anc_pc, anc_samp_ids, axis=0)
	proj_uns_pos_pc = np.delete(proj_pos_pc, pos_samp_ids, axis=0)

	fig = plt.figure(dpi=style.point_cloud_dpi, figsize=(4.1, 5.8))
	ax = fig.add_subplot()
	ax.axis('off')
	ax.set_aspect('equal')

	# Plot Anchor Point Cloud
	ax.scatter(proj_uns_anc_pc[:, 0], proj_uns_anc_pc[:, 1], zorder=1, **style.src_point_cloud)
	ax.scatter(proj_samp_anc_pc[:, 0], proj_samp_anc_pc[:, 1], zorder=2, **style.src_sampled_point_cloud)

	# Plot Matching Lines
	match_segments = np.concatenate(
		[proj_samp_anc_pc[:, :2].reshape(-1, 1, 2), proj_pred_pos_pc[:, :2].reshape(-1, 1, 2)], axis=1)
	match_lines = LineCollection(match_segments, **style.match_lines_styles(match_weights), zorder=11)
	ax.add_collection(match_lines)

	# Plot Positive Point Cloud
	ax.scatter(proj_uns_pos_pc[:, 0], proj_uns_pos_pc[:, 1], zorder=21,  **style.tgt_point_cloud)
	ax.scatter(proj_samp_pos_pc[:, 0], proj_samp_pos_pc[:, 1], zorder=22, **style.tgt_point_cloud)
	ax.scatter(proj_pred_pos_pc[:, 0], proj_pred_pos_pc[:, 1], zorder=23, **style.tgt_predicted_point_cloud)

	fig.tight_layout(pad=0.05)
	if image_path is None:
		fig.show()
		return

	print(f"Saving figure {image_path}.")
	fig.savefig(image_path)
	plt.close()


def infer_and_plot(*, dataset, device, model, frames, style, image_paths, augment_positive=None, **_):

	samples = get_samples(dataset, frames, model, device, augment_positive=augment_positive)

	with torch.no_grad():
		model(samples, metric_head=True, compute_embeddings=True,
			  compute_transl=False, compute_rotation=True, mode="pairs")

	batch_size = len(samples["anchor_idx"])
	if len(image_paths) != batch_size:
		print(f"Warning: Different number of image paths {len(image_paths)} than batches {batch_size}. Not saving.")
		image_paths = [None] * batch_size

	for i, image_path in enumerate(image_paths):
		anc_samp_ids = samples["keypoint_idxs"][i]
		pos_samp_ids = samples["keypoint_idxs"][batch_size + i]

		plot_matches(anc_pc=samples["anchor"][i], anc_samp_ids=anc_samp_ids,
					 pos_pc=samples["positive"][i], pos_samp_ids=pos_samp_ids,
					 p2a_transform=samples["p2a_transform"][i],
					 pred_pos_pc=samples["sinkhorn_matches"][i],
					 match_weights=samples["conf_weights"][i],
					 device=device, style=style,
					 image_path=image_path,
					 z_distance=120.)


def path_arg(value):
	return Path(value).expanduser()


def path_list_arg(value):
	return [path_arg(p) for p in value.split(",")]


def seq_arg(value):
	return f"{int(value):02d}"


def frames_arg(value):
	return [[int(frame) for frame in frames.split(",")] for frames in value.split(";")]


def parse_args():
	parser = ArgumentParser()

	parser.add_argument("--kitti_path", "-k", type=path_arg)
	parser.add_argument("--seq", "-s", type=seq_arg)
	parser.add_argument("--frames", "-i", type=frames_arg)

	parser.add_argument("--cp_path", "-c", type=path_arg)

	parser.add_argument("--gpu", "-g", type=int, default=0)

	parser.add_argument("--style", default="bw")

	parser.add_argument("--image_paths", "-f", type=path_list_arg)

	args = parser.parse_args()

	return vars(args)


def main(kitti_path, seq, cp_path, frames, image_paths, gpu=0, style="bw", augment_positive=None):
	kitti_dataset = load_dataset(kitti_path=kitti_path, seq=seq)

	torch.cuda.set_device(gpu)
	cuda_device = torch.device(gpu)

	model = load_model(cp_file=cp_path, device=cuda_device)

	if isinstance(style, str):
		style = Style(style)

	infer_and_plot(dataset=kitti_dataset, device=cuda_device, frames=frames, model=model,
				   style=style, image_paths=image_paths, augment_positive=augment_positive)


if __name__ == "__main__":
	cli_args = parse_args()
	main(**cli_args)

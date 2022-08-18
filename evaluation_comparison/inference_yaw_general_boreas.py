import argparse
import os
import pickle
import time

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from datasets.MulRan import MulRan
from datasets.Boreas import Boreas
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from scipy.spatial import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from models.get_models import get_model
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from utils.data import merge_inputs, Timer
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT
from utils.qcqp_layer import QuadQuatFastSolver
# from evaluation_comparison.TEASER import find_correspondences, get_teaser_solver, Rt2T, find_correspondences_faiss

import open3d as o3d
if hasattr(o3d, 'pipelines'):
	reg_module = o3d.pipelines.registration
else:
	reg_module = o3d.registration

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
	seed = seed + worker_id + epoch * 100
	seed = seed % (2**32 - 1)
	print(f"Init worker {worker_id} with seed {seed}")
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def get_database_embs(model, sample, exp_cfg, device):
	model.eval()
	margin = exp_cfg['margin']

	with torch.no_grad():
		if exp_cfg['training_type'] == "3D":
			anchor_list = []
			for i in range(len(sample['anchor'])):
				anchor = sample['anchor'][i].to(device)

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
					a = anchor_set[0, :].long()
					anchor_i = anchor[a]
				else:
					anchor_i = anchor

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_list.append(anchor_i[:, :3].unsqueeze(0))
				else:
					anchor_list.append(model.backbone.prepare_input(anchor_i))
					del anchor_i

			if exp_cfg['3D_net'] != 'PVRCNN':
				anchor = torch.cat(tuple(anchor_list), 0)
				model_in = anchor
				model_in = model_in / 100.
			else:
				model_in = KittiDataset.collate_batch(anchor_list)
				for key, val in model_in.items():
					if not isinstance(val, np.ndarray):
						continue
					model_in[key] = torch.from_numpy(val).float().to(device)

			batch_dict = model(model_in, metric_head=False)
			anchor_out = batch_dict['out_embedding']

		else:
			anchor_out = model(sample['anchor'].to(device), metric_head=False)

	if exp_cfg['norm_embeddings']:
		anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
	return anchor_out


class SamplePairs(Sampler):

	def __init__(self, data_source, pairs):
		super(SamplePairs, self).__init__(data_source)
		self.pairs = pairs

	def __len__(self):
		return len(self.pairs)

	def __iter__(self):
		return [self.pairs[i, 0] for i in range(len(self.pairs))]


class BatchSamplePairs(BatchSampler):

	def __init__(self, data_source, pairs, batch_size):
		# super(BatchSamplePairs, self).__init__(batch_size, True)
		self.pairs = pairs
		self.batch_size = batch_size
		self.count = 0

	def __len__(self):
		tot = 2*len(self.pairs)
		ret = (tot + self.batch_size - 1) // self.batch_size
		return ret

	def __iter__(self):
		self.count = 0
		while 2*self.count + self.batch_size < 2*len(self.pairs):
			current_batch = []
			for i in range(self.batch_size//2):
				current_batch.append(self.pairs[self.count+i, 0])
			for i in range(self.batch_size//2):
				current_batch.append(self.pairs[self.count+i, 1])
			yield current_batch
			self.count += self.batch_size//2
		if 2*self.count < 2*len(self.pairs):
			diff = 2*len(self.pairs)-2*self.count
			current_batch = []
			for i in range(diff//2):
				current_batch.append(self.pairs[self.count+i, 0])
			for i in range(diff//2):
				current_batch.append(self.pairs[self.count+i, 1])
			yield current_batch


def main_process(gpu, weights_path, *,
				 dataset_path,
				 dataset="boreas",
				 seq1=None,
				 seq2=None,
				 num_iters=1,
				 do_ransac=False,
				 do_icp=False,
				 save_path=None,
				 batch_size=2,
				 **_
				 ):
	global EPOCH
	rank = gpu

	torch.cuda.set_device(gpu)
	device = torch.device("cuda")

	saved_params = torch.load(weights_path, map_location='cpu')
	exp_cfg = saved_params['config']
	exp_cfg['batch_size'] = batch_size
	exp_cfg["pvrcnn_cfg_file"] = "./models/backbone3D/pv_rcnn_boreas.yaml"

	if 'loop_file' not in exp_cfg:
		exp_cfg['loop_file'] = 'loop_GT'
	if 'sinkhorn_type' not in exp_cfg:
		exp_cfg['sinkhorn_type'] = 'flot'
	if 'shared_embeddings' not in exp_cfg:
		exp_cfg['shared_embeddings'] = False
	if 'use_semantic' not in exp_cfg:
		exp_cfg['use_semantic'] = False
	if 'use_panoptic' not in exp_cfg:
		exp_cfg['use_panoptic'] = False
	if 'noneg' in exp_cfg['loop_file']:
		exp_cfg['loop_file'] = 'loop_GT_4m'
	if 'head' not in exp_cfg:
		exp_cfg['head'] = 'SuperGlue'
	# exp_cfg['without_ground'] = False

	current_date = datetime.now()

	if dataset == 'mulran':
		seq1 = seq1 if seq1 is not None else "Riverside01"
		seq2 = seq2 if seq2 is not None else "Riverside02"
		dataset_for_recall1 = MulRan(dataset_path, seq1)
		dataset_for_recall3 = MulRan(dataset_path, seq2)
	else:
		seq1 = seq1 if seq1 is not None else "boreas-2020-12-18-13-44"
		seq2 = seq2 if seq2 is not None else "boreas-2020-11-26-13-58"
		dataset_for_recall1 = Boreas(dataset_path, seq1)
		dataset_for_recall3 = Boreas(dataset_path, seq2)

	concat_dataset = torch.utils.data.ConcatDataset([dataset_for_recall1, dataset_for_recall3])

	test_pair_idxs = []
	test_pair_idxs_concat = []
	index = faiss.IndexFlatL2(3)
	poses1 = np.stack(dataset_for_recall1.poses).copy()
	poses3 = np.stack(dataset_for_recall3.poses).copy()
	index.add(poses3[:, :3, 3].astype(np.float32).copy())
	num_frames_with_loop = 0
	num_frames_with_reverse_loop = 0
	for i in tqdm(range(len(dataset_for_recall1.poses))):
		current_pose = poses1[i:i+1, :3, 3].astype(np.float32).copy()
		lims, D, I = index.range_search(current_pose, 4.**2)
		# for j in range(lims[0], lims[1]):
		if lims[1] > 0:
			j = 0
			if j == 0:
				num_frames_with_loop += 1
				yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses3[I[j]]) @ poses1[i])[-1]
				yaw_diff = yaw_diff % (2 * np.pi)
				if 0.79 <= yaw_diff <= 5.5:
					num_frames_with_reverse_loop += 1
				# else:
				#     print(yaw_diff)
			test_pair_idxs.append([I[j], i])
			test_pair_idxs_concat.append([len(dataset_for_recall1)+I[j], i])
	test_pair_idxs = np.array(test_pair_idxs)
	test_pair_idxs_concat = np.array(test_pair_idxs_concat)

	batch_sampler = BatchSamplePairs(concat_dataset, test_pair_idxs_concat, exp_cfg['batch_size'])
	RecallLoader = torch.utils.data.DataLoader(dataset=concat_dataset,
											   # batch_size=exp_cfg['batch_size'],
											   num_workers=2,
											   # sampler=sampler,
											   batch_sampler=batch_sampler,
											   # worker_init_fn=init_fn,
											   collate_fn=merge_inputs,
											   pin_memory=True)

	model = get_model(exp_cfg)

	model.load_state_dict(saved_params['state_dict'], strict=True)

	# model.train()
	model = model.to(device)

	local_iter = 0.
	transl_error_sum = 0
	yaw_error_sum = 0
	emb_list = []
	rot_errors = []
	transl_errors = []
	yaw_error = []
	for i in range(num_iters):
		rot_errors.append([])
		transl_errors.append([])

	time_net, time_ransac, time_icp = Timer(), Timer(), Timer()

	# Testing
	if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
		# all_feats = []
		# all_coords = []
		# save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
		current_frame = 0
		yaw_preds = torch.zeros((len(dataset_for_recall3.poses), len(dataset_for_recall1.poses)))
		transl_errors = []
		for batch_idx, sample in enumerate(tqdm(RecallLoader)):
			if batch_idx==1:
				time_net.reset()
				time_ransac.reset()
				time_icp.reset()
			if batch_idx % 10 == 9:
				print("")
				print("Time Network: ", time_net.avg)
				print("Time RANSAC: ", time_ransac.avg)
				print("Time ICP: ", time_icp.avg)

			start_time = time.time()

			### AAA
			model.eval()
			with torch.no_grad():

				anchor_list = []
				for i in range(len(sample['anchor'])):
					anchor = sample['anchor'][i].to(device)

					if exp_cfg['3D_net'] != 'PVRCNN':
						anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
						a = anchor_set[0, :].long()
						anchor_i = anchor[a]
					else:
						anchor_i = anchor

					if exp_cfg['3D_net'] != 'PVRCNN':
						anchor_list.append(anchor_i[:, :3].unsqueeze(0))
					else:
						anchor_list.append(model.backbone.prepare_input(anchor_i))
						del anchor_i

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor = torch.cat(anchor_list)
					model_in = anchor
					# Normalize between [-1, 1], more or less
					# model_in = model_in / 100.
					if exp_cfg['3D_net'] == 'RandLANet':
						model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
				else:
					model_in = KittiDataset.collate_batch(anchor_list)
					for key, val in model_in.items():
						if not isinstance(val, np.ndarray):
							continue
						model_in[key] = torch.from_numpy(val).float().to(device)

				torch.cuda.synchronize()
				time_net.tic()
				batch_dict = model(model_in, metric_head=True)
				torch.cuda.synchronize()
				time_net.toc()
				pred_transl = []
				yaw = batch_dict['out_rotation']

				if exp_cfg['rot_representation'].startswith('sincos'):
					yaw = torch.atan2(yaw[:, 1], yaw[:, 0])
					for i in range(batch_dict['batch_size'] // 2):
						yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] =yaw[i]
						pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
				elif exp_cfg['rot_representation'] == 'quat':
					yaw = F.normalize(yaw, dim=1)
					for i in range(batch_dict['batch_size'] // 2):
						yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(RT.quat2mat(yaw[i]))[-1]
						pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
				elif exp_cfg['rot_representation'] == 'bingham':
					to_quat = QuadQuatFastSolver()
					quat_out = to_quat.apply(yaw)[:, [1,2,3,0]]
					for i in range(yaw.shape[0]):
						yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(RT.quat2mat(quat_out[i]))[-1]
						pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
				elif exp_cfg['rot_representation'].startswith('rpm') and not do_ransac:
					transformation = batch_dict['transformation']
					homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
					transformation = torch.cat((transformation, homogeneous), dim=1)
					transformation = transformation.inverse()
					for i in range(batch_dict['batch_size'] // 2):
						yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(transformation[i])[-1].item()
						pred_transl.append(transformation[i][:3, 3].detach().cpu())
				elif exp_cfg['rot_representation'].startswith('6dof') and not do_ransac:
					transformation = batch_dict['transformation']
					homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
					transformation = torch.cat((transformation, homogeneous), dim=1)
					transformation = transformation.inverse()
					for i in range(batch_dict['batch_size'] // 2):
						# yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = mat2xyzrpy(transformation[i])[-1].item()
						yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(transformation[i])[-1].item()
						pred_transl.append(transformation[i][:3, 3].detach().cpu())
				elif do_ransac:
					coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
					feats = batch_dict['point_features'].squeeze(-1)
					for i in range(batch_dict['batch_size'] // 2):
						coords1 = coords[i]
						coords2 = coords[i + batch_dict['batch_size'] // 2]
						feat1 = feats[i]
						feat2 = feats[i + batch_dict['batch_size'] // 2]
						pcd1 = o3d.geometry.PointCloud()
						pcd1.points = o3d.utility.Vector3dVector(coords1[:, 1:].cpu().numpy())
						pcd2 = o3d.geometry.PointCloud()
						pcd2.points = o3d.utility.Vector3dVector(coords2[:, 1:].cpu().numpy())
						pcd1_feat = reg_module.Feature()
						pcd1_feat.data = feat1.permute(0, 1).cpu().numpy()
						pcd2_feat = reg_module.Feature()
						pcd2_feat.data = feat2.permute(0, 1).cpu().numpy()

						torch.cuda.synchronize()
						time_ransac.tic()
						try:
							result = reg_module.registration_ransac_based_on_feature_matching(
								pcd2, pcd1, pcd2_feat, pcd1_feat, True,
								0.6,
								reg_module.TransformationEstimationPointToPoint(False),
								3, [],
								reg_module.RANSACConvergenceCriteria(5000))
						except:
							result = reg_module.registration_ransac_based_on_feature_matching(
								pcd2, pcd1, pcd2_feat, pcd1_feat,
								0.6,
								reg_module.TransformationEstimationPointToPoint(False),
								3, [],
								reg_module.RANSACConvergenceCriteria(5000))
						time_ransac.toc()

						# index = faiss.IndexFlatL2(640)
						# feat1 = feat1.T.cpu().numpy()
						# feat2 = feat2.T.cpu().numpy()
						# torch.cuda.synchronize()
						# time_ransac.tic()
						# index.add(feat1)
						# _, corr = index.search(feat2, 1)
						# corr = np.stack((np.arange(4096), corr[:,0]), axis=1).astype(np.int32)
						# corr2 = o3d.utility.Vector2iVector(corr.copy())
						# try:
						#     result = reg_module.registration_ransac_based_on_correspondence(
						#         pcd2, pcd1, corr2,
						#         0.6,
						#         reg_module.TransformationEstimationPointToPoint(False),
						#         3, [],
						#         reg_module.RANSACConvergenceCriteria(500))
						# except:
						#     pass
						# time_ransac.toc()
						transformation = torch.tensor(result.transformation.copy())
						if do_icp:
							p1 = o3d.geometry.PointCloud()
							p1.points = o3d.utility.Vector3dVector(sample['anchor'][i][:, :3].cpu().numpy())
							p2 = o3d.geometry.PointCloud()
							p2.points = o3d.utility.Vector3dVector(
								sample['anchor'][i + batch_dict['batch_size'] // 2][:, :3].cpu().numpy())
							time_icp.tic()
							result2 = reg_module.registration_icp(
								p2, p1, 0.1, result.transformation,
								reg_module.TransformationEstimationPointToPoint())
							time_icp.toc()
							transformation = torch.tensor(result2.transformation.copy())
						yaw_preds[test_pair_idxs[current_frame + i, 0], test_pair_idxs[current_frame + i, 1]] = \
							mat2xyzrpy(transformation)[-1].item()
						pred_transl.append(transformation[:3, 3].detach().cpu())
				# if args.teaser:
				# 	coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
				# 	feats = batch_dict['point_features'].squeeze(-1)
				# 	for i in range(batch_dict['batch_size'] // 2):
				# 		coords1 = coords[i]
				# 		coords2 = coords[i + batch_dict['batch_size'] // 2]
				# 		feat1 = feats[i]
				# 		feat2 = feats[i + batch_dict['batch_size'] // 2]
				#
				# 		torch.cuda.synchronize()
				# 		time_ransac.tic()
				# 		corrs_A, corrs_B = find_correspondences_faiss(
				# 			feat1.cpu().numpy().T, feat2.cpu().numpy().T, mutual_filter=True)
				# 		A_corr = coords1.cpu().numpy()[corrs_A, 1:].T.copy()  # np array of size 3 by num_corrs
				# 		B_corr = coords2.cpu().numpy()[corrs_B, 1:].T.copy()  # np array of size 3 by num_corrs
				# 		NOISE_BOUND = 0.1
				# 		teaser_solver = get_teaser_solver(NOISE_BOUND)
				# 		# teaser_solver.solve(A_corr, B_corr)
				# 		teaser_solver.solve(B_corr, A_corr)
				# 		solution = teaser_solver.getSolution()
				# 		time_ransac.toc()
				# 		R_teaser = solution.rotation
				# 		t_teaser = solution.translation
				# 		T_teaser = Rt2T(R_teaser, t_teaser)
				# 		transformation = torch.tensor(T_teaser)
				# 		yaw_preds[test_pair_idxs[current_frame + i, 0], test_pair_idxs[current_frame + i, 1]] = \
				# 			mat2xyzrpy(transformation)[-1].item()
				# 		pred_transl.append(transformation[:3, 3].detach().cpu())

				for i in range(batch_dict['batch_size'] // 2):
					pose1 = dataset_for_recall3.poses[test_pair_idxs[current_frame+i, 0]]
					pose2 = dataset_for_recall1.poses[test_pair_idxs[current_frame+i, 1]]
					delta_pose = np.linalg.inv(pose1) @ pose2
					transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl[i]
					transl_errors.append(transl_error.norm())

					yaw_pred = yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]]
					yaw_pred = yaw_pred % (2 * np.pi)
					delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
					delta_yaw = delta_yaw % (2 * np.pi)
					diff_yaw = abs(delta_yaw - yaw_pred)
					diff_yaw = diff_yaw % (2 * np.pi)
					diff_yaw = (diff_yaw * 180) / np.pi
					if diff_yaw > 180.:
						diff_yaw = 360 - diff_yaw
					yaw_error.append(diff_yaw)

				current_frame += batch_dict['batch_size'] // 2

				# for i in range(batch_dict['point_features'].shape[0]):
				#     save_file = os.path.join(save_folder, f'{current_frame:06d}.h5')
				#     with h5py.File(save_file, 'w') as hf:
				#         hf.create_dataset('feats', data=batch_dict['point_features'].detach().cpu().numpy(),
				#                           compression='lzf', shuffle=True)
				#         hf.create_dataset('coords', data=batch_dict['point_coords'].detach().cpu().numpy(),
				#                           compression='lzf', shuffle=True)
				#     current_frame += 1
				# all_feats.append(batch_dict['point_features'].detach().cpu())
				# all_coords.append(batch_dict['point_coords'].detach().cpu())

			### AAA
	print(weights_path)
	print(exp_cfg['test_sequence'])

	transl_errors = np.array(transl_errors)
	yaw_error = np.array(yaw_error)

	valid = yaw_error <= 5.
	valid = valid & (np.array(transl_errors) <= 2.)
	succ_rate = valid.sum() / valid.shape[0]
	rte_suc = transl_errors[valid].mean()
	rre_suc = yaw_error[valid].mean()

	save_dict = {
		'rot': yaw_error,
		'transl': transl_errors,
		"Mean rotation error": yaw_error.mean(),
		"Median rotation error": np.median(yaw_error),
		"STD rotation error": yaw_error.std(),
		"Mean translation error": transl_errors.mean(),
		"Median translation error": np.median(transl_errors),
		"STD translation error": transl_errors.std(),
		"Success Rate": succ_rate,
		"RTE": rte_suc,
		"RRE": rre_suc,
	}

	# save_path = f'./results_for_paper/lcdnet00+08_{exp_cfg["test_sequence"]}'
	# if '360' in weights_path:
	# 	save_path = f'./results_for_paper/lcdnet++_{exp_cfg["test_sequence"]}'
	# else:
	# 	save_path = f'./results_for_paper/lcdnet00+08_{exp_cfg["test_sequence"]}'
	# if args.remove_random_angle > 0:
	# 	save_path = save_path + f'_remove{args.remove_random_angle}'
	# if args.icp:
	# 	save_path = save_path+'_icp'
	# elif args.ransac:
	# 	save_path = save_path+'_ransac'
	# if args.teaser:
	# 	save_path = save_path + '_teaser'

	print(f"Success Rate: {succ_rate}, RTE: {rte_suc}, RRE: {rre_suc}")

	if save_path:
		print("Saving to ", save_path)
		with open(save_path, 'wb') as f:
			pickle.dump(save_dict, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path', default='/home/cattaneo/Datasets/KITTI',
						help='dataset directory')
	parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
	parser.add_argument('--num_iters', type=int, default=1)
	parser.add_argument('--dataset', type=str, default='kitti')
	parser.add_argument('--do_ransac', action='store_true', default=False)
	parser.add_argument('--do_teaser', action='store_true', default=False)
	parser.add_argument('--do_icp', action='store_true', default=False)
	parser.add_argument('--remove_random_angle', type=int, default=-1)
	parser.add_argument('--save_path', type=str, default=None)
	tmp_args = parser.parse_args()

	# if args.device is not None and not args.no_cuda:
	#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	#     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

	main_process(**vars(tmp_args))

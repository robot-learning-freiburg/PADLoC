import argparse

from datasets.MulRan import MulRan
from datasets.Boreas import Boreas
import os
import pickle
import time
from collections import OrderedDict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.Freiburg import FreiburgDataset
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from evaluation_comparison.plot_PR_curve import compute_AP, compute_PR_pairs
from models.get_models import get_model
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from utils.data import merge_inputs
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
	seed = seed + worker_id + epoch * 100
	seed = seed % (2**32 - 1)
	print(f"Init worker {worker_id} with seed {seed}")
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def compute_PR_mulran(pair_dist, poses1, poses3, is_distance=True, ignore_last=False):
	real_loop = []
	detected_loop = []
	distances = []
	index = faiss.IndexFlatL2(3)
	index.add(poses3[:, :3, 3].astype(np.float32).copy())

	for i in tqdm(range(poses1.shape[0])):
		current_pose = poses1[i:i+1, :3, 3].copy().astype(np.float32)
		lims, D, I = index.range_search(current_pose, 4.**2)
		if lims[1] > 0:
			real_loop.append(1)
		else:
			real_loop.append(0)

		if is_distance:
			candidate = pair_dist[i, :].argmin()
			detected_loop.append(-pair_dist[i, candidate])
		else:
			candidate = pair_dist[i, :].argmax()
			detected_loop.append(pair_dist[i, candidate])
		candidate_pose = poses3[candidate][:3, 3]
		distances.append(np.linalg.norm(candidate_pose-current_pose))

	distances = np.array(distances)
	detected_loop = -np.array(detected_loop)
	real_loop = np.array(real_loop)
	precision_fn = []
	recall_fn = []
	for thr in np.unique(detected_loop):
		asd = detected_loop<=thr
		asd = asd & real_loop
		asd = asd & (distances <= 4)
		tp = asd.sum()
		fn = (detected_loop<=thr) & (distances > 4) & real_loop
		fn2 = (detected_loop > thr) & real_loop
		fn = fn.sum() + fn2.sum()
		fp = (detected_loop<=thr) & (distances > 4) & (1 - real_loop)
		fp = fp.sum()
		# fp = (detected_loop<=thr).sum() - tp
		# fn = (real_loop.sum()) - tp
		if (tp+fp) > 0:
			precision_fn.append(tp/(tp+fp))
		else:
			precision_fn.append(1.)
		recall_fn.append(tp/(tp+fn))
	precision_fp = []
	recall_fp = []
	for thr in np.unique(detected_loop):
		asd = detected_loop<=thr
		asd = asd & real_loop
		asd = asd & (distances <= 4)
		tp = asd.sum()
		fp = (detected_loop<=thr) & (distances > 4)
		fp = fp.sum()
		fn = (detected_loop > thr) & (real_loop)
		fn = fn.sum()
		if (tp+fp) > 0:
			precision_fp.append(tp/(tp+fp))
		else:
			precision_fp.append(1.)
		recall_fp.append(tp/(tp+fn))

	return precision_fn, recall_fn, precision_fp, recall_fp


def prepare_input(model, samples, exp_cfg, device):
	anchor_list = []
	for point_cloud in samples:
		if exp_cfg['3D_net'] != 'PVRCNN':
			anchor_set = furthest_point_sample(point_cloud[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
			a = anchor_set[0, :].long()
			anchor_i = point_cloud[a]
		else:
			anchor_i = point_cloud

		if exp_cfg['3D_net'] != 'PVRCNN':
			anchor_list.append(anchor_i[:, :3].unsqueeze(0))
		else:
			anchor_list.append(model.module.backbone.prepare_input(anchor_i))
			del anchor_i

	if exp_cfg['3D_net'] != 'PVRCNN':
		point_cloud = torch.cat(tuple(anchor_list), 0)
		model_in = point_cloud
		# model_in = model_in / 100.
		if exp_cfg['3D_net'] == 'RandLANet':
			model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
	else:
		model_in = KittiDataset.collate_batch(anchor_list)
		for key, val in model_in.items():
			if not isinstance(val, np.ndarray):
				continue
			model_in[key] = torch.from_numpy(val).float().to(device)
	return model_in


def geometric_verification(model, dataset, id_query, id_candidate, device, exp_cfg):
	model.eval()
	with torch.no_grad():
		anchor_list = []

		sample_query = dataset.__getitem__(id_query)
		sample_candidate = dataset.__getitem__(id_candidate)
		query_pc = sample_query['anchor'].to(device)
		candidate_pc = sample_candidate['anchor'].to(device)

		model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

		batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

		transformation = batch_dict['transformation']
		homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
		transformation = torch.cat((transformation, homogeneous), dim=1)
		# TEST
		# transformation[0, 0, 3] = 0
		# transformation[0, 1, 3] = 0
		# transformation[0, 2, 3] = 0

		query_intensity = query_pc[:, -1].clone()
		query_pc = query_pc.clone()
		query_pc[:, -1] = 1.
		transformed_query_pc = (transformation[0] @ query_pc.T).T
		transformed_query_pc[:, -1] = query_intensity

		model_in = prepare_input(model, [transformed_query_pc, candidate_pc], exp_cfg, device)

		batch_dict = model(model_in, metric_head=False, compute_embeddings=True)

		emb = batch_dict['out_embedding']
		if exp_cfg['norm_embeddings']:
			emb = emb / emb.norm(dim=1, keepdim=True)

	return (emb[0] - emb[1]).norm().detach().cpu()


def geometric_verification2(model, dataset, id_query, id_candidate, device, exp_cfg):
	model.eval()
	with torch.no_grad():

		sample_query = dataset.__getitem__(id_query)
		sample_candidate = dataset.__getitem__(id_candidate)
		query_pc = sample_query['anchor'].to(device)
		candidate_pc = sample_candidate['anchor'].to(device)

		model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

		batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

	return batch_dict['transport'].sum(-1).sum().detach().cpu()


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
		self.data_source = data_source
		self.pairs = pairs
		self.batch_size = batch_size
		self.count = 0

	def __len__(self):
		return 2*len(self.pairs)

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


def main_process(gpu, weights_path, common_seed, world_size, dataset_path,
				 dataset="boreas",
				 seq1=None,
				 seq2=None,
				 num_iters=1,
				 batch_size=15,
				 pr_filename=None,
				 stats_filename=None
				 ):
	global EPOCH
	rank = gpu
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)

	torch.cuda.set_device(gpu)
	device = torch.device(gpu)

	saved_params = torch.load(weights_path, map_location='cpu')

	# asd = torch.load('/home/cattaneod/rpmnet_08_4m_shared.tar', map_location='cpu')

	# exp_cfg = saved_params['config']
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

	final_dest = ''

	dataset1_sampler = torch.utils.data.distributed.DistributedSampler(
		dataset_for_recall1,
		num_replicas=world_size,
		rank=rank,
		seed=common_seed,
		shuffle=False
	)
	dataset3_sampler = torch.utils.data.distributed.DistributedSampler(
		dataset_for_recall3,
		num_replicas=world_size,
		rank=rank,
		seed=common_seed,
		shuffle=False
	)


	MapLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall1,
											sampler=dataset1_sampler,
											batch_size=exp_cfg['batch_size'],
											num_workers=2,
											collate_fn=merge_inputs,
											pin_memory=True)
	MapLoader3 = torch.utils.data.DataLoader(dataset=dataset_for_recall3,
											 sampler=dataset3_sampler,
											 batch_size=exp_cfg['batch_size'],
											 num_workers=2,
											 collate_fn=merge_inputs,
											 pin_memory=True)

	model = get_model(exp_cfg, is_training=False)
	renamed_dict = OrderedDict()
	for key in saved_params['state_dict']:
		if not key.startswith('module'):
			renamed_dict = saved_params['state_dict']
			break
		else:
			renamed_dict[key[7:]] = saved_params['state_dict'][key]

	res = model.load_state_dict(renamed_dict, strict=False)
	if len(res[0]) > 0:
		print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

	model.train()
	model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank, find_unused_parameters=True)

	local_iter = 0.
	transl_error_sum = 0
	yaw_error_sum = 0
	emb_list_map = []
	emb_list_map3 = []
	rot_errors = []
	transl_errors = []
	time_descriptors = []
	for i in range(num_iters):
		rot_errors.append([])
		transl_errors.append([])

	for batch_idx, sample in enumerate(tqdm(MapLoader)):

		model.eval()
		time1 = time.time()
		with torch.no_grad():

			anchor_list = []
			for i in range(len(sample['anchor'])):
				anchor = sample['anchor'][i].to(device)
				# anchor[:, -1] = 0.
				non_valid_idxs = torch.logical_and(anchor[:, 0] == 0, anchor[:, 1] == 0)
				non_valid_idxs = torch.logical_and(non_valid_idxs, anchor[:, 2] == 0)
				anchor = anchor[torch.logical_not(non_valid_idxs)]

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
					a = anchor_set[0, :].long()
					anchor_i = anchor[a]
				else:
					anchor_i = anchor

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_list.append(anchor_i[:, :3].unsqueeze(0))
				else:
					if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
						anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
					anchor_list.append(model.module.backbone.prepare_input(anchor_i))
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

			batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

			emb = batch_dict['out_embedding']
			dist.barrier()
			out_emb = [torch.zeros_like(emb) for _ in range(world_size)]
			dist.all_gather(out_emb, emb)
			if rank == 0:
				interleaved_out = torch.empty((emb.shape[0]*world_size, emb.shape[1]),
											  device=emb.device, dtype=emb.dtype)
				for current_rank in range(world_size):
					interleaved_out[current_rank::world_size] = out_emb[current_rank]
				emb_list_map.append(interleaved_out.detach().clone())

		time2 = time.time()
		time_descriptors.append(time2-time1)

	if rank == 0:
		emb_list_map = torch.cat(emb_list_map)
		emb_list_map = emb_list_map[:len(dataset_for_recall1)].cpu().numpy()

	dist.barrier()
	for batch_idx, sample in enumerate(tqdm(MapLoader3)):

		model.eval()
		with torch.no_grad():

			anchor_list = []
			for i in range(len(sample['anchor'])):
				anchor = sample['anchor'][i].to(device)
				# anchor[:, -1] = 0.
				non_valid_idxs = torch.logical_and(anchor[:, 0] == 0, anchor[:, 1] == 0)
				non_valid_idxs = torch.logical_and(non_valid_idxs, anchor[:, 2] == 0)
				anchor = anchor[torch.logical_not(non_valid_idxs)]

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
					a = anchor_set[0, :].long()
					anchor_i = anchor[a]
				else:
					anchor_i = anchor

				if exp_cfg['3D_net'] != 'PVRCNN':
					anchor_list.append(anchor_i[:, :3].unsqueeze(0))
				else:
					if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
						anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
					anchor_list.append(model.module.backbone.prepare_input(anchor_i))
					del anchor_i

			if exp_cfg['3D_net'] != 'PVRCNN':
				anchor = torch.cat(anchor_list)
				model_in = anchor
				if exp_cfg['3D_net'] == 'RandLANet':
					model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
			else:
				model_in = KittiDataset.collate_batch(anchor_list)
				for key, val in model_in.items():
					if not isinstance(val, np.ndarray):
						continue
					model_in[key] = torch.from_numpy(val).float().to(device)

			batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

			emb = batch_dict['out_embedding']
			dist.barrier()
			out_emb = [torch.zeros_like(emb) for _ in range(world_size)]
			dist.all_gather(out_emb, emb)
			if rank == 0:
				interleaved_out = torch.empty((emb.shape[0]*world_size, emb.shape[1]),
											  device=emb.device, dtype=emb.dtype)
				for current_rank in range(world_size):
					interleaved_out[current_rank::world_size] = out_emb[current_rank]
				emb_list_map3.append(interleaved_out.detach().clone())

	if rank == 0:
		emb_list_map3 = torch.cat(emb_list_map3)
		emb_list_map3 = emb_list_map3[:len(dataset_for_recall3)].cpu().numpy()

		# Recall@k
		recall = np.zeros(10)
		total_frame = 0

		emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
		emb_list_map_norm3 = emb_list_map3 / np.linalg.norm(emb_list_map3, axis=1, keepdims=True)
		pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm3)

		if pr_filename:
			print(f"Saving pairwise distances to {pr_filename}.")
			np.savez(pr_filename, pair_dist)

		precision_ours_fn, recall_ours_fn, precision_ours_fp, recall_ours_fp = compute_PR_mulran(pair_dist, dataset_for_recall1.poses, dataset_for_recall3.poses)
		ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
		ap_ours_fn = compute_AP(precision_ours_fn, recall_ours_fn)
		print(weights_path)
		print(exp_cfg['test_sequence'])
		print("AP FP: ", ap_ours_fp)
		print("AP FN: ", ap_ours_fn)
		# poses = np.concatenate([dataset_for_recall1.poses, dataset_for_recall3.poses])
		# precision_pair_ours, recall_pair_ours = compute_PR_pairs(pair_dist, poses)
		# precision_pair_ours = [x for _, x in sorted(zip(recall_pair_ours, precision_pair_ours))]
		# recall_pair_ours = sorted(recall_pair_ours)
		# ap_ours_pair = compute_AP(precision_pair_ours, recall_pair_ours)
		# print("AP Pairs: ", ap_ours_pair)

		if stats_filename:
			save_dict = {
				"AP FP": ap_ours_fp,
				"AP FN": ap_ours_fn,
				# "AP Pairs": ap_ours_pair
			}

			print(f"Saving Stats to {stats_filename}.")
			with open(stats_filename, "wb") as f:
				pickle.dump(save_dict, f)

		print("Done")


if __name__ == '__main__':

	def_gpu_count = torch.cuda.device_count()
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
						help='dataset directory')
	parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
	parser.add_argument('--num_iters', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=15)
	parser.add_argument('--dataset', type=str, default='boreas')
	parser.add_argument('--seq1', type=str, default=None)
	parser.add_argument('--seq2', type=str, default=None)
	parser.add_argument("--common_seed", type=int, default=42)
	parser.add_argument("--gpu_count", type=int, default=def_gpu_count)
	parser.add_argument("--pr_filename", type=str, default=None)
	parser.add_argument("--stats_filename", type=str, default=None)
	args = parser.parse_args()

	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '8989'

	args.gpu_count = torch.cuda.device_count()
	mp.spawn(main_process, nprocs=args.gpu_count, args=(
		args.weights_path, 42, args.gpu_count, args.data, args.dataset, args.seq1, args.seq2, args.num_iter,
		args.batch_size, args.pr_filename, args.stats_filename
	))

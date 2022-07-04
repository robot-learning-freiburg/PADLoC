# import argparse
from collections import namedtuple

from datasets.NCLTDataset import NCLTDataset
import os
# import pickle
# import time
# from collections import OrderedDict
#
# import faiss
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
# from pcdet.datasets.kitti.kitti_dataset import KittiDataset
# import random

# from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import KDTree
# from torch.utils.data.sampler import Sampler, BatchSampler
# from tqdm import tqdm

from datasets.Freiburg import FreiburgDataset
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from evaluation_comparison.plot_PR_curve import compute_PR, compute_AP, compute_PR_pairs
# from models.get_models import get_model
# from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
# from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
# from utils.data import merge_inputs
# from datetime import datetime
# from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

# torch.backends.cudnn.benchmark = True


def main_process(weights_path, dataset, npzfile, data, sequence):
	# global EPOCH
	# rank = gpu

	# torch.cuda.set_device(gpu)
	# device = torch.device(gpu)
	device = None

	saved_params = torch.load(weights_path, map_location='cpu')

	# asd = torch.load('/home/cattaneod/rpmnet_08_4m_shared.tar', map_location='cpu')

	# exp_cfg = saved_params['config']
	exp_cfg = saved_params['config']
	exp_cfg['batch_size'] = 6

	# current_date = datetime.now()

	# if sequence is None:
	# 	if dataset == 'kitti':
	# 		exp_cfg['test_sequence'] = "08"
	# 		sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
	# 	elif dataset == 'nclt':
	# 		exp_cfg['test_sequence'] = "2013-04-05"
	# 		sequences_training = ["2012-01-08", "2012-01-15", "2012-01-22", "2012-02-04", "2012-03-25",
	# 							  "2012-03-31", "2012-05-26", "2012-10-28", "2012-11-17", "2012-12-01"]
	# 	elif dataset == 'kitti360':
	# 		exp_cfg['test_sequence'] = "2013_05_28_drive_0002_sync"
	# 		sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
	# 							  "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
	# 							  "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
	# 	sequences_validation = [exp_cfg['test_sequence']]
	# 	sequences_training = set(sequences_training) - set(sequences_validation)
	# 	sequences_training = list(sequences_training)
	# 	sequence = sequences_validation[0]

	if 'loop_file' not in exp_cfg:
		exp_cfg['loop_file'] = 'loop_GT'
	if 'sinkhorn_type' not in exp_cfg:
		exp_cfg['sinkhorn_type'] = 'unbalanced'
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

	if dataset == 'kitti':
		dataset_for_recall = KITTILoader3DPoses(data, sequence,
												os.path.join(data, 'sequences', sequence, 'poses.txt'),
												exp_cfg['num_points'], device, train=False,
												use_semantic=exp_cfg['use_semantic'], use_panoptic=exp_cfg['use_panoptic'],
												without_ground=exp_cfg['without_ground'], loop_file=exp_cfg['loop_file'])
	elif dataset == 'kitti360':
		dataset_for_recall = KITTI3603DPoses(data, sequence,
											 train=False,
											 without_ground=exp_cfg['without_ground'], loop_file='loop_GT_4m_noneg')
	elif dataset == 'freiburg':
		dataset_for_recall = FreiburgDataset(data, without_ground=exp_cfg['without_ground'])
	elif dataset == 'nclt':
		dataset_for_recall = NCLTDataset(data, sequence)
	else:
		raise NotImplementedError

	# MapLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
	# 										batch_size=exp_cfg['batch_size'],
	# 										num_workers=2,
	# 										shuffle=False,
	# 										collate_fn=merge_inputs,
	# 										pin_memory=True)

	# model = get_model(exp_cfg, is_training=False)
	# renamed_dict = OrderedDict()
	# for key in saved_params['state_dict']:
	# 	if not key.startswith('module'):
	# 		renamed_dict = saved_params['state_dict']
	# 		break
	# 	else:
	# 		renamed_dict[key[7:]] = saved_params['state_dict'][key]
	#
	# res = model.load_state_dict(renamed_dict, strict=False)
	# if len(res[0]) > 0:
	# 	print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

	# model.train()
	# model = model.to(device)

	map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])

	# local_iter = 0.
	# transl_error_sum = 0
	# yaw_error_sum = 0
	# emb_list_map = []
	# rot_errors = []
	# transl_errors = []
	# time_descriptors = []
	# for i in range(num_iters):
	# 	rot_errors.append([])
	# 	transl_errors.append([])
	#
	# for batch_idx, sample in enumerate(tqdm(MapLoader)):
	#
	# 	model.eval()
	# 	time1 = time.time()
	# 	with torch.no_grad():
	#
	# 		anchor_list = []
	# 		for i in range(len(sample['anchor'])):
	# 			anchor = sample['anchor'][i].to(device)
	#
	# 			if exp_cfg['3D_net'] != 'PVRCNN':
	# 				anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
	# 				a = anchor_set[0, :].long()
	# 				anchor_i = anchor[a]
	# 			else:
	# 				anchor_i = anchor
	#
	# 			if exp_cfg['3D_net'] != 'PVRCNN':
	# 				anchor_list.append(anchor_i[:, :3].unsqueeze(0))
	# 			else:
	# 				if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
	# 					anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
	# 				anchor_list.append(model.backbone.prepare_input(anchor_i))
	# 				del anchor_i
	#
	# 		if exp_cfg['3D_net'] != 'PVRCNN':
	# 			anchor = torch.cat(anchor_list)
	# 			model_in = anchor
	# 			# Normalize between [-1, 1], more or less
	# 			# model_in = model_in / 100.
	# 			if exp_cfg['3D_net'] == 'RandLANet':
	# 				model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
	# 		else:
	# 			model_in = KittiDataset.collate_batch(anchor_list)
	# 			for key, val in model_in.items():
	# 				if not isinstance(val, np.ndarray):
	# 					continue
	# 				model_in[key] = torch.from_numpy(val).float().to(device)
	#
	# 		batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)
	#
	# 		emb = batch_dict['out_embedding']
	# 		# if exp_cfg['norm_embeddings']:
	# 		#     emb = emb / emb.norm(dim=1, keepdim=True)
	# 		emb_list_map.append(emb)
	#
	# 	time2 = time.time()
	# 	time_descriptors.append(time2-time1)

	# emb_list_map = torch.cat(emb_list_map).cpu().numpy()
	# # map_tree = KDTree(emb_list_map)

	# # Recall@k
	# recall = np.zeros(10)
	# total_frame = 0

	# emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
	# pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)
	# if save_path:
	# 	np.savez(save_path, pair_dist)


	pair_dist = np.load(npzfile)["arr_0"]

	poses = np.stack(dataset_for_recall.poses)
	precision_ours_fn, recall_ours_fn, precision_ours_fp, recall_ours_fp = compute_PR(pair_dist, poses, map_tree_poses)
	ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
	ap_ours_fn = compute_AP(precision_ours_fn, recall_ours_fn)
	# print(weights_path)
	print(exp_cfg['test_sequence'])
	print("AP FP: ", ap_ours_fp)
	print("AP FN: ", ap_ours_fn)
	precision_pair_ours, recall_pair_ours = compute_PR_pairs(pair_dist, poses)
	precision_pair_ours = [x for _,x in sorted(zip(recall_pair_ours, precision_pair_ours))]
	recall_pair_ours = sorted(recall_pair_ours)
	ap_ours_pair = compute_AP(precision_pair_ours, recall_pair_ours)
	print("AP Pairs: ", ap_ours_pair)


def main():
	PairFile = namedtuple("PairFile", ["weights", "dataset", "npz", "data", "sequence"])

	home_path = "/home/arceyd/MT/"
	cp_path = home_path + "cp/3D/"
	npz_path = home_path + "res/place_recognition/"

	kitti = "/data/arceyd/kitti/"
	kitti360 = home_path + "dat/kitti360"

	files = [
		# PairFile("16-09-2021_00-02-34/checkpoint_133_auc_0.859.tar", "kitti", "lcdnet_kitti_seq08.npz", kitti, "08"),
		# PairFile("04-04-2022_18-34-14/best_model_so_far_auc.tar"   , "kitti", "dcp_kitti_seq08.npz"   , kitti, "08"),
		# PairFile("12-05-2022_10-38-29/best_model_so_far_auc.tar"   , "kitti", "tf_kitti_seq08.npz"    , kitti, "08"),
		# PairFile("27-05-2022_19-10-54/best_model_so_far_auc.tar"   , "kitti", "padloc_kitti_seq08.npz", kitti, "08"),
		PairFile("16-09-2021_00-02-34/checkpoint_133_auc_0.859.tar", "kitti360", "lcdnet_kitti360_seq2013_05_28_drive_0002_sync.npz", kitti360, "2013_05_28_drive_0002_sync"),
		PairFile("04-04-2022_18-34-14/best_model_so_far_auc.tar"   , "kitti360", "dcp_kitti360_seq2013_05_28_drive_0002_sync.npz"   , kitti360, "2013_05_28_drive_0002_sync"),
		PairFile("12-05-2022_10-38-29/best_model_so_far_auc.tar"   , "kitti360", "tf_kitti360_seq2013_05_28_drive_0002_sync.npz"    , kitti360, "2013_05_28_drive_0002_sync"),
		PairFile("27-05-2022_19-10-54/best_model_so_far_auc.tar"   , "kitti360", "padloc_kitti360_seq2013_05_28_drive_0002_sync.npz", kitti360, "2013_05_28_drive_0002_sync"),
	]

	for file in files:
		print("\n"*2 + file.npz)
		main_process(cp_path + file.weights, file.dataset, npz_path + file.npz, file.data, file.sequence)


if __name__ == "__main__":
	main()

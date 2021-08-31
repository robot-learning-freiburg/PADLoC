import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from scipy.spatial import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.KITTI_data_loader import KITTILoader3DPoses
from datasets.NCLTDataset import NCLTDataset
from models.get_models import get_model
from utils.data import merge_inputs
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

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



def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    saved_params = torch.load(weights_path, map_location='cpu')
    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 2
    exp_cfg['PC_RANGE'] = [-70.4, -70.4, -0.5, 70.4, 70.4, 3.5]
    # exp_cfg['PC_RANGE'] = [-70.4, -70.4, -4, 70.4, 70.4, 0]

    current_date = datetime.now()

    if 'loop_file' not in exp_cfg:
        exp_cfg['loop_file'] = 'loop_GT'
    if 'sinkhorn_type' not in exp_cfg:
        exp_cfg['sinkhorn_type'] = 'unbalanced'

    dataset_map = NCLTDataset('/media/RAIDONE/DATASETS/NCLT/2012-01-08/')
    dataset_query_all = NCLTDataset('/media/RAIDONE/DATASETS/NCLT/2012-01-15/')
    with open('/media/RAIDONE/DATASETS/NCLT/2012-01-15/loops_on_2012-01-08.pickle', 'rb') as f:
        dict = pickle.load(f)

    test_pair_idxs = [elem['idx'] for elem in dict]
    dataset_query = torch.utils.data.Subset(dataset_query_all, test_pair_idxs)
    MapLoader = torch.utils.data.DataLoader(dataset=dataset_map,
                                            batch_size=exp_cfg['batch_size'],
                                            num_workers=2,
                                            shuffle=False,
                                            collate_fn=merge_inputs,
                                            pin_memory=True)
    QueryLoader = torch.utils.data.DataLoader(dataset=dataset_query,
                                              batch_size=exp_cfg['batch_size'],
                                              num_workers=2,
                                              shuffle=False,
                                              collate_fn=merge_inputs,
                                              pin_memory=True)

    model = get_model(exp_cfg)

    model.load_state_dict(saved_params['state_dict'], strict=False)

    model.train()
    model = model.to(device)

    local_iter = 0.
    transl_error_sum = 0
    yaw_error_sum = 0
    emb_list_map = []
    emb_list_query = []
    rot_errors = []
    transl_errors = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    for batch_idx, sample in tqdm(enumerate(MapLoader), total=len(dataset_map)//exp_cfg['batch_size']):

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
                model_in = model_in / 100.
            else:
                model_in = KittiDataset.collate_batch(anchor_list)
                for key, val in model_in.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False)

            emb = batch_dict['out_embedding']
            if exp_cfg['norm_embeddings']:
                emb = emb / emb.norm(dim=1, keepdim=True)
            emb_list_map.append(emb)
    for batch_idx, sample in tqdm(enumerate(QueryLoader), total=len(dataset_query)//exp_cfg['batch_size']):
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
                model_in = model_in / 100.
            else:
                model_in = KittiDataset.collate_batch(anchor_list)
                for key, val in model_in.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False)

            emb = batch_dict['out_embedding']
            if exp_cfg['norm_embeddings']:
                emb = emb / emb.norm(dim=1, keepdim=True)
            emb_list_query.append(emb)

    emb_list_map = torch.cat(emb_list_map).cpu().numpy()
    emb_list_query = torch.cat(emb_list_query).cpu().numpy()
    map_tree = KDTree(emb_list_map)
    recall = np.zeros(10)
    for i in range(emb_list_query.shape[0]):
        current_pose = dataset_query_all.poses[test_pair_idxs[i]][:3]
        current_pose = torch.tensor(current_pose)
        _, indices = map_tree.query(emb_list_query[i], k=10)
        for j in range(len(indices)):

            m = indices[j]
            x = dataset_map.poses[m][0]
            y = dataset_map.poses[m][1]
            z = dataset_map.poses[m][2]
            possible_match_pose = torch.tensor([x, y, z])
            distance = torch.norm(current_pose - possible_match_pose)
            if distance <= 1.5:
                recall[j] += 1
                break
    recall = np.cumsum(recall) / emb_list_query.shape[0]
    print(recall*100)

    # yaw_preds = yaw_preds*180/np.pi
    # yaw_preds = yaw_preds % 360
    # pred_error = pairwise_yaw[test_pair_idxs[:,0], test_pair_idxs[:,1]] - \
    #              yaw_preds[test_pair_idxs[:,0], test_pair_idxs[:,1]]
    # pred_error = pred_error.abs()
    # pred_error[pred_error>180] = 360 - pred_error[pred_error>180]
    # print(pred_error.mean())
    # with open(f'yaw_preds_{exp_cfg["test_sequence"]}_oreos.pickle', 'wb') as f:
    #     pickle.dump(yaw_preds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

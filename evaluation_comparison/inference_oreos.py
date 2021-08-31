import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.KITTI_data_loader import KITTILoader3DPoses
from models.get_models import get_model
from utils.data import merge_inputs
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy

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
    exp_cfg['batch_size'] = 4

    if 'loop_file' not in exp_cfg:
        exp_cfg['loop_file'] = 'loop_GT'
    if 'sinkhorn_type' not in exp_cfg:
        exp_cfg['sinkhorn_type'] = 'unbalanced'
    if 'shared_embeddings' not in exp_cfg:
        exp_cfg['shared_embeddings'] = True

    current_date = datetime.now()

    exp_cfg['test_sequence'] = "08"
    sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    sequences_validation = [exp_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)
    exp_cfg['sinkhorn_iter'] = 5

    dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
                                            os.path.join(args.data, 'sequences', sequences_validation[0], 'poses_SEMANTICKITTI.txt'),
                                            exp_cfg['num_points'], device, train=False,
                                            without_ground=exp_cfg['without_ground'], loop_file=exp_cfg['loop_file'])


    dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    final_dest = ''

    with open(f'/home/cattaneod/CODES/overlapnet_custom/GT/{exp_cfg["test_sequence"]}/GT2.pickle', 'rb') as f:
        gts = pickle.load(f)
    pairwise_overlap = np.zeros((len(gts), len(gts)))
    for i in range(len(gts)):
        pairwise_overlap[i, i:] = gts[i][i:, 4]
    pairwise_yaw = np.zeros((len(gts), len(gts)))
    for i in range(len(gts)):
        # pairwise_yaw[i, i:] = gts[i][i:, 3]
        pairwise_yaw[i, i:] = gts[i][i:, 7]
    # pairwise_yaw = (180 - pairwise_yaw) % 360
    # pairwise_yaw = torch.tensor(180 - pairwise_yaw) % 360
    pairwise_yaw = torch.tensor(pairwise_yaw * 180 / np.pi).float() % 360

    with open(f'/home/cattaneod/CODES/deep_lcd/oreos_pairs_{exp_cfg["test_sequence"]}.pickle', 'rb') as f:
        test_pair_idxs = pickle.load(f)['pairs']
    test_pair_idxs = test_pair_idxs[:, ::-1]
    # sampler = SamplePairs(dataset_for_recall, test_pair_idxs)
    batch_sampler = BatchSamplePairs(dataset_for_recall, test_pair_idxs, exp_cfg['batch_size'])
    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
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
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    # Testing
    if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
        # all_feats = []
        # all_coords = []
        # save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
        current_frame = 0
        yaw_preds = torch.zeros((len(gts), len(gts)))
        transl_errors = []
        for batch_idx, sample in enumerate(tqdm(RecallLoader)):

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
                    model_in = model_in / 100.
                else:
                    model_in = KittiDataset.collate_batch(anchor_list)
                    for key, val in model_in.items():
                        if not isinstance(val, np.ndarray):
                            continue
                        model_in[key] = torch.from_numpy(val).float().to(device)

                batch_dict = model(model_in, metric_head=True)

                transformation = batch_dict['transformation']
                homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                transformation = torch.cat((transformation, homogeneous), dim=1)
                transformation = transformation.inverse()
                for i in range(batch_dict['transformation'].shape[0]):
                    yaw_preds[test_pair_idxs[current_frame, 0], test_pair_idxs[current_frame, 1]] = mat2xyzrpy(transformation[i])[-1].item()
                    pose1 = dataset_for_recall.poses[test_pair_idxs[current_frame, 0]]
                    pose2 = dataset_for_recall.poses[test_pair_idxs[current_frame, 1]]
                    delta_pose = np.linalg.inv(pose1) @ pose2
                    transl_error = torch.tensor(delta_pose[:3, 3]) - transformation[i][:3, 3].detach().cpu()
                    transl_errors.append(transl_error.norm())
                    current_frame += 1

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

    yaw_preds = yaw_preds*180/np.pi
    yaw_preds = yaw_preds % 360
    pred_error = pairwise_yaw[test_pair_idxs[:,0], test_pair_idxs[:,1]] - \
                 yaw_preds[test_pair_idxs[:,0], test_pair_idxs[:,1]]
    pred_error = pred_error.abs()
    pred_error[pred_error>180] = 360 - pred_error[pred_error>180]
    print("Mean rotation error: ", pred_error.mean())
    print("STD rotation error: ", pred_error.std())
    transl_errors = torch.tensor(transl_errors)
    print("Mean translation error: ", transl_errors.mean())
    print("STD translation error: ", transl_errors.std())
    with open(f'yaw_preds_{exp_cfg["test_sequence"]}_oreos.pickle', 'wb') as f:
        pickle.dump(yaw_preds, f)


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

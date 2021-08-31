import argparse
import os
import pickle
import time

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from scipy.spatial import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.Freiburg import FreiburgRegistrationDataset, FreiburgRegistrationRPMDataset
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_RPMNet import KITTIRPM3DPoses, KITTIRPM3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from models.RPMNet.rpmnet import RPMNetEarlyFusion
from models.get_models import get_model
from utils.data import merge_inputs, Timer
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    saved_params = torch.load(weights_path, map_location='cpu')
    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 2

    if 'loop_file' not in exp_cfg:
        exp_cfg['loop_file'] = 'loop_GT'

    current_date = datetime.now()

    if args.dataset == 'kitti':
        exp_cfg['test_sequence'] = "00"
        sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    else:
        exp_cfg['test_sequence'] = "2013_05_28_drive_0002_sync"
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    sequences_validation = [exp_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)
    exp_cfg['sinkhorn_iter'] = 50

    dataset_for_recall = FreiburgRegistrationRPMDataset(args.data, without_ground=False)

    dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    final_dest = ''

    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               # batch_size=exp_cfg['batch_size'],
                                               num_workers=2,
                                               # sampler=sampler,
                                               # batch_sampler=batch_sampler,
                                               # worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    model = RPMNetEarlyFusion(exp_cfg['features'], exp_cfg['feature_dim'],
                              exp_cfg['radius'], exp_cfg['num_neighbors'])

    model.load_state_dict(saved_params['state_dict'], strict=True)

    # model.train()
    model = model.to(device)

    rot_errors = []
    transl_errors = []
    yaw_error = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    # Testing
    if True:
        # all_feats = []
        # all_coords = []
        # save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
        current_frame = 0
        yaw_preds = torch.zeros((25612, 25612))
        transl_errors = []
        time_rpm = Timer()
        for batch_idx, sample in enumerate(tqdm(RecallLoader)):
            if batch_idx == 1:
                time_rpm.reset()
            if batch_idx % 10 == 9:
                print("")
                print("Time: ", time_rpm.avg)

            start_time = time.time()

            ### AAA
            model.eval()
            with torch.no_grad():

                anchor_list = []
                positive_list = []
                for i in range(len(sample['anchor'])):
                    anchor = sample['anchor'][i].float().to(device)
                    positive = sample['positive'][i].float().to(device)

                    anchor_i = anchor
                    anchor_i[:, :3] = anchor_i[:, :3] / 100.

                    positive_i = positive
                    positive_i[:, :3] = positive_i[:, :3] / 100.

                    anchor_list.append(anchor_i.unsqueeze(0))
                    positive_list.append(positive_i.unsqueeze(0))

                anchor = torch.cat(anchor_list + positive_list)
                B = anchor.shape[0]
                model_in = {'points_ref': anchor[B//2:], 'points_src': anchor[:B//2]}

                # Normalize between [-1, 1], more or less

                torch.cuda.synchronize()
                time_rpm.tic()
                transforms, endpoints = model(model_in, 5)
                torch.cuda.synchronize()
                time_rpm.toc()

                transformation = transforms[-1]
                homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                transformation = torch.cat((transformation, homogeneous), dim=1)
                transformation = transformation.inverse()
                for i in range(transformation.shape[0]):
                    yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = mat2xyzrpy(transformation[i])[-1].item()
                    delta_pose = sample['transformation'][i]
                    transl_error = torch.tensor(delta_pose[:3, 3]) - transformation[i][:3, 3].detach().cpu()
                    transl_errors.append(transl_error.norm())

                    yaw_pred = yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]]
                    yaw_pred = yaw_pred % (2 * np.pi)
                    delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                    delta_yaw = delta_yaw % (2 * np.pi)
                    diff_yaw = abs(delta_yaw - yaw_pred)
                    diff_yaw = diff_yaw % (2 * np.pi)
                    diff_yaw = (diff_yaw * 180) / np.pi
                    if diff_yaw > 180.:
                        diff_yaw = 360 - diff_yaw
                    yaw_error.append(diff_yaw)

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
    # with open('results_for_paper/yaw_error_360-09_rpmnet.pickle', 'wb') as f:
    #     pickle.dump(yaw_error, f)
    # print('results_for_paper/yaw_error_360-09_rpmnet.pickle')
    # yaw_preds = yaw_preds*180/np.pi
    # yaw_preds = yaw_preds % 360
    # pred_error = pairwise_yaw[test_pair_idxs[:,0], test_pair_idxs[:,1]] - \
    #              yaw_preds[test_pair_idxs[:,0], test_pair_idxs[:,1]]
    # pred_error = pred_error.abs()
    # pred_error[pred_error>180] = 360 - pred_error[pred_error>180]
    yaw_error = np.array(yaw_error)
    print("Mean rotation error: ", yaw_error.mean())
    print("STD rotation error: ", yaw_error.std())
    transl_errors = torch.tensor(transl_errors)
    print("Mean translation error: ", transl_errors.mean())
    print("STD translation error: ", transl_errors.std())
    # with open(f'yaw_preds_{exp_cfg["test_sequence"]}_oreos.pickle', 'wb') as f:
    #     pickle.dump(yaw_preds, f)
    save_dict = {'rot': yaw_error, 'transl': transl_errors}
    save_path = f'./results_for_paper/rpmnet_{exp_cfg["test_sequence"]}'
    print(save_path)
    # with open(f'{save_path}.pickle', 'wb') as f:
    #     pickle.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

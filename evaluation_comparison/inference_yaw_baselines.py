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

import open3d as o3d
try:
    o3d_reg = o3d.pipelines.registration
except:
    o3d_reg = o3d.registration

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_errors(delta_pose, pred_transformation):

    yaw_pred = mat2xyzrpy(pred_transformation)[-1].item()
    pred_transl = pred_transformation[:3, 3].detach().cpu()

    transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl

    yaw_pred = yaw_pred % (2 * np.pi)
    delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
    delta_yaw = delta_yaw % (2 * np.pi)
    diff_yaw = abs(delta_yaw - yaw_pred)
    diff_yaw = diff_yaw % (2 * np.pi)
    diff_yaw = (diff_yaw * 180) / np.pi
    if diff_yaw > 180.:
        diff_yaw = 360 - diff_yaw
    return transl_error.norm(), diff_yaw


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



def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    saved_params = torch.load(weights_path, map_location='cpu')
    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 4

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

    current_date = datetime.now()

    if args.dataset == 'kitti':
        exp_cfg['test_sequence'] = "08"
        sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    else:
        exp_cfg['test_sequence'] = "2013_05_28_drive_0009_sync"
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    sequences_validation = [exp_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)
    exp_cfg['sinkhorn_iter'] = 50

    if args.dataset == 'kitti':
        dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
                                                os.path.join(args.data, 'sequences', sequences_validation[0],'poses_SEMANTICKITTI.txt'),
                                                exp_cfg['num_points'], 'cpu', train=False,
                                                without_ground=exp_cfg['without_ground'], loop_file=exp_cfg['loop_file'])
    else:
        dataset_for_recall = KITTI3603DPoses(args.data, sequences_validation[0],
                                             train=False,
                                             without_ground=exp_cfg['without_ground'], loop_file='loop_GT_4m_noneg')


    dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    final_dest = ''

    MapLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                            batch_size=exp_cfg['batch_size'],
                                            num_workers=2,
                                            shuffle=False,
                                            collate_fn=merge_inputs,
                                            pin_memory=True)
    map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])
    test_pair_idxs = []
    index = faiss.IndexFlatL2(3)
    poses = np.stack(dataset_for_recall.poses).copy()
    index.add(poses[:50, :3, 3].copy())
    num_frames_with_loop = 0
    num_frames_with_reverse_loop = 0
    for i in tqdm(range(100, len(dataset_for_recall.poses))):
        current_pose = poses[i:i+1, :3, 3].copy()
        index.add(poses[i-50:i-49, :3, 3].copy())
        lims, D, I = index.range_search(current_pose, 4.**2)
        for j in range(lims[0], lims[1]):
            if j == 0:
                num_frames_with_loop += 1
                yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
                yaw_diff = yaw_diff % (2 * np.pi)
                if 0.79 <= yaw_diff <= 5.5:
                    num_frames_with_reverse_loop += 1
                # else:
                #     print(yaw_diff)
            test_pair_idxs.append([I[j], i])
    test_pair_idxs = np.array(test_pair_idxs)

    batch_sampler = BatchSamplePairs(dataset_for_recall, test_pair_idxs, exp_cfg['batch_size'])
    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               # batch_size=exp_cfg['batch_size'],
                                               num_workers=2,
                                               # sampler=sampler,
                                               batch_sampler=batch_sampler,
                                               # worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    yaw_error = []
    yaw_error_p2p = []
    yaw_error_p2pl = []
    yaw_error_fgr = []
    yaw_error_fpfh = []
    transl_errors = []
    transl_errors_p2p = []
    transl_errors_p2pl = []
    transl_errors_fgr = []
    transl_errors_fpfh = []

    time_p2p, time_p2pl, time_fpfh, time_fgr = Timer(), Timer(), Timer(), Timer()
    time_normal, time_feature = Timer(), Timer()

    # Testing
    if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
        current_frame = 0

        for batch_idx, sample in enumerate(tqdm(RecallLoader)):
            if batch_idx % 10 == 9:
                print("")
                print(f'Time P2P: {time_p2p.avg}')
                print(f'Time Compute Normal: {time_normal.avg}')
                print(f'Time P2Pl: {time_p2pl.avg}')
                print(f'Time Compute FPFH: {time_feature.avg}')
                print(f'Time register FPFH: {time_fpfh.avg}')
                print(f'Time FGR: {time_fgr.avg}')

            start_time = time.time()

            ### AAA
            for i in range(len(sample['anchor']) // 2):

                pose1 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 0]]
                pose2 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 1]]
                delta_pose = np.linalg.inv(pose1) @ pose2

                coords1 = sample['anchor'][i]
                coords2 = sample['anchor'][i + len(sample['anchor']) // 2]
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(coords1[:, 1:].cpu().numpy())
                pcd1 = pcd1.voxel_down_sample(args.voxel_size)
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(coords2[:, 1:].cpu().numpy())
                pcd2 = pcd2.voxel_down_sample(args.voxel_size)
                time_normal.tic()
                pcd1.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2, max_nn=30))
                pcd2.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*2, max_nn=30))
                time_normal.toc()

                # ICP-p2p
                time_p2p.tic()
                reg_p2p = o3d_reg.registration_icp(
                    pcd2, pcd1, args.voxel_size*2,
                    estimation_method=o3d_reg.TransformationEstimationPointToPoint())
                time_p2p.toc()
                transformation = torch.tensor(reg_p2p.transformation)
                tr_err, rot_err = get_errors(delta_pose, transformation)
                transl_errors_p2p.append(tr_err)
                yaw_error_p2p.append(rot_err)

                # ICP-p2pl
                time_p2pl.tic()
                reg_p2pl = o3d_reg.registration_icp(
                    pcd2, pcd1, args.voxel_size*2,
                    estimation_method=o3d_reg.TransformationEstimationPointToPlane())
                time_p2pl.toc()
                transformation = torch.tensor(reg_p2pl.transformation)
                tr_err, rot_err = get_errors(delta_pose, transformation)
                transl_errors_p2pl.append(tr_err)
                yaw_error_p2pl.append(rot_err)

                # FPFH
                time_feature.tic()
                fpfh1 = o3d_reg.compute_fpfh_feature(
                    pcd1,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*5, max_nn=100))
                fpfh2 = o3d_reg.compute_fpfh_feature(
                    pcd2,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size*5, max_nn=100))
                time_feature.toc()
                time_fpfh.tic()
                reg_fpfh = o3d_reg.registration_ransac_based_on_feature_matching(
                    pcd2, pcd1, fpfh2, fpfh1, True,
                    args.voxel_size*2,
                    o3d_reg.TransformationEstimationPointToPoint(False),
                    3, [
                        o3d_reg.CorrespondenceCheckerBasedOnDistance(
                            args.voxel_size*2)
                    ], o3d_reg.RANSACConvergenceCriteria(100000, 1000))
                time_fpfh.toc()
                transformation = torch.tensor(reg_fpfh.transformation)
                tr_err, rot_err = get_errors(delta_pose, transformation)
                transl_errors_fpfh.append(tr_err)
                yaw_error_fpfh.append(rot_err)

                #FGR
                time_fgr.tic()
                reg_fgr = o3d_reg.registration_fast_based_on_feature_matching(
                    pcd2, pcd1, fpfh2, fpfh1,
                    o3d_reg.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=args.voxel_size*2))
                time_fgr.toc()
                transformation = torch.tensor(reg_fgr.transformation)
                tr_err, rot_err = get_errors(delta_pose, transformation)
                transl_errors_fgr.append(tr_err)
                yaw_error_fgr.append(rot_err)

            current_frame += len(sample['anchor']) // 2

    print(weights_path)
    print(exp_cfg['test_sequence'])

    save_dict = {}
    for method in ['p2p', 'p2pl', 'fpfh', 'fgr']:
        print(f"### {method} ###")
        yaw_error = eval(f'yaw_error_{method}')
        transl_errors = eval(f'transl_errors_{method}')
        transl_errors = torch.tensor(transl_errors)
        yaw_error = np.array(yaw_error)
        print("Mean rotation error: ", yaw_error.mean())
        print("Median rotation error: ", torch.tensor(yaw_error).median())
        print("STD rotation error: ", yaw_error.std())
        transl_errors = torch.tensor(transl_errors)
        print("Mean translation error: ", transl_errors.mean())
        print("Median translation error: ", transl_errors.median())
        print("STD translation error: ", transl_errors.std())
        save_dict[method] = {'rot': yaw_error, 'transl': transl_errors}
    # with open(f'baseline_{args.voxel_size}_{exp_cfg["test_sequence"]}.pickle', 'wb') as f:
    #     pickle.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--ransac', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=float, default=0.3)
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

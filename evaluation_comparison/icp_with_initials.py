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

from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses
from models.get_models import get_model
from utils.data import merge_inputs
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT
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


def icp_sks(source, target, T_initial):
    import sksurgerypclpython as sks
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source[:, :3].copy())
    source_pcd.transform(T_initial)
    a = np.asarray(source_pcd.points)
    b = target[:, :3].copy().astype(np.float64)
    result = np.eye(4)
    transformed_source_points = a.copy()
    time1 = time.time()
    residual = sks.icp(a,
                       b,
                       1000,                        # Number of iterations
                       200.,         # Max correspondence distance, so sys.float_info.max means "unused" in this test
                       1e-6,                    # Transformation epsilon
                       1e-6,                    # Cost function epsilon
                       False,                      # Use LM-ICP
                       result,                     # Output 4x4
                       transformed_source_points)  # Output transformed points
    time2 = time.time()
    del a, b, result, transformed_source_points, source_pcd
    return residual, time2 - time1


def run_icp(source, target, T_initial):
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source[:, :3].copy())
    target_pcd.points = o3d.utility.Vector3dVector(target[:, :3].copy())
    time1 = time.time()
    reg_p2p = reg_module.registration_icp(
        source_pcd, target_pcd, 200., T_initial,
        reg_module.TransformationEstimationPointToPoint(),
        reg_module.ICPConvergenceCriteria(max_iteration=1000))
    time2 = time.time()
    return reg_p2p.inlier_rmse, time2 - time1

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
        save_pickle = f'results_for_paper/icp_{exp_cfg["test_sequence"]}_ransac.pickle'
    else:
        exp_cfg['test_sequence'] = "2013_05_28_drive_0009_sync"
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
        save_pickle = 'results_for_paper/icp_360-09_ransac.pickle'
    sequences_validation = [exp_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)
    exp_cfg['sinkhorn_iter'] = 50

    if args.dataset == 'kitti':
        dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
                                                os.path.join(args.data, 'sequences', sequences_validation[0],'poses_SEMANTICKITTI.txt'),
                                                exp_cfg['num_points'], device, train=False,
                                                without_ground=exp_cfg['without_ground'], loop_file=exp_cfg['loop_file'])
        # pd = np.load(f'pairs_dist_ours_{exp_cfg["test_sequence"]}.npz')['arr_0']
    else:
        dataset_for_recall = KITTI3603DPoses(args.data, sequences_validation[0],
                                             train=False,
                                             without_ground=False, loop_file='loop_GT_4m_noneg')
        # pd = np.load(f'pairs_dist_ours_360-02_trained.npz')['arr_0']

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
        if lims[1] > 0:
            # test_pair_idxs.append([pd[i,:i-50].argmin(), i])
            test_pair_idxs.append([I[0], i])
        # for j in range(lims[0], lims[1]):
        #     if j == 0:
        #         num_frames_with_loop += 1
        #         yaw_diff = RT.npto_XYZRPY(np.linalg.inv(poses[I[j]]) @ poses[i])[-1]
        #         yaw_diff = yaw_diff % (2 * np.pi)
        #         if 0.79 <= yaw_diff <= 5.5:
        #             num_frames_with_reverse_loop += 1
        #         # else:
        #         #     print(yaw_diff)
        #     test_pair_idxs.append([I[j], i])
    random.shuffle(test_pair_idxs)
    test_pair_idxs = np.array(test_pair_idxs)

    times_without_init = []
    times_with_init = []
    rmse_without_init = []
    rmse_with_init = []
    # if os.path.exists(save_pickle):
    #     with open(save_pickle, 'rb') as f:
    #         save_dict = pickle.load(f)
    #         times_without_init = list(save_dict['times_without_init'])
    #         times_with_init = list(save_dict['times_with_init'])
    #         rmse_without_init = list(save_dict['rmse_without_init'])
    #         rmse_with_init = list(save_dict['rmse_with_init'])
    #     test_pair_idxs = test_pair_idxs[len(times_without_init):]

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
    yaw_error = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    # Testing
    if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
        # all_feats = []
        # all_coords = []
        # save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
        current_frame = 0
        yaw_preds = torch.zeros((len(dataset_for_recall.poses), len(dataset_for_recall.poses)))
        transl_errors = []
        for batch_idx, sample in enumerate(tqdm(RecallLoader)):
            # if batch_idx == 50:
            #     break

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
                coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
                feats = batch_dict['point_features'].squeeze(-1)
                for i in range(batch_dict['transformation'].shape[0]):
                    yaw_preds[test_pair_idxs[current_frame, 0], test_pair_idxs[current_frame, 1]] = mat2xyzrpy(transformation[i])[-1].item()
                    pose1 = dataset_for_recall.poses[test_pair_idxs[current_frame, 0]]
                    pose2 = dataset_for_recall.poses[test_pair_idxs[current_frame, 1]]
                    delta_pose = np.linalg.inv(pose1) @ pose2
                    transl_error = torch.tensor(delta_pose[:3, 3]) - transformation[i][:3, 3].detach().cpu()
                    transl_errors.append(transl_error.norm())

                    yaw_pred = yaw_preds[test_pair_idxs[current_frame, 0], test_pair_idxs[current_frame, 1]]
                    yaw_pred = yaw_pred % (2 * np.pi)
                    delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                    delta_yaw = delta_yaw % (2 * np.pi)
                    diff_yaw = abs(delta_yaw - yaw_pred)
                    diff_yaw = diff_yaw % (2 * np.pi)
                    diff_yaw = (diff_yaw * 180) / np.pi
                    if diff_yaw > 180.:
                        diff_yaw = 360 - diff_yaw
                    yaw_error.append(diff_yaw)

                    # RANSAC
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

                    # reg1, time1 = icp_sks(sample['anchor'][i].numpy(),
                    #                       sample['anchor'][i+exp_cfg['batch_size']//2].numpy(),
                    #                       np.eye(4))
                    # reg2, time2 = icp_sks(sample['anchor'][i].numpy(),
                    #                       sample['anchor'][i+exp_cfg['batch_size']//2].numpy(),
                    #                       transformation[i].inverse().detach().cpu().numpy())
                    reg2, time2 = run_icp(sample['anchor'][i].numpy(),
                                          sample['anchor'][i+exp_cfg['batch_size']//2].numpy(),
                                          np.linalg.inv(result.transformation))
                    # _, _ = icp_sks(sample['anchor'][i].numpy(),
                    #                sample['anchor'][i+exp_cfg['batch_size']//2].numpy(),
                    #                delta_pose.inverse().detach().cpu().numpy())
                    # times_without_init.append(time1)
                    times_with_init.append(time2)
                    # rmse_without_init.append(reg1)
                    rmse_with_init.append(reg2)

                    current_frame += 1
                del sample, transformation

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
            with open(save_pickle, 'wb') as f:
                save_dict = {
                    # 'times_without_init': np.array(times_without_init),
                    'times_with_init': np.array(times_with_init),
                    # 'rmse_without_init': np.array(rmse_without_init),
                    'rmse_with_init': np.array(rmse_with_init)}
                pickle.dump(save_dict, f)
    # print(save_pickle)
    # print("Mean icp time without initial: ", np.array(times_without_init).mean())
    # print("Mean icp time with initial: ", np.array(times_with_init).mean())
    # print("STD icp time without initial: ", np.array(times_without_init).std())
    # print("STD icp time with initial: ", np.array(times_with_init).std())
    # print("Mean rmse error without initial: ", np.array(rmse_without_init).mean())
    # print("Mean rmse error with initial: ", np.array(rmse_with_init).mean())
    # print("STD rmse error without initial: ", np.array(rmse_without_init).std())
    # print("STD rmse error with initial: ", np.array(rmse_with_init).std())
    # with open(f'yaw_preds_{exp_cfg["test_sequence"]}_oreos.pickle', 'wb') as f:
    #     pickle.dump(yaw_preds, f)
    print("Done")


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

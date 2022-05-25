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
from scipy.spatial.transform import Rotation
from scipy.stats import circmean, circstd
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


def rot2aa(rotation):
    if isinstance(rotation, torch.Tensor):
        rotation = rotation.numpy()

    r = Rotation.from_matrix(rotation)
    v = r.as_rotvec()

    angle = np.linalg.norm(v)
    if angle < 1e-5:
        # Any arbitrary axis is ok, since the rotation is pretty much 0
        axis = np.array([1., 0, 0])
    else:
        axis = v / angle

    return axis, angle


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
    exp_cfg['sinkhorn_iter'] = 5

    if args.dataset == 'kitti':
        dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
                                                os.path.join(args.data, 'sequences',
                                                             sequences_validation[0], 'poses.txt'),
                                                exp_cfg['num_points'], device, train=False,
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

    time_net, time_ransac, time_icp = Timer(), Timer(), Timer()

    # Testing
    if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
        # all_feats = []
        # all_coords = []
        # save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
        current_frame = 0
        yaw_preds = torch.zeros((len(dataset_for_recall.poses), len(dataset_for_recall.poses)))
        transl_errors = []
        rot_errors = []
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
                pred_transf = []
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
                elif exp_cfg['rot_representation'].startswith('6dof') and not args.ransac:
                    transformation = batch_dict['transformation']
                    homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                    transformation = torch.cat((transformation, homogeneous), dim=1)
                    transformation = transformation.inverse()
                    for i in range(batch_dict['batch_size'] // 2):
                        yaw_preds[test_pair_idxs[current_frame+i, 0], test_pair_idxs[current_frame+i, 1]] = mat2xyzrpy(transformation[i])[-1].item()
                        pred_transf.append(transformation[i].inverse().detach().cpu())
                        pred_transl.append(transformation[i][:3, 3].detach().cpu())
                elif args.ransac:
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
                        if args.icp:
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
                        pred_transf.append(transformation.detach().cpu())
                        pred_transl.append(transformation[:3, 3].detach().cpu())
                for i in range(batch_dict['batch_size'] // 2):
                    pose1 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 0]]
                    pose2 = dataset_for_recall.poses[test_pair_idxs[current_frame+i, 1]]
                    delta_pose = np.linalg.inv(pose1) @ pose2
                    transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl[i]
                    transl_errors.append(transl_error.norm())
                    rel_pose_err = np.matmul(delta_pose, pred_transf[i].numpy())
                    rel_pose_tra_err = rel_pose_err[:3, 3]  # Just for comparison with transl_error
                    rel_pose_rot_err_mat = rel_pose_err[:3, :3]
                    rel_pose_rot_err_ax, rel_pos_rot_err_ang = rot2aa(rel_pose_rot_err_mat)
                    rel_pose_rot_err_ang = np.abs(rel_pos_rot_err_ang) % (2 * np.pi)
                    rot_errors.append(rel_pose_rot_err_ang)

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
    rot_errors = np.array(rot_errors)

    yaw_mean = yaw_error.mean()
    yaw_median = np.median(yaw_error)
    yaw_std = yaw_error.std()
    tra_mean = transl_errors.mean()
    tra_median = np.median(transl_errors)
    tra_std = transl_errors.std()
    rot_mean = circmean(rot_errors) * 180. / np.pi
    rot_std = circstd(rot_errors) * 180. / np.pi
    print("Mean yaw error: ", yaw_mean)
    print("Median yaw error: ", yaw_median)
    print("STD yaw error: ", yaw_std)
    print("Mean rotation error: ", rot_mean)
    print("STD rotation error: ", rot_std)
    print("Mean translation error: ", tra_mean)
    print("Median translation error: ", tra_median)
    print("STD translation error: ", tra_std)

    valid = yaw_error <= 5.
    valid = valid & (np.array(transl_errors) <= 2.)
    succ_rate = valid.sum() / valid.shape[0]
    rte_suc = transl_errors[valid].mean()
    rre_suc = yaw_error[valid].mean()

    print(f"Success Rate: {succ_rate}, RTE: {rte_suc}, RRE: {rre_suc}")

    save_dict = {
        "yaw": yaw_error,
        "yaw_mean": yaw_mean,
        "yaw_median": yaw_median,
        "yaw_std": yaw_std,
        "rot": rot_errors,
        "rot_mean": rot_mean,
        "rot_std": rot_std,
        "tra": transl_errors,
        "tra_mean": tra_mean,
        "tra_std": tra_std,
        "tra_median": tra_median,
        "success_rate": succ_rate,
        "RTE": rte_suc,
        "RRE": rre_suc
    }

    if args.save:
        # save_path = f'./evaluation_results/lcdnet00+08_{exp_cfg["test_sequence"]}'
        save_path = args.save
        # if  '360' in weights_path:
        #     save_path = f'./results_for_paper/lcdnet++_{exp_cfg["test_sequence"]}'
        # else:
        #     save_path = f'./results_for_paper/lcdnet_{exp_cfg["test_sequence"]}'
        # if args.icp:
        #     save_path = save_path+'_icp'
        # elif args.ransac:
        #     save_path = save_path+'_ransac'

        print("Saving to ", save_path)
        with open(f'{save_path}.pickle', 'wb') as f:
            pickle.dump(save_dict, f)

    return save_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--ransac', action='store_true', default=False)
    parser.add_argument('--icp', action='store_true', default=False)
    parser.add_argument('--save', default="")
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

import argparse
import pickle
import time
from typing import Callable, List, Optional

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.FordCampus import FordCampusDataset
from models.get_models import load_model
from utils.data import merge_inputs
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT
from utils.tools import set_seed

import open3d as o3d
if hasattr(o3d, 'pipelines'):
    reg_module = o3d.pipelines.registration
else:
    reg_module = o3d.registration

torch.backends.cudnn.benchmark = True

EPOCH = 1


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


def ransac_registration(*,
                        anc_coords: torch.Tensor,
                        anc_feats: torch.Tensor,
                        pos_coords: torch.Tensor,
                        pos_feats: torch.Tensor,
                        initial_transformation: Optional[torch.Tensor] = None,
                        ) -> torch.Tensor:

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anc_coords.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pos_coords.cpu().numpy())
    pcd1_feat = reg_module.Feature()
    pcd1_feat.data = anc_feats.permute(0, 1).cpu().numpy()
    pcd2_feat = reg_module.Feature()
    pcd2_feat.data = pos_feats.permute(0, 1).cpu().numpy()

    torch.cuda.synchronize()
    # time_ransac.tic()
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

    # time_ransac.toc()
    transformation = torch.tensor(result.transformation.copy())
    return transformation


def icp_registration(*,
                     anc_coordinates: torch.Tensor,
                     pos_coordinates: torch.Tensor,
                     initial_transformation: torch.Tensor
                     ) -> torch.Tensor:

    p1 = o3d.geometry.PointCloud()
    p1.points = o3d.utility.Vector3dVector(anc_coordinates.cpu().numpy())
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(pos_coordinates.cpu().numpy())

    # time_icp.tic()
    result = reg_module.registration_icp(
        p2, p1, 0.1, initial_transformation.cpu().numpy(),
        reg_module.TransformationEstimationPointToPoint())
    # time_icp.toc()

    transformation = torch.tensor(result.transformation.copy())

    return transformation


def batch_coord_feat_registration(*,
                                  reg_func: Callable,
                                  batch_coords: torch.Tensor,
                                  batch_feats: torch.Tensor,
                                  batch_size: int,
                                  initial_transformations: Optional[torch.Tensor] = None,
                                  ) -> torch.Tensor:
    transformations = []
    for i in range(batch_size // 2):
        coords1 = batch_coords[i]
        coords2 = batch_coords[i + batch_size // 2]
        feat1 = batch_feats[i]
        feat2 = batch_feats[i + batch_size // 2]

        if initial_transformations is not None:
            initial_transformation = initial_transformations[i]
        else:
            initial_transformation = None

        transformation = reg_func(anc_coords=coords1[:, 1:], anc_feats=feat1,
                                  pos_coords=coords2[:, 1:], pos_feats=feat2,
                                  initial_transformation=initial_transformation)

        transformations.append(transformation)
    return torch.stack(transformations)


def batch_coord_registration(*,
                             reg_func: Callable,
                             batch_coords: torch.Tensor,
                             batch_size: int,
                             initial_transformations: Optional[torch.Tensor] = None,
                             ) -> torch.Tensor:
    transformations = []
    for i in range(batch_size // 2):
        coords1 = batch_coords[i]
        coords2 = batch_coords[i + batch_size // 2]

        if initial_transformations is not None:
            initial_transformation = initial_transformations[i]
        else:
            initial_transformation = None

        transformation = reg_func(anc_coords=coords1[:, 1:], pos_coords=coords2[:, 1:],
                                  initial_transformation=initial_transformation)

        transformations.append(transformation)
    return torch.stack(transformations)


def batch_ransac_registration(*,
                              batch_coords: torch.Tensor,
                              batch_feats: torch.Tensor,
                              batch_size: int,
                              initial_transformations: Optional[torch.Tensor] = None,
                              ) -> torch.Tensor:
    return batch_coord_feat_registration(reg_func=ransac_registration,
                                         batch_coords=batch_coords, batch_feats=batch_feats,
                                         batch_size=batch_size,
                                         initial_transformations=initial_transformations,
                                         )

def batch_icp_registration(*,
                           batch_coords: torch.Tensor,
                           batch_size: int,
                           initial_transformations: Optional[torch.Tensor] = None,
                           ) -> torch.Tensor:
    return batch_coord_registration(reg_func=icp_registration, batch_coords=batch_coords, batch_size=batch_size,
                                    initial_transformations=initial_transformations)


def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    set_seed(args.seed)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # saved_params = torch.load(weights_path, map_location='cpu')
    # exp_cfg = saved_params['config']
    override_cfg = dict(
        batch_size=args.batch_size,
    )

    # if 'loop_file' not in exp_cfg:
    #     exp_cfg['loop_file'] = 'loop_GT'
    # if 'sinkhorn_type' not in exp_cfg:
    #     exp_cfg['sinkhorn_type'] = 'flot'
    # if 'shared_embeddings' not in exp_cfg:
    #     exp_cfg['shared_embeddings'] = False
    # if 'use_semantic' not in exp_cfg:
    #     exp_cfg['use_semantic'] = False
    # if 'use_panoptic' not in exp_cfg:
    #     exp_cfg['use_panoptic'] = False
    # if 'noneg' in exp_cfg['loop_file']:
    #     exp_cfg['loop_file'] = 'loop_GT_4m'

    # current_date = datetime.now()
    #
    # if args.dataset == 'kitti':
    #     exp_cfg['test_sequence'] = "00"
    #     sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    # else:
    #     exp_cfg['test_sequence'] = "2013_05_28_drive_0009_sync"
    #     sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
    #                           "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
    #                           "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    # sequences_validation = [exp_cfg['test_sequence']]
    # sequences_training = set(sequences_training) - set(sequences_validation)
    # sequences_training = list(sequences_training)
    # exp_cfg['sinkhorn_iter'] = 5

    dataset_for_recall = FordCampusDataset(args.data, args.seq, without_ground=False)


    # dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    # final_dest = ''
    #
    # MapLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
    #                                         batch_size=exp_cfg['batch_size'],
    #                                         num_workers=2,
    #                                         shuffle=False,
    #                                         collate_fn=merge_inputs,
    #                                         pin_memory=True)
    # map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])
    test_pair_idxs = []
    num_frames_with_loop = 0
    index = faiss.IndexFlatL2(3)
    poses = np.stack(dataset_for_recall.poses).astype(np.float32).copy()
    index.add(poses[:1, :3, 3].copy())
    print("Computing list of Pairs")
    for i in tqdm(range(1001, len(dataset_for_recall.poses))):
        current_pose = poses[i:i + 1, :3, 3].copy()
        index.add(poses[i - 1000:i - 999, :3, 3].copy())
        lims, D, I = index.range_search(current_pose, args.positive_distance ** 2)
        for j in range(lims[0], lims[1]):
            if j == 0:
                num_frames_with_loop += 1
            test_pair_idxs.append([I[j], i])
    test_pair_idxs = np.array(test_pair_idxs)

    batch_sampler = BatchSamplePairs(dataset_for_recall, test_pair_idxs, args.batch_size)
    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               # batch_size=exp_cfg['batch_size'],
                                               num_workers=2,
                                               # sampler=sampler,
                                               batch_sampler=batch_sampler,
                                               # worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    # model = get_model(exp_cfg)
    #
    # model.load_state_dict(saved_params['state_dict'], strict=True)

    # model.train()

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg, is_training=args.non_deterministic)

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
    print("Testing registration")
    if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
        # all_feats = []
        # all_coords = []
        # save_folder = '/media/RAIDONE/CATTANEOD/LCD_FEATS/00/'
        current_frame = 0
        yaw_preds = torch.zeros((len(dataset_for_recall.poses), len(dataset_for_recall.poses)))
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

                if exp_cfg['rot_representation'].startswith('6dof') and not args.ransac:
                    transformation = batch_dict['transformation']
                    homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                    transformation = torch.cat((transformation, homogeneous), dim=1)
                    transformation = transformation.inverse()

                elif args.ransac:
                    coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
                    feats = batch_dict['point_features'].squeeze(-1)
                    transformation = batch_ransac_registration(batch_coords=coords, batch_feats=feats,
                                                               batch_size=batch_dict["batch_size"])

                if args.icp:
                    coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
                    transformation = batch_icp_registration(batch_coords=coords, batch_size=batch_dict["batch_size"],
                                                            initial_transformations=transformation)

                for i in range(transformation.shape[0]):
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
                    diff_yaw = diff_yaw * 180 / np.pi
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
    # with open('temp2.pickle', 'wb') as f:
    #     pickle.dump(yaw_error, f)
    # # yaw_preds = yaw_preds*180/np.pi
    # # yaw_preds = yaw_preds % 360
    # # pred_error = pairwise_yaw[test_pair_idxs[:,0], test_pair_idxs[:,1]] - \
    # #              yaw_preds[test_pair_idxs[:,0], test_pair_idxs[:,1]]
    # # pred_error = pred_error.abs()
    # # pred_error[pred_error>180] = 360 - pred_error[pred_error>180]
    # print("Mean rotation error: ", np.array(yaw_error).mean())
    # print("STD rotation error: ", np.array(yaw_error).std())
    # transl_errors = torch.tensor(transl_errors)
    # print("Mean translation error: ", transl_errors.mean())
    # print("STD translation error: ", transl_errors.std())
    # # with open(f'yaw_preds_{exp_cfg["test_sequence"]}_oreos.pickle', 'wb') as f:
    # #     pickle.dump(yaw_preds, f)

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

    print(f"Success Rate: {succ_rate}, RTE: {rte_suc}, RRE: {rre_suc}")

    if args.save_path:
        print("Saving to ", args.save_path)
        with open(args.save_path, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/arceyd/MT/dat/Ford Campus/',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/arceyd/MT/cp/3D/')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--seq', type=str, default='1')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--ransac', action='store_true', default=False)
    parser.add_argument('--icp', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--positive_distance", type=float, default=10.)
    parser.add_argument("--non_deterministic", action="store_true")
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

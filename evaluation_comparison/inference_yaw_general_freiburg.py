import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from tqdm import tqdm

from datasets.Freiburg import FreiburgRegistrationDataset
from models.get_models import load_model
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from utils.data import merge_inputs, Timer
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as RT
from utils.qcqp_layer import QuadQuatFastSolver
from utils.tools import set_seed

import open3d as o3d
if hasattr(o3d, 'pipelines'):
    reg_module = o3d.pipelines.registration
else:
    reg_module = o3d.registration

torch.backends.cudnn.benchmark = True

EPOCH = 1


def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    set_seed(args.seed)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # saved_params = torch.load(weights_path, map_location='cpu')
    # exp_cfg = saved_params['config']
    # exp_cfg['batch_size'] = 2
    #
    # if 'loop_file' not in exp_cfg:
    #     exp_cfg['loop_file'] = 'loop_GT'
    # if 'sinkhorn_type' not in exp_cfg:
    #     exp_cfg['sinkhorn_type'] = 'unbalanced'
    # if 'shared_embeddings' not in exp_cfg:
    #     exp_cfg['shared_embeddings'] = False
    # if 'use_semantic' not in exp_cfg:
    #     exp_cfg['use_semantic'] = False
    # if 'use_panoptic' not in exp_cfg:
    #     exp_cfg['use_panoptic'] = False
    # if 'noneg' in exp_cfg['loop_file']:
    #     exp_cfg['loop_file'] = 'loop_GT_4m'
    # if 'head' not in exp_cfg:
    #     exp_cfg['head'] = 'SuperGlue'
    #
    # current_date = datetime.now()

    override_cfg = dict(
        batch_size=args.batch_size,
    )

    if args.dataset == 'kitti':
        override_cfg['test_sequence'] = "08"
        sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    else:
        override_cfg['test_sequence'] = "2013_05_28_drive_0009_sync"
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    sequences_validation = [override_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)
    override_cfg['sinkhorn_iter'] = 5

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg, is_training=args.non_deterministic)

    dataset_for_recall = FreiburgRegistrationDataset(args.data, without_ground=exp_cfg['without_ground'],
                                                     z_offset=args.z_offset)

    # dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               # batch_size=exp_cfg['batch_size'],
                                               num_workers=2,
                                               # sampler=sampler,
                                               # worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    # model = get_model(exp_cfg)

    # model.load_state_dict(saved_params['state_dict'], strict=True)

    # model.train()
    model = model.to(device)

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
        yaw_preds = torch.zeros((25612, 25612))
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
                positive_list = []
                for i in range(len(sample['anchor'])):
                    anchor = sample['anchor'][i].to(device)
                    positive = sample['positive'][i].to(device)

                    if exp_cfg['3D_net'] != 'PVRCNN':
                        anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                        a = anchor_set[0, :].long()
                        anchor_i = anchor[a]
                        positive_set = furthest_point_sample(positive[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                        p = positive_set[0, :].long()
                        positive_i = positive[p]
                    else:
                        anchor_i = anchor
                        positive_i = positive

                    if exp_cfg['3D_net'] != 'PVRCNN':
                        anchor_list.append(anchor_i[:, :3].unsqueeze(0))
                        positive_list.append(positive_i[:, :3].unsqueeze(0))
                    else:
                        anchor_list.append(model.backbone.prepare_input(anchor_i))
                        positive_list.append(model.backbone.prepare_input(positive_i))
                        del anchor_i
                        del positive_i

                if exp_cfg['3D_net'] != 'PVRCNN':
                    anchor = torch.cat(anchor_list)
                    positive = torch.cat(positive_list)
                    model_in = torch.cat((anchor, positive))
                    # Normalize between [-1, 1], more or less
                    # model_in = model_in / 100.
                    if exp_cfg['3D_net'] == 'RandLANet':
                        model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
                else:
                    model_in = KittiDataset.collate_batch(anchor_list + positive_list)
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
                        yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] =yaw[i]
                        pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
                elif exp_cfg['rot_representation'] == 'quat':
                    yaw = F.normalize(yaw, dim=1)
                    for i in range(batch_dict['batch_size'] // 2):
                        yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = mat2xyzrpy(RT.quat2mat(yaw[i]))[-1]
                        pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
                elif exp_cfg['rot_representation'] == 'bingham':
                    to_quat = QuadQuatFastSolver()
                    quat_out = to_quat.apply(yaw)[:, [1,2,3,0]]
                    for i in range(yaw.shape[0]):
                        yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = mat2xyzrpy(RT.quat2mat(quat_out[i]))[-1]
                        pred_transl.append(batch_dict['out_translation'][i].detach().cpu())
                elif exp_cfg['rot_representation'].startswith('6dof') and not args.ransac:
                    transformation = batch_dict['transformation']
                    homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                    transformation = torch.cat((transformation, homogeneous), dim=1)
                    transformation = transformation.inverse()
                    for i in range(batch_dict['batch_size'] // 2):
                        yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = mat2xyzrpy(transformation[i])[-1].item()
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
                                sample['positive'][i][:, :3].cpu().numpy())
                            time_icp.tic()
                            result2 = reg_module.registration_icp(
                                        p2, p1, 0.1, result.transformation,
                                        reg_module.TransformationEstimationPointToPoint())
                            time_icp.toc()
                            transformation = torch.tensor(result2.transformation.copy())
                        yaw_preds[sample['anchor_id'][i], sample['positive_id'][i]] = \
                            mat2xyzrpy(transformation)[-1].item()
                        pred_transl.append(transformation[:3, 3].detach().cpu())
                for i in range(batch_dict['batch_size'] // 2):
                    delta_pose = sample['transformation'][i]
                    transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl[i]
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
    print("freiburg")

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

    print("Mean rotation error: ", yaw_error.mean())
    print("Median rotation error: ", np.median(yaw_error))
    print("STD rotation error: ", yaw_error.std())
    print("Mean translation error: ", transl_errors.mean())
    print("Median translation error: ", np.median(transl_errors))
    print("STD translation error: ", transl_errors.std())
    # save_dict = {'rot': yaw_error, 'transl': transl_errors}
    # # save_path = f'./results_for_paper/lcdnet00+08_{exp_cfg["test_sequence"]}'/
    # if '360' in weights_path:
    #     save_path = f'./results_for_paper/lcdnet++_freiburg'
    # elif '00+08' in weights_path:
    #     save_path = f'./results_for_paper/lcdnet00+08_freiburg'
    # else:
    #     save_path = f'./results_for_paper/lcdnet_freiburg'
    # if args.icp:
    #     save_path = save_path+'_icp'
    # elif args.ransac:
    #     save_path = save_path+'_ransac'
    # print("Saving to ", save_path)
    # with open(f'{save_path}.pickle', 'wb') as f:
    #     pickle.dump(save_dict, f)

    print(f"Success Rate: {succ_rate}, RTE: {rte_suc}, RRE: {rre_suc}")

    if args.save_path:
        print("Saving to ", args.save_path)
        with open(args.save_path, 'wb') as f:
            pickle.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--ransac', action='store_true', default=False)
    parser.add_argument('--icp', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument("--z_offset", type=float, default=0.283)
    parser.add_argument("--non_deterministic", action="store_true")
    args = parser.parse_args()

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main_process(0, args.weights_path, args)

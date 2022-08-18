import argparse
import faulthandler; faulthandler.enable()
import os
import time
from functools import partial
from shutil import copy2

import yaml
import wandb
import numpy as np
import open3d as o3d
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random
from torch.nn.parallel import DistributedDataParallel

from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses, KITTILoader3DDictPairs
from datasets.NCLTDataset import NCLTDatasetPairs, NCLTDataset, NCLTDatasetTriplets
from loss import TotalLossFunction
from models.backbone3D.RandLANet.RandLANet import prepare_randlanet_input
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from models.get_models import get_model
from utils.data import datasets_concat_kitti, merge_inputs, datasets_concat_kitti_triplets, datasets_concat_kitti360
from evaluate_kitti import evaluate_model_with_emb
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import get_rt_matrix, mat2xyzrpy
# from utils.qcqp_layer import QuadQuatFastSolver
from utils.rotation_conversion import quaternion_from_matrix, quat2mat
from utils.tools import _pairwise_distance, update_bn_swa, SVDNonConvergenceError, NaNLossError

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.swa_utils import SWALR, AveragedModel



torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def move_from_sample_to_model_in(exp_cfg, sample, model_in):
    if (exp_cfg["load_panoptic"] or exp_cfg["load_semantic"]) and not exp_cfg["use_logits"]:
        input_types = ["anchor", "positive", "negative"]
        data_type_suffixes = ["panoptic", "semantic", "instance", "supersem"]
        panoptic_keys = [i + "_" + t for i in input_types for t in data_type_suffixes]
        panoptic_inputs = {k: sample[k] for k in panoptic_keys if k in sample}
        model_in.update(panoptic_inputs)


def train(model, optimizer, sample, loss_fn, exp_cfg, device, mode='pairs'):
    # with torch.autograd.detect_anomaly():
    if True:
        model.train()
        margin = exp_cfg['margin']

        optimizer.zero_grad()

        if mode == 'pairs':
            if 'sequence' in sample:
                neg_mask = sample['sequence'].view(1,-1) != sample['sequence'].view(-1, 1)
            else:
                neg_mask = torch.zeros((sample['anchor_pose'].shape[0], sample['anchor_pose'].shape[0]),
                                       dtype=torch.bool)

            pair_dist = _pairwise_distance(sample['anchor_pose'])
            neg_mask = ((pair_dist > exp_cfg['negative_distance']) | neg_mask)
            neg_mask = neg_mask.repeat(2, 2).to(device)
        elif mode == 'triplets':
            neg_mask = torch.zeros((sample['anchor_pose'].shape[0]*3, sample['anchor_pose'].shape[0]*3),
                                   dtype=torch.bool)
            for i in range(sample['anchor_pose'].shape[0]):
                neg_mask[i, i + 2*sample['anchor_pose'].shape[0]] = 1.

        if exp_cfg['training_type'] != "3D":
            raise NotImplementedError

        anchor_transl = sample['anchor_pose'].to(device)
        positive_transl = sample['positive_pose'].to(device)
        anchor_rot = sample['anchor_rot'].to(device)
        positive_rot = sample['positive_rot'].to(device)

        anchor_list = []
        positive_list = []
        negative_list = []

        delta_transl = []
        delta_rot = []
        delta_pose = []
        delta_quat = []
        yaw_diff_list = []
        for i in range(anchor_transl.shape[0]):
            anchor = sample['anchor'][i].to(device)
            positive = sample['positive'][i].to(device)
            if mode == 'triplets':
                negative = sample['negative'][i].to(device)

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                positive_set = furthest_point_sample(positive[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                a = anchor_set[0, :].long()
                p = positive_set[0, :].long()
                anchor_i = anchor[a].clone()
                positive_i = positive[p].clone()
                del anchor, positive
                if mode == 'triplets':
                    negative_set = furthest_point_sample(negative[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                    n = negative_set[0, :].long()
                    negative_i = negative[n].clone()
                    del negative
            else:
                anchor_i = anchor
                positive_i = positive
                if mode == 'triplets':
                    negative_i = negative
            # n = negative_set[i, :].long()
            anchor_transl_i = anchor_transl[i]  # Aggiunta
            anchor_rot_i = anchor_rot[i]  # Aggiunta
            positive_transl_i = positive_transl[i]  # Aggiunta
            positive_rot_i = positive_rot[i]  # Aggiunta

            anchor_i_reflectance = anchor_i[:, 3].clone()
            positive_i_reflectance = positive_i[:, 3].clone()
            anchor_i[:, 3] = 1.
            positive_i[:, 3] = 1.
            if mode == 'triplets':
                negative_i_reflectance = negative_i[:, 3].clone()
                negative_i[:, 3] = 1.

            # anchor_test = anchor_i.detach().clone()  # Test
            # positive_test = positive_i.detach().clone()  # Test

            rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
            rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')

            if exp_cfg['point_cloud_augmentation']:

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (np.pi / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3. - 1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_anch_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                anchor_i = rt_anch_augm.inverse() @ anchor_i.T
                anchor_i = anchor_i.T
                anchor_i[:, 3] = anchor_i_reflectance.clone()

                rotz = np.random.rand() * 360 - 180
                rotz = rotz * (3.141592 / 180.0)

                roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                T = torch.rand(3)*3.-1.5
                T[-1] = torch.rand(1)*0.5 - 0.25
                T = T.to(device)

                rt_pos_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                positive_i = rt_pos_augm.inverse() @ positive_i.T
                positive_i = positive_i.T
                positive_i[:, 3] = positive_i_reflectance.clone()

                rt_anch_concat = rt_anchor @ rt_anch_augm
                rt_pos_concat = rt_positive @ rt_pos_augm

                if mode == 'triplets':
                    rotz = np.random.rand() * 360 - 180
                    rotz = rotz * (3.141592 / 180.0)

                    roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                    rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                    T = torch.rand(3)*3.-1.5
                    T[-1] = torch.rand(1)*0.5 - 0.25
                    T = T.to(device)
                    rt_neg_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                    negative_i = rt_neg_augm.inverse() @ negative_i.T
                    negative_i = negative_i.T
                    negative_i[:, 3] = negative_i_reflectance.clone()

                rt_anchor2positive = rt_anch_concat.inverse() @ rt_pos_concat
                ext = mat2xyzrpy(rt_anchor2positive)
                delta_transl_i = ext[0:3]
                delta_rot_i = ext[3:]

            else:
                raise NotImplementedError()


            # negative_i = negative[i, n, 0:3].unsqueeze(0)
            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor_list.append(anchor_i[:, :3].unsqueeze(0))
                positive_list.append(positive_i[:, :3].unsqueeze(0))
                if mode == 'triplets':
                    negative_list.append(negative_i[:, :3].unsqueeze(0))
            else:
                if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
                    anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
                    positive_i = torch.cat((positive_i, sample['positive_logits'][i].to(device)), dim=1)
                anchor_list.append(model.module.backbone.prepare_input(anchor_i))
                positive_list.append(model.module.backbone.prepare_input(positive_i))
                del anchor_i, positive_i
                if mode == 'triplets':
                    if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
                        negative_i = torch.cat((negative_i, sample['negative_logits'][i].to(device)), dim=1)
                    negative_list.append(model.module.backbone.prepare_input(negative_i))
                    del negative_i
            delta_transl.append(delta_transl_i.unsqueeze(0))
            delta_rot.append(delta_rot_i.unsqueeze(0))
            delta_pose.append(rt_anchor2positive.unsqueeze(0))
            delta_quat.append(quaternion_from_matrix(rt_anchor2positive).unsqueeze(0))

        delta_transl = torch.cat(delta_transl)
        delta_rot = torch.cat(delta_rot)
        delta_pose = torch.cat(delta_pose)
        delta_quat = torch.cat(delta_quat)

        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor = torch.cat(anchor_list)
            positive = torch.cat(positive_list)
            model_in = torch.cat((anchor, positive))
            if mode == 'triplets':
                negative = torch.cat(negative_list)
                model_in = torch.cat((anchor, positive, negative))
            if exp_cfg['3D_net'] == 'RandLANet':
                model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
            # Normalize between [-1, 1], more or less
            # model_in = model_in / 100.
        else:
            if mode == 'pairs':
                model_in = KittiDataset.collate_batch(anchor_list + positive_list)
            elif mode == 'triplets':
                model_in = KittiDataset.collate_batch(anchor_list + positive_list + negative_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

        move_from_sample_to_model_in(exp_cfg, sample, model_in)

        metric_head = exp_cfg['weight_transl'] != 0 or exp_cfg['weight_rot'] != 0
        compute_transl = not exp_cfg['weight_transl'] == 0
        compute_rotation = exp_cfg['weight_rot'] != 0
        compute_embeddings = exp_cfg['weight_metric_learning'] != 0

        batch_dict = model(model_in, metric_head, compute_embeddings,
                           compute_transl, compute_rotation, mode=mode)

        # Add the stuff that is missing in the batch dict for the Loss function
        batch_dict["neg_mask"] = neg_mask
        batch_dict["delta_transl"] = delta_transl
        batch_dict["delta_rot"] = delta_rot
        batch_dict["delta_quat"] = delta_quat
        batch_dict["delta_pose"] = delta_pose
        if exp_cfg["load_semantic"] or exp_cfg["load_panoptic"]:
            batch_dict["class_one_hot_map"] = sample["class_one_hot_map"]
            batch_dict["superclass_one_hot_map"] = sample["superclass_one_hot_map"]

        total_loss, sub_losses = loss_fn(batch_dict)
        loss_rot = sub_losses.pop("Loss: Rotation", torch.tensor([0.], device=device))
        loss_transl = sub_losses.pop("Loss: Translation", torch.tensor([0.], device=device))

        if torch.any(torch.isnan(total_loss)):
            raise NaNLossError("Total Loss has NAN")

        if 'TZMGrad' not in exp_cfg['loss_type']:
            total_loss.backward()
        else:
            raise NotImplementedError("TZMGrad not implemented in DDP")
        optimizer.step()

        return total_loss, loss_rot, loss_transl, sub_losses


def test(model, sample, exp_cfg, device):
    model.eval()
    margin = exp_cfg['margin']

    with torch.no_grad():
        if exp_cfg['training_type'] == "3D":
            anchor_transl = sample['anchor_pose'].to(device)
            positive_transl = sample['positive_pose'].to(device)
            anchor_rot = sample['anchor_rot'].to(device)
            positive_rot = sample['positive_rot'].to(device)

            anchor_list = []
            positive_list = []
            delta_transl_list = []
            delta_rot_list = []
            delta_pose_list = []
            for i in range(anchor_transl.shape[0]):
                anchor = sample['anchor'][i].to(device)
                positive = sample['positive'][i].to(device)

                if exp_cfg['3D_net'] != 'PVRCNN':
                    anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                    positive_set = furthest_point_sample(positive[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                    a = anchor_set[0, :].long()
                    p = positive_set[0, :].long()
                    anchor_i = anchor[a]
                    positive_i = positive[p]
                else:
                    anchor_i = anchor
                    positive_i = positive

                anchor_transl_i = anchor_transl[i]  # Aggiunta
                anchor_rot_i = anchor_rot[i]  # Aggiunta
                positive_transl_i = positive_transl[i]  # Aggiunta
                positive_rot_i = positive_rot[i]  # Aggiunta

                rt_anchor = get_rt_matrix(anchor_transl_i, anchor_rot_i, rot_parmas='xyz')
                rt_positive = get_rt_matrix(positive_transl_i, positive_rot_i, rot_parmas='xyz')
                rt_anchor2positive = rt_anchor.inverse() @ rt_positive
                ext = mat2xyzrpy(rt_anchor2positive)
                delta_transl_i = ext[0:3]
                delta_rot_i = ext[3:]
                delta_transl_list.append(delta_transl_i.unsqueeze(0))
                delta_rot_list.append(delta_rot_i.unsqueeze(0))
                delta_pose_list.append(rt_anchor2positive.unsqueeze(0))

                if exp_cfg['3D_net'] != 'PVRCNN':
                    anchor_list.append(anchor_i[:, :3].unsqueeze(0))
                    positive_list.append(positive_i[:, :3].unsqueeze(0))
                else:
                    if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
                        anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
                        positive_i = torch.cat((positive_i, sample['positive_logits'][i].to(device)), dim=1)
                    anchor_list.append(model.module.backbone.prepare_input(anchor_i))
                    positive_list.append(model.module.backbone.prepare_input(positive_i))
                    del anchor_i, positive_i

            delta_transl = torch.cat(delta_transl_list)
            delta_rot = torch.cat(delta_rot_list)
            delta_pose_list = torch.cat(delta_pose_list)

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor = torch.cat(anchor_list)
                positive = torch.cat(positive_list)
                model_in = torch.cat((anchor, positive))
                if exp_cfg['3D_net'] == 'RandLANet':
                    model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
                # Normalize between [-1, 1], more or less
                # model_in = model_in / 100.
            else:
                model_in = KittiDataset.collate_batch(anchor_list + positive_list)
                for key, val in model_in.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    model_in[key] = torch.from_numpy(val).float().to(device)

            move_from_sample_to_model_in(exp_cfg, sample, model_in)

            batch_dict = model(model_in, metric_head=True)
            anchor_out = batch_dict['out_embedding']
            transl = batch_dict['out_translation']
            yaw = batch_dict['out_rotation']
            # transl_diff = anchor_transl - positive_transl
            transl_diff = delta_transl
            if exp_cfg['weight_transl'] > 0. and transl is not None:
                gt_pred_diff = transl_diff - transl
                transl_comps_error = gt_pred_diff.norm(dim=1).mean()
            else:
                transl_comps_error = torch.tensor([0.], device=device)

            # Rotation Loss
            # anchor_yaws = anchor_rot[:, 0]
            # positive_yaws = positive_rot[:, 0]
            diff_yaws = delta_rot[:, 2] % (2*np.pi)
            if exp_cfg['rot_representation'].startswith('sincos'):
                yaw = torch.atan2(yaw[:, 1], yaw[:, 0])
            elif exp_cfg['rot_representation'] == 'yaw':
                yaw = yaw[0]
            elif exp_cfg['rot_representation'].startswith('ce'):
                token = exp_cfg['rot_representation'].split('_')
                num_classes = int(token[1])
                bin_size = 2*np.pi / num_classes
                yaw_bin = yaw[:, :-1]
                yaw_delta = yaw[:, -1]
                yaw_bin = torch.argmax(yaw_bin, dim=1)
                yaw_bin = yaw_bin * bin_size
                yaw = yaw_bin + yaw_delta
            elif exp_cfg['rot_representation'] == 'quat':
                yaw = F.normalize(yaw, dim=1)
                final_yaws = torch.zeros(yaw.shape[0], device=yaw.device, dtype=yaw.dtype)
                for i in range(yaw.shape[0]):
                    final_yaws[i] = mat2xyzrpy(quat2mat(yaw[i]))[-1]
                yaw = final_yaws
            elif exp_cfg['rot_representation'] == 'bingham':
                to_quat = QuadQuatFastSolver()
                quat_out = to_quat.apply(yaw)[:, [1,2,3,0]]
                final_yaws = torch.zeros(yaw.shape[0], device=yaw.device, dtype=yaw.dtype)
                for i in range(yaw.shape[0]):
                    final_yaws[i] = mat2xyzrpy(quat2mat(quat_out[i]))[-1]
                yaw = final_yaws
            elif exp_cfg['rot_representation'] == '6dof':
                transformation = batch_dict['transformation']
                homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                transformation = torch.cat((transformation, homogeneous), dim=1)
                transformation = transformation.inverse()
                final_yaws = torch.zeros(transformation.shape[0], device=transformation.device,
                                         dtype=transformation.dtype)
                for i in range(transformation.shape[0]):
                    final_yaws[i] = mat2xyzrpy(transformation[i])[-1]
                yaw = final_yaws
                transl_comps_error = (transformation[:,:3,3] - delta_pose_list[:,:3,3]).norm(dim=1).mean()

            yaw = yaw % (2*np.pi)
            yaw_error_deg = torch.abs(diff_yaws - yaw)
            yaw_error_deg[yaw_error_deg>np.pi] = 2*np.pi - yaw_error_deg[yaw_error_deg>np.pi]
            yaw_error_deg = yaw_error_deg.mean() * 180 / np.pi

        else:
            anchor_out = model(sample['anchor'].to(device), metric_head=False)

    if exp_cfg['norm_embeddings']:
        anchor_out = anchor_out / anchor_out.norm(dim=1, keepdim=True)
    anchor_out = anchor_out[:anchor_transl.shape[0]]
    return anchor_out, transl_comps_error, yaw_error_deg


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
                    if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
                        anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
                    anchor_list.append(model.module.backbone.prepare_input(anchor_i))
                    del anchor_i

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor = torch.cat(tuple(anchor_list), 0)
                model_in = anchor
                if exp_cfg['3D_net'] == 'RandLANet':
                    model_in = prepare_randlanet_input(ConfigSemanticKITTI2(), model_in.cpu(), device)
                # model_in = model_in / 100.
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


def main_process(gpu, exp_cfg, common_seed, world_size, args):
    global EPOCH
    rank = gpu

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    local_seed = (common_seed + common_seed ** gpu) ** 2
    local_seed = local_seed % (2**32 - 1)
    np.random.seed(common_seed)
    torch.random.manual_seed(common_seed)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    print(f"Process {rank}, seed {common_seed}")

    # t = torch.rand(1).to(device)
    # gather_t = [torch.ones_like(t) for _ in range(dist.get_world_size())]
    # dist.all_gather(gather_t, t)
    # print(rank, t, gather_t)

    current_date = datetime.now()

    dt_fmt = "%d/%m/%Y %H:%M:%S"
    dt_folder_fmt = "%d-%m-%Y_%H-%M-%S"
    dt_string = current_date.strftime(dt_fmt)
    dt_string_folder = current_date.strftime(dt_folder_fmt)

    resume_wandb = False
    if args.weights is not None and args.resume:
        weight_dir = os.path.dirname(args.weights).split(os.path.sep)[-1]
        dt_string_folder = weight_dir
        dt_string = datetime.strptime(dt_string_folder, dt_folder_fmt).strftime(dt_fmt)
        if rank == 0:
            print("\n\n Resuming training from " + dt_string)
        resume_wandb = True

    workers = exp_cfg.get("num_workers") or 2
    exp_cfg['num_workers'] = workers

    exp_cfg['effective_batch_size'] = exp_cfg['batch_size'] * world_size
    if args.wandb and rank == 0:
        project_name = exp_cfg.get("project", "deep_lcd")
        wandb_id = dt_string_folder

        wandb.init(project=project_name, name=dt_string, config=exp_cfg, id=wandb_id,
                   tags=exp_cfg.get("tags", None), notes=exp_cfg.get("notes", None), resume=resume_wandb,
                   settings=wandb.Settings(start_method="fork"))

    if args.dataset == 'kitti':
        sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    else:
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]

    sequences_validation = exp_cfg['test_sequence']
    if not isinstance(sequences_validation, list):
        sequences_validation = [sequences_validation]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)

    data_transform = None

    cfg_load_semantic = exp_cfg.get("load_semantic", False)
    cfg_load_panoptic = exp_cfg.get("load_panoptic", False)
    cfg_use_semantic = exp_cfg.get("use_semantic", False)
    cfg_use_panoptic = exp_cfg.get("use_panoptic", False)
    cfg_filter_dynamic = exp_cfg.get("filter_dynamic", False)
    cfg_dynamic_classes = exp_cfg.get("dynamic_classes")
    cfg_panoptic_weight = exp_cfg.get("panoptic_weight", 0.0)
    cfg_semantic_weight = exp_cfg.get("semantic_weight", 0.0)
    cfg_supersem_weight = exp_cfg.get("supersem_weight", 0.0)
    cfg_semantic_matching_cost = exp_cfg.get("semantic_matching_cost", 0.0)
    use_logits = exp_cfg.get("use_logits", False)

    load_semantic = cfg_load_semantic or cfg_use_semantic or \
                    cfg_panoptic_weight > 0 or cfg_semantic_weight > 0 or cfg_supersem_weight > 0 or \
                    cfg_semantic_matching_cost > 0
    load_panoptic = cfg_load_panoptic or cfg_use_panoptic or \
                    cfg_panoptic_weight > 0 or cfg_semantic_weight > 0 or cfg_supersem_weight > 0 or \
                    cfg_semantic_matching_cost > 0

    exp_cfg["load_semantic"] = load_semantic
    exp_cfg["load_panoptic"] = load_panoptic
    exp_cfg["use_semantic"] = cfg_use_semantic
    exp_cfg["use_panoptic"] = cfg_use_panoptic
    exp_cfg["use_logits"] = use_logits
    exp_cfg["panoptic_weight"] = cfg_panoptic_weight
    exp_cfg["semantic_weight"] = cfg_semantic_weight
    exp_cfg["supersem_weight"] = cfg_supersem_weight
    exp_cfg["semantic_matching_cost"] = cfg_semantic_matching_cost
    exp_cfg["filter_dynamic"] = cfg_filter_dynamic
    exp_cfg["dynamic_classes"] = cfg_dynamic_classes


    if args.dataset == 'kitti':
        if exp_cfg['mode'] == 'pairs':
            training_dataset, dataset_list_train = datasets_concat_kitti(args.data,
                                                                         sequences_training,
                                                                         data_transform,
                                                                         exp_cfg['training_type'],
                                                                         exp_cfg['num_points'],
                                                                         device,
                                                                         without_ground=exp_cfg['without_ground'],
                                                                         loop_file=exp_cfg['loop_file'],
                                                                         jitter=exp_cfg['point_cloud_jitter'],
                                                                         use_semantic=load_semantic,
                                                                         use_panoptic=load_panoptic,
                                                                         use_logits=use_logits,
                                                                         filter_dynamic=exp_cfg["filter_dynamic"],
                                                                         dynamic_classes=exp_cfg["dynamic_classes"]
                                                                         )
        else:
            training_dataset, dataset_list_train = datasets_concat_kitti_triplets(args.data,
                                                                                   sequences_training,
                                                                                   data_transform,
                                                                                   exp_cfg['training_type'],
                                                                                   exp_cfg['num_points'],
                                                                                   device,
                                                                                   without_ground=exp_cfg['without_ground'],
                                                                                   loop_file=exp_cfg['loop_file'],
                                                                                   hard_negative=exp_cfg['hard_mining'],
                                                                                   use_semantic=load_semantic,
                                                                                   use_panoptic=load_panoptic,
                                                                                   jitter=exp_cfg['point_cloud_jitter'],
                                                                                  use_logits=use_logits,
                                                                                  filter_dynamic=exp_cfg["filter_dynamic"],
                                                                                  dynamic_classes=exp_cfg["dynamic_classes"]
                                                                                  )
        validation_dataset = KITTILoader3DDictPairs(args.data, sequences_validation[0],
                                                    os.path.join(args.data, 'sequences', sequences_validation[0], 'poses.txt'),
                                                    exp_cfg['num_points'], device, without_ground=exp_cfg['without_ground'],
                                                    loop_file=exp_cfg['loop_file'], use_semantic=load_semantic,
                                                    use_panoptic=load_panoptic, use_logits=use_logits,
                                                    filter_dynamic=exp_cfg["filter_dynamic"],
                                                    dynamic_classes=exp_cfg["dynamic_classes"]
                                                    )
        dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
                                                os.path.join(args.data, 'sequences', sequences_validation[0], 'poses.txt'),
                                                exp_cfg['num_points'], device, train=False, use_semantic=load_semantic,
                                                use_panoptic=load_panoptic, without_ground=exp_cfg['without_ground'],
                                                loop_file=exp_cfg['loop_file'], use_logits=use_logits,
                                                filter_dynamic=exp_cfg["filter_dynamic"],
                                                dynamic_classes=exp_cfg["dynamic_classes"]
                                                )
    elif args.dataset == 'kitti360':
        training_dataset, dataset_list_train = datasets_concat_kitti360(args.data,
                                                                        sequences_training,
                                                                        data_transform,
                                                                        exp_cfg['training_type'],
                                                                        exp_cfg['num_points'],
                                                                        device,
                                                                        without_ground=exp_cfg['without_ground'],
                                                                        loop_file=exp_cfg['loop_file'],
                                                                        jitter=exp_cfg['point_cloud_jitter'],
                                                                        use_semantic=load_semantic,
                                                                        use_panoptic=load_panoptic,
                                                                        use_logits=use_logits)
        validation_dataset = KITTI3603DDictPairs(args.data, sequences_validation[0],
                                                 without_ground=exp_cfg['without_ground'],
                                                 loop_file=exp_cfg['loop_file'], use_semantic=load_semantic,
                                                 use_panoptic=load_panoptic,
                                                 use_logits=use_logits)
        dataset_for_recall = KITTI3603DPoses(args.data, sequences_validation[0],
                                             train=False, use_semantic=load_semantic,
                                             use_panoptic=load_panoptic,
                                             use_logits=use_logits, without_ground=exp_cfg['without_ground'],
                                             loop_file=exp_cfg['loop_file'])
    elif args.dataset == 'nclt':
        training_dataset = NCLTDatasetTriplets(args.data, '2012-01-08', '2012-01-15', 'loops_on_2012-01-08.pickle')
        validation_dataset = NCLTDatasetPairs(args.data, '2012-12-01', '2012-12-01', 'loop_GT.pickle')
        dataset_for_recall = NCLTDataset(os.path.join(args.data, '2012-12-01'), 'loop_GT.pickle')
        exp_cfg['PC_RANGE'] = [-70.4, -70.4, -0.5, 70.4, 70.4, 3.5]


    dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)
    train_indices = list(range(len(training_dataset)))
    np.random.shuffle(train_indices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed,
    )
    recall_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_for_recall,
        num_replicas=world_size,
        rank=rank,
        seed=common_seed,
        shuffle=False
    )

    loss_fn = TotalLossFunction(exp_cfg)

    positive_distance = 5.
    negative_distance = 25.
    if 'OREOS' in exp_cfg['loop_file']:
        positive_distance = 1.5
        negative_distance = 5.
    elif '15m' in exp_cfg['loop_file']:
        positive_distance = 15.
        negative_distance = 30.
    elif '4m' in exp_cfg['loop_file']:
        positive_distance = 4.
        negative_distance = 10.

    if rank == 0:
        print("Positive distance: ", positive_distance)
    exp_cfg['negative_distance'] = negative_distance

    final_dest = ''
    init_fn = partial(_init_fn, epoch=0, seed=local_seed)
    TrainLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                              sampler=train_sampler,
                                              batch_size=exp_cfg['batch_size'],
                                              num_workers=workers,
                                              worker_init_fn=init_fn,
                                              collate_fn=merge_inputs,
                                              pin_memory=True,
                                              drop_last=True)

    TestLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                             sampler=val_sampler,
                                             batch_size=exp_cfg['batch_size'],
                                             num_workers=workers,
                                             worker_init_fn=init_fn,
                                             collate_fn=merge_inputs,
                                             pin_memory=True)

    RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                               sampler=recall_sampler,
                                               batch_size=exp_cfg['batch_size'],
                                               num_workers=workers,
                                               worker_init_fn=init_fn,
                                               collate_fn=merge_inputs,
                                               pin_memory=True)

    if rank == 0:
        if not os.path.exists(args.checkpoints_dest):
            raise TypeError('Folder for saving checkpoints does not exist!')
        elif args.wandb:
            final_dest = args.checkpoints_dest + '/' + exp_cfg['training_type'] + '/' + dt_string_folder
            if not os.path.exists(final_dest):
                os.mkdir(final_dest)
            wandb.save(f'{final_dest}/best_model_so_far.tar', base_path=final_dest)
            #copy2('wandb_config.yaml', f'{final_dest}/wandb_config.yaml')
            with open(f'{final_dest}/wandb_config.yaml', "w") as wandb_cfg_file:
                yaml.dump({'experiment': exp_cfg}, wandb_cfg_file)
            wandb.save(f'{final_dest}/wandb_config.yaml', base_path=final_dest)
            print("Tracking wandb_config and best_model_so_far.tar files for WandB.")
        else:
            print('Saving checkpoints mod OFF.')

        print(len(TrainLoader), len(train_indices))
        print(len(TestLoader))

    model = get_model(exp_cfg)

    model_params = {k: p for k, p in model.named_parameters()}
    frozen_params = set([])
    unfrozen_params = set([])

    if args.weights is not None:
        if rank == 0:
            print('\n\nLoading pre-trained params from ' + args.weights)
        saved_params = torch.load(args.weights, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(saved_params['state_dict'],
                                                              strict=args.strict_weight_load)

        loaded_keys = [p for p in model.state_dict().keys()
                         if p not in missing_keys and p not in unexpected_keys]
        # loaded_params = [p for p in model_params if p in loaded_keys]

        if rank == 0:
            if loaded_keys:
                print("Loaded values: " + str(len(loaded_keys)))
                print(" - " + "\n - ".join(loaded_keys))
            if missing_keys:
                print("Missing parameters: " + str(len(missing_keys)))
                print(" - " + "\n - ".join(missing_keys))
            if unexpected_keys:
                print("Unexpected parameters found in checkpoint: " + str(len(unexpected_keys)))
                print(" - " + "\n - ".join(unexpected_keys))

        if args.freeze_loaded_weights:
            for param_name, param in model_params.items():
                if param_name in loaded_keys and param.requires_grad:
                    param.requires_grad = False
                    frozen_params.add(param_name)

    if args.freeze_weights_containing:
        for param_name, param in model_params.items():
            if any(param_str in param_name for param_str in args.freeze_weights_containing) and param.requires_grad:
                param.requires_grad = False
                frozen_params.add(param_name)

    if args.unfreeze_weights_containing:
        for param_name, param in model_params.items():
            if any(param_str in param_name for param_str in args.unfreeze_weights_containing) and param_name in frozen_params:
                param.requires_grad = True
                unfrozen_params.add(param_name)
                if param_name in frozen_params:
                    frozen_params.remove(param_name)

    if rank == 0:
        if frozen_params:
            print("\n\nFrozen Parameters: " + str(len(frozen_params)))
            print(" - " + "\n - ".join(frozen_params))

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    # checkpoint = torch.load('checkpoint_3_0.053.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    # Only use for debugging, otherwise it can slow down things
    # model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
    #                                 find_unused_parameters=True)
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank)
    # if args.wandb and rank == 0:
    #     wandb.watch(model)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    start_full_time = time.time()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=exp_cfg['learning_rate'], betas=(exp_cfg['beta1'], exp_cfg['beta2']),
                           eps=exp_cfg['eps'], weight_decay=exp_cfg['weight_decay'], amsgrad=False)

    starting_epoch = 1
    scheduler_epoch = -1
    swa_start = -1
    if args.resume:
        optimizer.load_state_dict(saved_params['optimizer'])
        starting_epoch = saved_params['epoch']
        scheduler_epoch = saved_params['epoch']

    if exp_cfg['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5,
                                                         last_epoch=scheduler_epoch)
    elif exp_cfg['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, exp_cfg['learning_rate'], epochs=exp_cfg['epochs'],
                                            steps_per_epoch=len(TrainLoader), pct_start=0.4, div_factor=10,
                                            final_div_factor=100000, last_epoch=scheduler_epoch*len(TrainLoader))
    elif exp_cfg['scheduler'] == 'swa':
        swa_model = AveragedModel(model.module)
        swa_start = 110
        swa_scheduler = SWALR(optimizer, swa_lr=exp_cfg['learning_rate']*0.5, anneal_epochs=5)

    min_test_loss = None
    best_rot_error = 1000
    max_recall = 0.
    max_auc = 0.
    best_model = {}
    best_model_loss = {}
    savefilename_loss = ''
    savefilename = ''
    old_saved_file = None
    old_saved_file_recall = None
    old_saved_file_auc = None

    np.random.seed(local_seed)
    torch.random.manual_seed(local_seed)

    for epoch in range(starting_epoch, exp_cfg['epochs'] + 1):
        # if epoch == starting_epoch+1:
        #     break
        dist.barrier()

        if epoch == 50 and exp_cfg['mode'] == 'triplets' and exp_cfg['hard_mining'] and args.dataset == 'nclt':
            training_dataset = NCLTDatasetTriplets(args.data, '2012-01-08', '2012-01-15',
                                                   'loops_on_2012-01-08.pickle', hard_negative=True)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset,
                num_replicas=world_size,
                rank=rank,
                seed=common_seed
            )
        if epoch == 50 and exp_cfg['hard_mining'] and args.dataset == 'kitti':
            print("Switching to hard triplet mining")
            training_dataset, dataset_list_train = datasets_concat_kitti_triplets(args.data,
                                                                                  sequences_training,
                                                                                  data_transform,
                                                                                  exp_cfg['training_type'],
                                                                                  exp_cfg['num_points'],
                                                                                  device,
                                                                                  without_ground=exp_cfg['without_ground'],
                                                                                  loop_file=exp_cfg['loop_file'],
                                                                                  hard_negative=True,
                                                                                  use_semantic=load_semantic,
                                                                                  use_panoptic=load_panoptic,
                                                                                  use_logits=use_logits,
                                                                                  jitter=exp_cfg['point_cloud_jitter'])
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset,
                num_replicas=world_size,
                rank=rank,
                seed=common_seed
            )
            exp_cfg['mode'] = 'triplets'

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        recall_sampler.set_epoch(epoch)
        EPOCH = epoch

        init_fn = partial(_init_fn, epoch=epoch, seed=local_seed)
        TrainLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                  sampler=train_sampler,
                                                  batch_size=exp_cfg['batch_size'],
                                                  num_workers=workers,
                                                  worker_init_fn=init_fn,
                                                  collate_fn=merge_inputs,
                                                  pin_memory=True,
                                                  drop_last=True)

        TestLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                 sampler=val_sampler,
                                                 batch_size=exp_cfg['batch_size'],
                                                 num_workers=workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 pin_memory=True)

        RecallLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                                   sampler=recall_sampler,
                                                   batch_size=exp_cfg['batch_size'],
                                                   num_workers=workers,
                                                   worker_init_fn=init_fn,
                                                   collate_fn=merge_inputs,
                                                   pin_memory=True)

        # training_dataset.update_seeds(local_seed, epoch)
        if epoch > starting_epoch:
            if exp_cfg['scheduler'] == 'multistep':
                scheduler.step()
            if args.wandb and rank == 0:
                # wandb.log({"LR": scheduler.get_last_lr()[0]}, commit=False)
                wandb.log({"LR": optimizer.param_groups[0]['lr']}, commit=False, step=epoch)
        if rank == 0:
            ordinal = "st" if epoch == 1 else ("nd" if epoch==2 else ("d" if epoch==3 else "th"))
            print("\n"*3 + "="*80)
            print('\nThis is %d-%s epoch \n' % (epoch, ordinal))
            print("="*80 + "\n"*2)
        epoch_start_time = time.time()
        total_train_loss = 0
        total_rot_loss = 0.
        total_transl_loss = 0.
        total_other_losses = {}
        local_loss = 0.
        local_iter = 0
        total_iter = 0
        other_local_loss_keys = {
            # "Loss: Panoptic Mismatch": "loss_pan",
            "Loss: Semantic Mismatch": "loss_sem",
            # "Loss: Meta-Semantic Mismatch": "loss_mse",
        }
        other_local_loss = {}
        store_data = False

        if rank == 0:
            print('\nTraining')
            print("="*40 + "\n")
        ## Training ##
        for batch_idx, sample in enumerate(TrainLoader):
            # break
            # if batch_idx==3:
            #     break

            # Test to see if memory increases due to variability in PointCloud sizes by skipping some batches
            #if batch_idx < 100:
            #    continue

            start_time = time.time()
            skipped_batches = torch.zeros(1).to(device)

            try:
                loss, loss_rot, loss_transl, other_losses = train(model, optimizer, sample, loss_fn, exp_cfg,
                                                    device, mode=exp_cfg['mode'])
            except (SVDNonConvergenceError, NaNLossError) as err:
                print(err)
                print("Iter {}: Runtime Error found, ignoring batch".format(batch_idx))
                loss = torch.zeros(1).to(device)
                loss_rot = torch.zeros(1).to(device)
                loss_transl = torch.zeros(1).to(device)
                other_losses = {}
                skipped_batches[0] = 1
                #dist.barrier()
                #continue

            if exp_cfg['scheduler'] == 'onecycle':
                scheduler.step()

            dist.barrier()
            dist.reduce(loss, 0)
            dist.reduce(loss_rot, 0)
            dist.reduce(loss_transl, 0)
            dist.reduce(skipped_batches, 0)

            #other_reduced_losses = {}
            if other_losses:
                for v in other_losses.values():
                    dist.reduce(v, 0)
                #other_reduced_losses = {k: dist.reduce(v, 0) for k, v in other_losses.items()}

            batch_world_size = world_size - skipped_batches[0]

            if rank == 0 and batch_world_size:

                loss = (loss / batch_world_size).item()
                loss_rot = (loss_rot / batch_world_size).item()
                loss_transl = (loss_transl / batch_world_size).item()
                local_loss += loss
                local_iter += 1

                for k in other_local_loss_keys.keys():
                    cnt = 0.
                    tmp_other_local_loss = 0
                    if k in other_losses:
                        tmp_other_local_loss += other_losses[k].item()
                        cnt += 1.

                    if k + " (Reverse)" in other_losses:
                        tmp_other_local_loss += other_losses[k + " (Reverse)"].item()
                        cnt += 1.

                    if cnt > 1:
                        tmp_other_local_loss = tmp_other_local_loss / cnt

                    tmp_other_local_loss = tmp_other_local_loss / batch_world_size

                    if k in other_local_loss:
                        other_local_loss[k] += tmp_other_local_loss
                    else:
                        other_local_loss[k] = tmp_other_local_loss


                if batch_idx % args.print_iteration == 0 and batch_idx != 0:
                    other_loss_sum = ""

                    if args.print_other_losses:
                        for k1, k2 in other_local_loss_keys.items():
                            other_loss_sum += f"{k2}={other_local_loss[k1] / local_iter:.3f}, "
                            other_local_loss[k1] = 0.
                    print('Iter %d / %d training loss = %.3f, %stime = %.2f' % (batch_idx,
                                                                               len(TrainLoader),
                                                                               local_loss / local_iter,
                                                                               other_loss_sum,
                                                                               time.time() - start_time))
                    local_loss = 0.
                    local_iter = 0.

                batch_anchor_size = sample['anchor_pose'].shape[0] - skipped_batches
                total_train_loss += loss * batch_anchor_size
                total_rot_loss += loss_rot * batch_anchor_size
                total_transl_loss += loss_transl * batch_anchor_size

                if other_losses:
                    for k, other_loss in other_losses.items():
                        tmp_total_loss = total_other_losses[k] if k in total_other_losses else 0
                        total_other_losses[k] = tmp_total_loss + ((other_loss / batch_world_size).item() *
                                                                  batch_anchor_size)

                total_iter += batch_anchor_size

        if rank == 0:
            print("\n------------------------------------")
            print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_sampler)))
            print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
            print("------------------------------------\n")

        total_test_loss = 0.
        local_loss = 0.0
        local_iter = 0.
        transl_error_sum = 0
        yaw_error_sum = 0
        emb_list = []

        # Testing
        if rank == 0:
            print('\nTesting')
            print("="*40 + "\n")

        if exp_cfg['weight_rot'] > 0. or exp_cfg['weight_transl'] > 0.:
            for batch_idx, sample in enumerate(TestLoader):
                # break
                # if batch_idx == 3:
                #     break
                start_time = time.time()
                _, transl_error, yaw_error = test(model, sample, exp_cfg, device)
                dist.barrier()
                dist.reduce(transl_error, 0)
                dist.reduce(yaw_error, 0)
                if rank == 0:
                    transl_error = (transl_error / world_size).item()
                    yaw_error = (yaw_error / world_size).item()
                    transl_error_sum += transl_error
                    yaw_error_sum += yaw_error
                    local_iter += 1

                    if batch_idx % 20 == 0 and batch_idx != 0:
                        print('Iter %d / %d testing time = %.2f' % (batch_idx, len(TestLoader),
                                                                    time.time() - start_time))
                        local_iter = 0.


        if exp_cfg['weight_metric_learning'] > 0.:
            if rank == 0:
                print('\nEvaluating with embeddings')
                print("="*40 + "\n")

            for batch_idx, sample in enumerate(RecallLoader):
                emb = get_database_embs(model, sample, exp_cfg, device)
                dist.barrier()
                out_emb = [torch.zeros_like(emb) for _ in range(world_size)]
                # print(f'{rank} {emb}')
                dist.all_gather(out_emb, emb)
                # print(f'{rank} {out_emb}')
                # if rank != 0:
                #     dist.gather(emb)
                if rank == 0:
                    interleaved_out = torch.empty((emb.shape[0]*world_size, emb.shape[1]),
                                                  device=emb.device, dtype=emb.dtype)
                    for current_rank in range(world_size):
                        interleaved_out[current_rank::world_size] = out_emb[current_rank]
                    emb_list.append(interleaved_out.detach().clone())

                    if batch_idx % 20 == 0 and batch_idx != 0:
                        print('Iter %d / %d evaluation' % (batch_idx, len(RecallLoader)))

        if rank == 0:
            if exp_cfg['weight_metric_learning'] > 0.:
                emb_list = torch.cat(emb_list)
                emb_list = emb_list[:len(dataset_for_recall)]
                recall, maxF1, auc, auc2 = evaluate_model_with_emb(emb_list, dataset_list_valid, positive_distance)
            final_transl_error = transl_error_sum / len(TestLoader)
            final_yaw_error = yaw_error_sum / len(TestLoader)

            if args.wandb:
                if exp_cfg['weight_rot'] > 0.:
                    wandb.log({"Rotation Loss": (total_rot_loss / len(train_sampler)),
                               "Rotation Mean Error": final_yaw_error}, commit=False, step=epoch)
                if exp_cfg['weight_transl'] > 0. or exp_cfg['rot_representation'] == '6dof':
                    wandb.log({"Translation Loss": (total_transl_loss / len(train_sampler)),
                               "Translation Error": final_transl_error}, commit=False, step=epoch)
                if exp_cfg['weight_metric_learning'] > 0.:
                    wandb.log({"Validation Recall @ 1": recall[0],
                               "Validation Recall @ 5": recall[4],
                               "Validation Recall @ 10": recall[9],
                               "Max F1": maxF1,
                               "AUC": auc,
                               "Real AUC": auc2}, commit=False, step=epoch)
                if total_other_losses:
                    for k, v in total_other_losses.items():
                        wandb.log({k: v / len(train_sampler)}, commit=False, step=epoch)

                wandb.log({"Training Loss": (total_train_loss / len(train_sampler))}, step=epoch)

            print("-" * 40)
            if exp_cfg['weight_metric_learning'] > 0.:
                print(recall)
                print("Max F1: ", maxF1)
                print("AUC: ", auc)
                print("Real AUC: ", auc2)
            print("Translation Error: ", final_transl_error)
            print("Rotation Error: ", final_yaw_error)
            print("-" * 40)

            if epoch > swa_start and exp_cfg['scheduler'] == 'swa':
                swa_model.update_parameters(model.module)
                swa_scheduler.step()

            print("\n Saving models and summaries:")
            print("-" * 40)

            if final_yaw_error < best_rot_error:
                best_rot_error = final_yaw_error
                if args.wandb:
                    savefilename = f'{final_dest}/checkpoint_{epoch}_rot_{final_yaw_error:.3f}.tar'
                    best_model = {
                        'config': exp_cfg,
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "Rotation Mean Error": final_yaw_error
                    }
                    print(" * Best Yaw Error Model")
                    torch.save(best_model, savefilename)
                    if old_saved_file is not None:
                        os.remove(old_saved_file)
                    wandb.run.summary["best_rot_error"] = final_yaw_error
                    temp = f'{final_dest}/best_model_so_far_rot.tar'
                    torch.save(best_model, temp)
                    wandb.save(temp, base_path=final_dest)
                    old_saved_file = savefilename

            if exp_cfg['weight_metric_learning'] > 0.:
                if recall[0] > max_recall:
                    max_recall = recall[0]
                    savefilename_recall = f'{final_dest}/checkpoint_{epoch}_recall_{max_recall:.3f}.tar'
                    best_model_recall = {
                        'epoch': epoch,
                        'config': exp_cfg,
                        'state_dict': model.module.state_dict(),
                        'recall@1': recall[0],
                        'max_F1': maxF1,
                        'AUC': auc,
                        'AUC2': auc2,
                        'optimizer': optimizer.state_dict(),
                    }
                    if args.wandb:
                        wandb.run.summary["best_recall_1"] = max_recall
                        print(" * Best Recall Model")
                        torch.save(best_model_recall, savefilename_recall)

                        temp = f'{final_dest}/best_model_so_far_recall.tar'
                        torch.save(best_model_recall, temp)
                        wandb.save(temp, base_path=final_dest)
                        if old_saved_file_recall is not None:
                            os.remove(old_saved_file_recall)
                        old_saved_file_recall = savefilename_recall
                if auc2 > max_auc:
                    max_auc = auc2
                    savefilename_auc = f'{final_dest}/checkpoint_{epoch}_auc_{max_auc:.3f}.tar'
                    best_model_auc = {
                        'epoch': epoch,
                        'config': exp_cfg,
                        'state_dict': model.module.state_dict(),
                        'recall@1': recall[0],
                        'max_F1': maxF1,
                        'AUC': auc,
                        'AUC2': auc2,
                        'optimizer': optimizer.state_dict(),
                    }
                    if args.wandb:
                        wandb.run.summary["best_auc"] = max_auc
                        print(" * Best AUC Model")
                        torch.save(best_model_auc, savefilename_auc)

                        temp = f'{final_dest}/best_model_so_far_auc.tar'
                        torch.save(best_model_auc, temp)
                        wandb.save(temp, base_path=final_dest)
                        if old_saved_file_auc is not None:
                            os.remove(old_saved_file_auc)
                        old_saved_file_auc = savefilename_auc
            if args.wandb:
                savefilename = f'{final_dest}/checkpoint_last_iter.tar'
                best_model = {
                    'config': exp_cfg,
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                print(" * Latest Model")
                torch.save(best_model, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    if exp_cfg['scheduler'] == 'swa' and rank == 0:
        print("Updating SWA BN")
        RecallLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                   batch_size=exp_cfg['batch_size'],
                                                   num_workers=workers,
                                                   worker_init_fn=init_fn,
                                                   collate_fn=merge_inputs,
                                                   pin_memory=True)
        for batch_idx, sample in enumerate(RecallLoader):
            update_bn_swa(swa_model, sample, exp_cfg, device)

        savefilename = f'{final_dest}/checkpoint_swa.tar'
        print(f"Saving SWA model to {savefilename}")
        best_model = {
            'config': exp_cfg,
            'epoch': epoch,
            'state_dict': swa_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(best_model, savefilename)
        wandb.save(savefilename, base_path=final_dest)

    if args.wandb and rank == 0:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/export/arceyd/dat/kitti/dataset',
                        help='dataset directory')
    parser.add_argument('--dataset', default='kitti',
                        help='dataset')
    parser.add_argument('--epochs', default=100,
                        help='training epochs')
    parser.add_argument('--checkpoints_dest', default='/home/arceyd/MasterThesis/cp',
                        help='training epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--wandb', action="store_true",
                        help='Activate wandb service')
    parser.add_argument('--augmentation', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--gpu_count', type=int, default=-1)
    parser.add_argument('--port', type=str, default='8888')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--strict_weight_load', action='store_true', default=False,
                        help='Loads saved weights only if all keys and shapes match.')
    parser.add_argument('--freeze_loaded_weights', action='store_true', default=False,
                        help='Freeze all of the loaded weights,'
                             'so that only parameters not in the loaded checkpoint are trained.')
    parser.add_argument('--freeze_weights_containing', type=str, default='',
                        help='Comma separated list of strings to be matched with the model\'s parameter names'
                             ' to become frozen during training.')
    parser.add_argument('--unfreeze_weights_containing', type=str, default='',
                        help='Comma separated list of strings to be matched with the model\'s parameter names'
                             ' to be unfrozen.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--print_iteration', type=int, default=20,
                        help='Print stats after this number of iterations.')
    parser.add_argument('--print_other_losses', action='store_true', default=False,
                        help='Print other losses during training')

    parser.add_argument('--config', default="wandb_config.yaml")

    args, override_cfg = parser.parse_known_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.freeze_weights_containing = list(filter(None, args.freeze_weights_containing.split(",")))
    args.unfreeze_weights_containing = list(filter(None, args.unfreeze_weights_containing.split(",")))

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    if override_cfg:
        try:
            # Override configuration from cli arguments
            override_cfg = {k: v for k, v in (c.split(":") for c in override_cfg)}
            override_cfg_str = "\n".join(('"' + k + '": ' + v for k, v in override_cfg.items()))
            override_cfg_dict = yaml.safe_load(override_cfg_str)
            cfg['experiment'].update(override_cfg_dict)
        except Exception as e:
            print("Invalid configuration override:")
            print(e)
            print(override_cfg)

    if args.gpu_count == -1:
        args.gpu_count = torch.cuda.device_count()
    if args.gpu == -1:
        mp.spawn(main_process, nprocs=args.gpu_count, args=(cfg['experiment'], 42, args.gpu_count, args,))
    else:
        main_process(0, cfg['experiment'], 42, args.gpu_count, args)

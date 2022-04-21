import argparse
import os
import time
from functools import partial

import yaml
import wandb
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DPoses
from datasets.KITTI_RPMNet import KITTIRPM3DDictPairs
from datasets.KITTI_data_loader import KITTILoader3DPoses, KITTILoader3DDictPairs
from datasets.NCLTDataset import NCLTDatasetPairs, NCLTDataset, NCLTDatasetTriplets
from loss import SmoothMetricLossV2, NPairLoss, CircleLoss, TripletLoss, sinkhorn_matches_loss, pose_loss, \
    rpm_loss_for_rpmnet
from models.RPMNet.rpmnet import RPMNetEarlyFusion
from models.get_models import get_model
from triple_selector import hardest_negative_selector, random_negative_selector, \
    semihard_negative_selector
from utils.data import datasets_concat_kitti, merge_inputs, datasets_concat_kitti_triplets, datasets_concat_kitti360, \
    datasets_concat_kitti_rpmnet
from evaluate_kitti import evaluate_model_with_emb
from datetime import datetime
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import get_rt_matrix, mat2xyzrpy
from utils.qcqp_layer import QuadQuatFastSolver
from utils.rotation_conversion import quaternion_from_matrix, quaternion_atan_loss, quat2mat
from utils.tools import _pairwise_distance
from pytorch_metric_learning import distances

import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, optimizer, sample, exp_cfg, device):
    # with torch.autograd.detect_anomaly():
    if True:
        model.train()

        optimizer.zero_grad()

        if True:
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

                anchor_i = anchor
                positive_i = positive

                # n = negative_set[i, :].long()
                anchor_transl_i = anchor_transl[i]  # Aggiunta
                anchor_rot_i = anchor_rot[i]  # Aggiunta
                positive_transl_i = positive_transl[i]  # Aggiunta
                positive_rot_i = positive_rot[i]  # Aggiunta

                anchor_i_normals = anchor_i[:, 3:].clone()
                positive_i_normals = positive_i[:, 3:].clone()

                anchor_i = anchor_i[:, :4]
                positive_i = positive_i[:, :4]

                anchor_i[:, 3] = 1.
                positive_i[:, 3] = 1.

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
                    anchor_i = anchor_i.T[:, :3] / 100.
                    anchor_i = torch.cat((anchor_i, anchor_i_normals.clone()), dim=1)

                    rotz = np.random.rand() * 360 - 180
                    rotz = rotz * (3.141592 / 180.0)

                    roty = (np.random.rand() * 6 - 3) * (np.pi / 180.0)
                    rotx = (np.random.rand() * 6 - 3) * (np.pi / 180.0)

                    T = torch.rand(3)*3.-1.5
                    T[-1] = torch.rand(1)*0.5 - 0.25
                    T = T.to(device)

                    rt_pos_augm = get_rt_matrix(T, torch.tensor([rotx, roty, rotz]).to(device))
                    positive_i = rt_pos_augm.inverse() @ positive_i.T
                    positive_i = positive_i.T[:, :3] / 100.
                    positive_i = torch.cat((positive_i, positive_i_normals.clone()), dim=1)

                    rt_anch_concat = rt_anchor @ rt_anch_augm
                    rt_pos_concat = rt_positive @ rt_pos_augm

                    rt_anchor2positive = rt_anch_concat.inverse() @ rt_pos_concat
                    ext = mat2xyzrpy(rt_anchor2positive)
                    delta_transl_i = ext[0:3]
                    delta_rot_i = ext[3:]

                else:
                    raise NotImplementedError()


                # negative_i = negative[i, n, 0:3].unsqueeze(0)
                anchor_list.append(anchor_i.unsqueeze(0))
                positive_list.append(positive_i.unsqueeze(0))

                delta_transl.append(delta_transl_i.unsqueeze(0))
                delta_rot.append(delta_rot_i.unsqueeze(0))
                delta_pose.append(rt_anchor2positive.unsqueeze(0))
                delta_quat.append(quaternion_from_matrix(rt_anchor2positive).unsqueeze(0))

            delta_transl = torch.cat(delta_transl)
            delta_rot = torch.cat(delta_rot)
            delta_pose = torch.cat(delta_pose)
            delta_quat = torch.cat(delta_quat)

            anchor = torch.cat(anchor_list)
            positive = torch.cat(positive_list)
            model_in = {'points_ref': positive, 'points_src': anchor}
            # Normalize between [-1, 1], more or less

            transforms, endpoints = model(model_in)

            # Translation loss
            transl_diff = delta_transl
            total_loss = 0.

            loss_rot = rpm_loss_for_rpmnet(anchor[:, :, :4]*100., transforms, delta_pose)
            inlier_loss = torch.tensor([0.], device=anchor.device, dtype=anchor.dtype)
            for i in range(len(transforms)):
                discount = 0.5
                discount = discount ** (len(transforms) - i - 1)
                inlier_loss += ((1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * exp_cfg['wt_inliers']).mean() * discount
                inlier_loss += ((1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * exp_cfg['wt_inliers']).mean() * discount
            loss_rot += inlier_loss

            total_loss = loss_rot

        total_loss.backward()
        optimizer.step()

        return total_loss


def test(model, sample, exp_cfg, device):
    model.eval()

    with torch.no_grad():
        if True:
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

                anchor_i = anchor
                positive_i = positive
                anchor_i[:, :3] = anchor_i[:, :3] / 100.
                positive_i[:, :3] = positive_i[:, :3] / 100.

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

                anchor_list.append(anchor_i.unsqueeze(0))
                positive_list.append(positive_i.unsqueeze(0))

            delta_transl = torch.cat(delta_transl_list)
            delta_rot = torch.cat(delta_rot_list)
            delta_pose_list = torch.cat(delta_pose_list)

            anchor = torch.cat(anchor_list)
            positive = torch.cat(positive_list)
            model_in = {'points_ref': positive, 'points_src': anchor}

            transforms, endpoints = model(model_in)

            transl_diff = delta_transl

            diff_yaws = delta_rot[:, 2] % (2*np.pi)

            transformation = transforms[-1]
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

    return transl_comps_error, yaw_error_deg


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
    dt_string = current_date.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_folder = current_date.strftime("%d-%m-%Y_%H-%M-%S")

    exp_cfg['effective_batch_size'] = exp_cfg['batch_size'] * world_size
    if args.wandb and rank == 0:
        wandb.init(project="deep_lcd", name=dt_string, config=exp_cfg)

    if args.dataset == 'kitti':
        sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    else:
        sequences_training = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                              "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                              "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    sequences_validation = [exp_cfg['test_sequence']]
    sequences_training = set(sequences_training) - set(sequences_validation)
    sequences_training = list(sequences_training)

    data_transform = None

    if args.dataset == 'kitti':
        training_dataset, dataset_list_train = datasets_concat_kitti_rpmnet(args.data,
                                                                     sequences_training,
                                                                     exp_cfg['num_points'],
                                                                     device,
                                                                     without_ground=False,
                                                                     loop_file=exp_cfg['loop_file'],
                                                                     jitter=exp_cfg['point_cloud_jitter'],
                                                                     use_semantic=False,
                                                                     use_panoptic=False)
        validation_dataset = KITTIRPM3DDictPairs(args.data, sequences_validation[0],
                                                 os.path.join(args.data, 'sequences', sequences_validation[0], 'poses_SEMANTICKITTI.txt'),
                                                 exp_cfg['num_points'], device, without_ground=False,
                                                 loop_file=exp_cfg['loop_file'], use_semantic=False,
                                                 use_panoptic=False)
    elif args.dataset == 'kitti360':
        raise NotImplementedError()
    elif args.dataset == 'nclt':
        raise NotImplementedError()


    # dataset_list_valid = [dataset_for_recall]

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

    loss_fn = None

    final_dest = ''
    init_fn = partial(_init_fn, epoch=0, seed=local_seed)
    TrainLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                              sampler=train_sampler,
                                              batch_size=exp_cfg['batch_size'],
                                              num_workers=2,
                                              worker_init_fn=init_fn,
                                              collate_fn=merge_inputs,
                                              pin_memory=True,
                                              drop_last=True)

    TestLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                             sampler=val_sampler,
                                             batch_size=exp_cfg['batch_size'],
                                             num_workers=2,
                                             worker_init_fn=init_fn,
                                             collate_fn=merge_inputs,
                                             pin_memory=True)

    if rank == 0:
        if not os.path.exists(args.checkpoints_dest):
            raise TypeError('Folder for saving checkpoints does not exist!')
        elif args.wandb:
            final_dest = args.checkpoints_dest + '/' + '3D' + '/' + dt_string_folder
            os.mkdir(final_dest)
            wandb.save(f'{final_dest}/best_model_so_far.tar')
        else:
            print('Saving checkpoints mod OFF.')

        print(len(TrainLoader), len(train_indices))
        print(len(TestLoader))

    model = RPMNetEarlyFusion(exp_cfg['features'], exp_cfg['feature_dim'],
                              exp_cfg['radius'], exp_cfg['num_neighbors'])
    if args.weights is not None:
        print('Loading pre-trained params...')
        saved_params = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(saved_params['state_dict'])

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    # checkpoint = torch.load('checkpoint_3_0.053.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
                                    find_unused_parameters=True)
    # if args.wandb and rank == 0:
    #     wandb.watch(model)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    start_full_time = time.time()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=exp_cfg['learning_rate'])

    starting_epoch = 1
    scheduler_epoch = -1
    if args.resume:
        optimizer.load_state_dict(saved_params['optimizer'])
        starting_epoch = saved_params['epoch']
        scheduler_epoch = saved_params['epoch']

    min_test_loss = None
    best_rot_error = 1000
    best_model = {}
    best_model_loss = {}
    savefilename_loss = ''
    savefilename = ''
    old_saved_file = None

    np.random.seed(local_seed)
    torch.random.manual_seed(local_seed)

    for epoch in range(starting_epoch, exp_cfg['epochs'] + 1):
        dist.barrier()

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        EPOCH = epoch

        init_fn = partial(_init_fn, epoch=epoch, seed=local_seed)
        TrainLoader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                  sampler=train_sampler,
                                                  batch_size=exp_cfg['batch_size'],
                                                  num_workers=2,
                                                  worker_init_fn=init_fn,
                                                  collate_fn=merge_inputs,
                                                  pin_memory=True,
                                                  drop_last=True)

        TestLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                 sampler=val_sampler,
                                                 batch_size=exp_cfg['batch_size'],
                                                 num_workers=2,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 pin_memory=True)

        # training_dataset.update_seeds(local_seed, epoch)
        if rank == 0:
            print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        total_rot_loss = 0.
        total_transl_loss = 0.
        local_loss = 0.
        local_iter = 0
        total_iter = 0
        store_data = False

        ## Training ##
        for batch_idx, sample in enumerate(TrainLoader):
            # break
            # if batch_idx==3:
            #     break
            start_time = time.time()
            loss = train(model, optimizer, sample, exp_cfg, device)

            dist.barrier()
            dist.reduce(loss, 0)
            if rank == 0:
                loss = (loss / world_size).item()
                local_loss += loss
                local_iter += 1

                if batch_idx % 20 == 0 and batch_idx != 0:
                    print('Iter %d / %d training loss = %.3f , time = %.2f' % (batch_idx,
                                                                               len(TrainLoader),
                                                                               local_loss / local_iter,
                                                                               time.time() - start_time))
                    local_loss = 0.
                    local_iter = 0.

                total_train_loss += loss * sample['anchor_pose'].shape[0]

                total_iter += sample['anchor_pose'].shape[0]

        if rank == 0:
            print("------------------------------------")
            print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_sampler)))
            print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
            print("------------------------------------")

        total_test_loss = 0.
        local_loss = 0.0
        local_iter = 0.
        transl_error_sum = 0
        yaw_error_sum = 0
        emb_list = []

        # Testing
        if True:
            for batch_idx, sample in enumerate(TestLoader):
                # break
                # if batch_idx == 3:
                #     break
                start_time = time.time()
                transl_error, yaw_error = test(model, sample, exp_cfg, device)
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
                        print('Iter %d time = %.2f' % (batch_idx,
                                                       time.time() - start_time))
                        local_iter = 0.

        if rank == 0:
            final_transl_error = transl_error_sum / len(TestLoader)
            final_yaw_error = yaw_error_sum / len(TestLoader)

            if args.wandb:
                wandb.log({"Rotation Loss": (total_rot_loss / len(train_sampler)),
                               "Rotation Mean Error": final_yaw_error}, commit=False)
                wandb.log({"Translation Loss": (total_transl_loss / len(train_sampler)),
                               "Translation Error": final_transl_error}, commit=False)
                wandb.log({"Training Loss": (total_train_loss / len(train_sampler))})

            print("------------------------------------")
            print("Translation Error: ", final_transl_error)
            print("Rotation Error: ", final_yaw_error)
            print("------------------------------------")

            if final_yaw_error < best_rot_error:
                best_rot_error = final_yaw_error
                if args.wandb:
                    savefilename = f'{final_dest}/checkpoint_{epoch}_rot_{final_yaw_error:.3f}.tar'
                    best_model = {
                        'config': exp_cfg,
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "Rotation Mean Error": final_yaw_error,
                        "Translation Mean Error": final_transl_error
                    }
                    torch.save(best_model, savefilename)
                    if old_saved_file is not None:
                        os.remove(old_saved_file)
                    wandb.run.summary["best_rot_error"] = final_yaw_error
                    temp = f'{final_dest}/best_model_so_far_rot.tar'
                    torch.save(best_model, temp)
                    wandb.save(temp)
                    old_saved_file = savefilename

            if args.wandb:
                savefilename = f'{final_dest}/checkpoint_last_iter.tar'
                best_model = {
                    'config': exp_cfg,
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "Rotation Mean Error": final_yaw_error,
                    "Translation Mean Error": final_transl_error
                }
                torch.save(best_model, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                        help='dataset directory')
    parser.add_argument('--dataset', default='kitti',
                        help='dataset')
    parser.add_argument('--epochs', default=100,
                        help='training epochs')
    parser.add_argument('--checkpoints_dest', default='/home/cattaneo/checkpoints/deep_lcd',
                        help='training epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--wandb', default=False,
                        help='Activate wandb service')
    parser.add_argument('--augmentation', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--gpu_count', type=int, default=-1)
    parser.add_argument('--port', type=str, default='8888')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # if args.device is not None and not args.no_cuda:
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'dryrun'

    with open("wandb_config_rpmnet.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    if args.gpu_count == -1:
        args.gpu_count = torch.cuda.device_count()
    if args.gpu == -1:
        mp.spawn(main_process, nprocs=args.gpu_count, args=(cfg['experiment'], 42, args.gpu_count, args,))
    else:
        main_process(args.gpu, cfg['experiment'], 42, args.gpu_count, args)

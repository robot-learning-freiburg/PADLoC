import argparse
import collections
import os
import pickle
import time
from collections import OrderedDict
from functools import partial

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import random

from models.get_models import get_model
from utils.data import merge_inputs
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.geometry import mat2xyzrpy

from sensor_msgs.msg import PointCloud2, PointField
import rospy
import sensor_msgs.point_cloud2 as pc2
from rospy_numpy import pointcloud2_to_array
from std_msgs.msg import Int32MultiArray


torch.backends.cudnn.benchmark = True


def prepare_input(model, samples, exp_cfg, device, id=None):
    anchor_list = []
    for point_cloud in samples:
        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor_set = furthest_point_sample(point_cloud[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
            a = anchor_set[0, :].long()
            anchor_i = point_cloud[a]
        else:
            anchor_i = point_cloud

        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor_list.append(anchor_i[:, :3].unsqueeze(0))
        else:
            anchor_list.append(model.backbone.prepare_input(anchor_i))
            del anchor_i

    # if id in range(380, 440) or id in range(30, 90):
    #     saved_anchors[id] = anchor_list
    if exp_cfg['3D_net'] != 'PVRCNN':
        point_cloud = torch.cat(tuple(anchor_list), 0)
        model_in = point_cloud
        model_in = model_in / 100.
    else:
        model_in = KittiDataset.collate_batch(anchor_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)
    return model_in


def callback(message, model=None, device=None, exp_cfg=None):
    time1 = time.time()
    struc_arr = pointcloud2_to_array(message)
    pc = struc_arr.view(np.float32).reshape(struc_arr.shape[0], -1)[:, [0, 1, 2, 4]]
    pc = torch.from_numpy(pc).to(device)
    # print(pc.shape)
    pc_index.append(message.header.seq)

    model.eval()
    with torch.no_grad():
        model_in = prepare_input(model, [pc], exp_cfg, device, message.header.seq)
        batch_dict = model(model_in, metric_head=False, compute_embeddings=True,
                           compute_rotation=False, compute_transl=False)

        emb = batch_dict['out_embedding']
        if exp_cfg['norm_embeddings']:
            emb = emb / emb.norm(dim=1, keepdim=True)
        pc_queue.append(emb.detach().cpu().numpy())
        if len(pc_queue) == 50:
            index.add(pc_queue.popleft())
            # print("Added PC ", message.header.seq-50)
            nearest = index.search(emb.detach().cpu().numpy(), 1)
            # print("Min Distance: ", nearest[0][0][0])
            loop_array = [pc_index[-1], pc_index[nearest[1][0][0]]]
            if nearest[0][0][0] < 0.15:  # THRESHOLD
                loop_msg = Int32MultiArray(data=loop_array)
                pub_loops.publish(loop_msg)
                print("")
                print(f"LOOP CLOSURE FOUND: {pc_index[-1]-1} -> {pc_index[nearest[1][0][0]-1]}")
                print("Distance: ", nearest[0][0][0])
                # pc_list = [saved_anchors[pc_index[-1]][0], saved_anchors[pc_index[nearest[1][0][0]]][0]]
                # model_in = KittiDataset.collate_batch(pc_list)
                # for key, val in model_in.items():
                #     if not isinstance(val, np.ndarray):
                #         continue
                #     model_in[key] = torch.from_numpy(val).float().to(device)
                # batch_dict = model(model_in, metric_head=True, compute_embeddings=False,
                #                    compute_rotation=True)
                # yaw = mat2xyzrpy(batch_dict['transformation'][0])[-1].item()
                # print("Yaw: ", (yaw*180/np.pi) % 360)

    # print(message.header.seq)
    # print("Time: ", time.time() - time1)


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/home/cattaneo/Datasets/KITTI',
                    help='dataset directory')
parser.add_argument('--weights_path', default='/home/cattaneo/checkpoints/deep_lcd')
parser.add_argument('--num_iters', type=int, default=1)
args = parser.parse_args()

torch.cuda.set_device(0)
device = torch.device(0)

saved_params = torch.load(args.weights_path, map_location='cpu')

exp_cfg = saved_params['config']
exp_cfg['batch_size'] = 1

if 'loop_file' not in exp_cfg:
    exp_cfg['loop_file'] = 'loop_GT'
if 'sinkhorn_type' not in exp_cfg:
    exp_cfg['sinkhorn_type'] = 'unbalanced'
if 'shared_embeddings' not in exp_cfg:
    exp_cfg['shared_embeddings'] = True

model = get_model(exp_cfg, is_training=False)
renamed_dict = OrderedDict()
for key in saved_params['state_dict']:
    if not key.startswith('module'):
        renamed_dict = saved_params['state_dict']
        break
    else:
        renamed_dict[key[7:]] = saved_params['state_dict'][key]

res = model.load_state_dict(renamed_dict, strict=False)
if len(res[0]) > 10:
    print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

model.train()
model = model.to(device)

pc_index = []
saved_anchors = {}
pc_queue = collections.deque()
index = faiss.IndexFlatL2(256)

rospy.init_node('LoopClosureDetection', anonymous=True)

pub_loops = rospy.Publisher('loop_closure_from_python', Int32MultiArray, queue_size=1)

callback_partial = partial(callback, model=model, device=device, exp_cfg=exp_cfg)
rospy.Subscriber("/lio_sam/mapping/keyframes", PointCloud2, callback_partial)
rospy.spin()

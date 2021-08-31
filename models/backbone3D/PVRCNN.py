import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

from pcdet.models.backbones_2d.map_to_bev import HeightCompression
from pcdet.models.backbones_3d import VoxelBackBone8x
from pcdet.models.backbones_3d.vfe import MeanVFE

from models.backbone3D.MyVoxelSetAbstraction import MyVoxelSetAbstraction


class ReduceInputDimensionality(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReduceInputDimensionality, self).__init__()
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, batch_dict):
        x = batch_dict['voxel_features']
        B = x.shape[0]
        C = x.shape[-1]
        # x = x.view(-1, C)
        coords = x[:, :4]
        out = self.activation(self.mlp(x[:, 4:]))
        out = torch.cat((coords, out), dim=1)
        batch_dict['voxel_features'] = out.view(B, self.out_dim+4)

        x2 = batch_dict['points']
        B = x2.shape[0]
        coords2 = x2[:, :5]
        out2 = self.activation(self.mlp(x2[:, 5:]))
        out2 = torch.cat((coords2, out2), dim=1)
        batch_dict['points'] = out2.view(B, self.out_dim+5)

        return batch_dict


class PVRCNN(nn.Module):

    def __init__(self, model_cfg, training=True, norm=None, shared_embeddings=False,
                 use_semantic=False, use_panoptic=False):
        super(PVRCNN, self).__init__()
        point_cloud_range = np.array(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        voxel_size = model_cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE']
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        # print(grid_size.astype(np.int64))
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if norm == 'instance':
            norm_fn = partial(nn.InstanceNorm2d, affine=True)  # Used in FLOT
        if norm == 'group':
            norm_fn = partial(nn.GroupNorm, 8)  # Used in RPMNet

        raw_in_channel = len(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list)
        in_channel = raw_in_channel
        if use_semantic:
            in_channel += 19
        if use_panoptic:
            in_channel += 56
        self.point_feature_size = model_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES

        # self.point_feature_encoder = PointFeatureEncoder(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING, point_cloud_range)
        self.data_processor = DataProcessor(model_cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, training)
        self.vfe = MeanVFE(model_cfg, in_channel)

        self.reduce_input_dimensionality = False
        if use_semantic or use_panoptic:
            self.reduce_input_dimensionality = True
            out_channel = 3
            self.reduce_input = ReduceInputDimensionality(in_channel-raw_in_channel, out_channel)
            in_channel = raw_in_channel + out_channel

        self.backbone = VoxelBackBone8x(model_cfg, in_channel, grid_size.astype(np.int64), norm_fn=norm_fn)
        self.to_bev = HeightCompression(model_cfg.MODEL.MAP_TO_BEV)
        self.vsa = MyVoxelSetAbstraction(model_cfg.MODEL.PFE, voxel_size,
                                         point_cloud_range, 256, in_channel, shared_embeddings)

    def prepare_input(self, point_cloud):
        batch_dict = {'points': point_cloud.cpu().numpy(), 'use_lead_xyz': True}
        # batch_dict = self.point_feature_encoder.forward(batch_dict)
        batch_dict = self.data_processor.forward(batch_dict)
        return batch_dict

    def forward(self, batch_dict, compute_embeddings=True, compute_rotation=True):
        # torch.cuda.synchronize()
        # time1 = time.time()
        batch_dict = self.vfe(batch_dict)

        if self.reduce_input_dimensionality:
            batch_dict = self.reduce_input(batch_dict)
        # torch.cuda.synchronize()
        # time2 = time.time()
        batch_dict = self.backbone(batch_dict)
        # torch.cuda.synchronize()
        # time3 = time.time()
        batch_dict = self.to_bev(batch_dict)
        # torch.cuda.synchronize()
        # time4 = time.time()
        batch_dict = self.vsa(batch_dict, compute_embeddings, compute_rotation)
        # torch.cuda.synchronize()
        # time5 = time.time()

        if compute_rotation:
            out = batch_dict['point_features'].view(batch_dict['batch_size'], -1, self.point_feature_size)
            out = out.permute(0, 2, 1).unsqueeze(-1)
            batch_dict['point_features'] = out

        if compute_embeddings:
            out2 = batch_dict['point_features_NV'].view(batch_dict['batch_size'], -1, self.point_feature_size)
            out2 = out2.permute(0, 2, 1).unsqueeze(-1)
            batch_dict['point_features_NV'] = out2
        # B x C x N x 1

        return batch_dict


if __name__ == '__main__':
    cfg_from_yaml_file('pv_rcnn.yaml', cfg)
    model = PVRCNN(cfg)

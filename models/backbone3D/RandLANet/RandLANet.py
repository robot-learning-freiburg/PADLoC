import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

import models.backbone3D.RandLANet.pytorch_utils as pt_utils
from datasets.KITTI_RandLA import KITTIRandLA3DDictPairs
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI, ConfigSemanticKITTI2, DataProcessing

class RandLANet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 512, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(512, 640, kernel_size=(1,1), bn=True)
        # self.dropout = nn.Dropout(0.5)
        # self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points, *args):

        features = end_points['features']  # Batch*channel*npoints

        batch_dict = {}
        pc = end_points['xyz'][0]
        B, N, C = pc.shape
        batch_idx = torch.arange(B, device=pc.device).view(-1, 1).repeat(1, pc.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), pc.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords.clone()
        batch_dict['batch_size'] = B

        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        # features = self.dropout(features)
        # features = self.fc3(features)
        f_out = features.squeeze(3)

        # end_points['logits'] = f_out
        batch_dict['point_features_NV'] = f_out.unsqueeze(-1)
        batch_dict['point_features'] = f_out.unsqueeze(-1)
        return batch_dict

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Network2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        return features

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def prepare_randlanet_input(self, batch_pc):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.config.num_layers):
            neighbour_idx = DataProcessing.knn_search(batch_pc, batch_pc, self.config.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.config.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        flat_inputs = input_points + input_neighbors + input_pools + input_up_samples
        flat_inputs += [features]

        inputs = {}
        num_layers = self.config.num_layers
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(tmp)
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = flat_inputs[4 * num_layers].transpose(1, 2).float()
        inputs['labels'] = None
        inputs['input_inds'] = None
        inputs['cloud_inds'] = None

        return inputs


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def prepare_randlanet_input(config, batch_pc, device, **args):
    features = batch_pc
    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    for i in range(config.num_layers):
        neighbour_idx = DataProcessing.knn_search(batch_pc, batch_pc, config.k_n)
        sub_points = batch_pc[:, :batch_pc.shape[1] // config.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :batch_pc.shape[1] // config.sub_sampling_ratio[i], :]
        up_i = DataProcessing.knn_search(sub_points, batch_pc, 1)
        input_points.append(batch_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_pc = sub_points

    flat_inputs = input_points + input_neighbors + input_pools + input_up_samples
    flat_inputs += [features]

    inputs = {}
    num_layers = config.num_layers
    inputs['xyz'] = []
    for tmp in flat_inputs[:num_layers]:
        inputs['xyz'].append(tmp.to(device))
    inputs['neigh_idx'] = []
    for tmp in flat_inputs[num_layers: 2 * num_layers]:
        inputs['neigh_idx'].append(torch.from_numpy(tmp).long().to(device))
    inputs['sub_idx'] = []
    for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
        inputs['sub_idx'].append(torch.from_numpy(tmp).long().to(device))
    inputs['interp_idx'] = []
    for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
        inputs['interp_idx'].append(torch.from_numpy(tmp).long().to(device))
    inputs['features'] = flat_inputs[4 * num_layers].transpose(1, 2).float().to(device)
    inputs['labels'] = None
    inputs['input_inds'] = None
    inputs['cloud_inds'] = None

    return inputs


if __name__ == '__main__':
    device = torch.device('cuda:0')
    config = ConfigSemanticKITTI2()
    model = RandLANet(config).to(device)
    print(model)
    dataset = KITTIRandLA3DDictPairs('/media/RAIDONE/DATASETS/KITTI/ODOMETRY/', '00',
                                     os.path.join('/media/RAIDONE/DATASETS/KITTI/ODOMETRY/', 'sequences', '00', 'poses_SEMANTICKITTI.txt'),
                                     4096, device)
    sample = dataset[0]
    pc_a = sample['anchor'].to(device)
    pc_p = sample['positive'].to(device)
    anchor_set = furthest_point_sample(pc_a[:, 0:3].unsqueeze(0).contiguous(), config.num_points)
    positive_set = furthest_point_sample(pc_p[:, 0:3].unsqueeze(0).contiguous(), config.num_points)
    a = anchor_set[0, :].long()
    p = positive_set[0, :].long()
    anchor_i = pc_a[a].clone()
    positive_i = pc_p[p].clone()

    model_in = model.prepare_randlanet_input(torch.stack((anchor_i, positive_i)).cpu())

    for key in model_in:
        if type(model_in[key]) is list:
            for i in range(len(model_in[key])):
                model_in[key][i] = model_in[key][i].to(device)
        elif model_in[key] is None:
            pass
        else:
            model_in[key] = model_in[key].to(device)


    model_out = model(model_in)
    print(model_out)

    print("done")

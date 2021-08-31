from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

# from .Pointnet2_PyTorch.models import pointnet2_msg_sem as PN2
# from .MySONet import MySONet
# import config as cfg


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim,
                 gating=True, add_norm=True, is_training=True, normalization='batch'):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_norm
        self.cluster_size = cluster_size
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_norm:
            self.cluster_biases = None
            self.bn1 = norm(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = norm(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_norm, normalization=normalization)

    def forward(self, x):
        #print("MAXX: ",x.max())
        x = x.transpose(1, 3).contiguous()
        batch_size = x.shape[0]
        feature_size = x.shape[-1]
        x = x.view((batch_size, -1, feature_size))
        max_samples = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        #activation = activation.view((-1, max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1).contiguous()
        vlad0 = vlad - a

        vlad1 = F.normalize(vlad0, dim=1, p=2, eps=1e-6)
        vlad2 = vlad1.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad2, dim=1, p=2, eps=1e-6)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)  # I THINK NORMALIZATION HERE IS WRONG

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True, normalization='batch'):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = norm(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True, normalization='batch'):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        if normalization == 'instance':
            norm = lambda x: nn.InstanceNorm2d(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm2d(x)
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = norm(64)
            self.bn2 = norm(128)
            self.bn3 = norm(1024)
            self.bn4 = norm(512)
            self.bn5 = norm(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1, self.k * self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False,
                 max_pool=True, normalization='batch', shared_embeddings=True, out_feat_size=640):
        super(PointNetfeat, self).__init__()
        self.apply_feature_trans = feature_transform
        self.shared_embeddings = shared_embeddings
        self.out_feat_size = out_feat_size
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False, normalization=normalization)
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        if normalization == 'instance':
            norm = lambda x: nn.InstanceNorm2d(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm2d(x)
        self.bn1 = norm(64)
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.bn2 = norm(64)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False, normalization=normalization)
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.bn3 = norm(64)
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.bn4 = norm(128)
        self.conv5 = torch.nn.Conv2d(128, out_feat_size, (1, 1))
        self.bn5 = norm(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        if not shared_embeddings:
            self.conv5_2 = torch.nn.Conv2d(128, out_feat_size, (1, 1))
            # self.bn5_2 = norm(1024)
            self.mp1_2 = torch.nn.MaxPool2d((num_points, 1), 1)

        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x, compute_embeddings, compute_rotation):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        batchsize = x.size()[0]

        batch_dict = {}
        batch_idx = torch.arange(batchsize, device=x.device).view(-1, 1).repeat(1, x.shape[2]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), x.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords.clone()
        batch_dict['batch_size'] = batchsize
        x = x / 100.

        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        # x = x.transpose(2,1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = self.bn5(self.conv5(x))
        x1 = self.conv5(x)
        if not self.max_pool:
            if self.shared_embeddings:
                batch_dict['point_features'] = x1
                batch_dict['point_features_NV'] = x1
            else:
                x2 = self.conv5_2(x)
                batch_dict['point_features'] = x1
                batch_dict['point_features_NV'] = x2
        else:
            x1 = self.mp1(x1)
            x1 = x1.view(-1, self.out_feat_size)
            if self.global_feat:
                batch_dict['point_features'] = x1
            else:
                x1 = x1.view(-1, self.out_feat_size, 1).repeat(1, 1, self.num_points)
                batch_dict['point_features'] = torch.cat([x1, pointfeat], 1)#, trans

            if self.shared_embeddings:
                batch_dict['point_features_NV'] = batch_dict['point_features']
            else:
                x2 = self.conv5_2(x)
                x2 = self.mp1_2(x2)
                x2 = x2.view(-1, self.out_feat_size)
                if self.global_feat:
                    batch_dict['point_features_NV'] = x2
                else:
                    x2 = x2.view(-1, self.out_feat_size, 1).repeat(1, 1, self.num_points)
                    batch_dict['point_features_NV'] = torch.cat([x2, pointfeat], 1)#, trans
        return batch_dict


class PointNetVlad(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVlad, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        # self.net_vlad = netvlad.NetVLAD(dim=output_dim)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_norm=True,
                                     is_training=True)

    def forward(self, x):
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x


'''
class PointNet2Vlad(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super().__init__()
        # self.point_net = PN2.Pointnet2MSGFeatTest(num_points, input_channels=3, output_dim=output_dim, use_xyz=False)
        # self.net_vlad = NetVLADLoupe(feature_size=output_dim, max_samples=num_points, cluster_size=64,
        #                                     output_dim=output_dim, gating=True, add_batch_norm=True,
        #                                     is_training=True)
        self.point_net = PN2.Pointnet2MSGFeatTest2(num_points, input_channels=3, output_dim=1024, use_xyz=False)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_norm=True,
                                     is_training=True)

    def forward(self, x):
        #print("Max Input: ",x.max())
        x1 = self.point_net(x.squeeze(), None)
        x2 = self.net_vlad(x1)
        return x2


class PointNet2NoVlad(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super().__init__()
        self.point_net = PN2.Pointnet2MSGTest3(output_dim, input_channels=3, use_xyz=False)

    def forward(self, x):
        x = self.point_net(x.squeeze(), None)
        return x
'''


# class SONetVlad(nn.Module):
#     def __init__(self, num_points=2500, output_dim=1024):
#         super().__init__()
#         # self.point_net = PN2.Pointnet2MSGFeatTest(num_points, input_channels=3, output_dim=output_dim, use_xyz=False)
#         # self.net_vlad = NetVLADLoupe(feature_size=output_dim, max_samples=num_points, cluster_size=64,
#         #                                     output_dim=output_dim, gating=True, add_batch_norm=True,
#         #                                     is_training=True)
#         self.SONet = MySONet()
#         self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
#                                      output_dim=output_dim, gating=True, add_norm=True,
#                                      is_training=True)
#
#     def forward(self, pc, sn, som_node, som_knn_I, require_grad=True):
#         torch.set_grad_enabled(require_grad)
#         l1 = self.SONet(pc, sn, som_node, som_knn_I, require_grad)
#         x = self.net_vlad(l1.unsqueeze(1))
#         return x


'''
if __name__ == '__main__':
    num_points = 4096
    sim_data = Variable(torch.rand(1, 1, num_points, 3))
    sim_data2 = sim_data.clone().cuda()
    sim_data = sim_data.cuda()
    #
    # trans = STN3d(num_points=num_points).cuda()
    # out = trans(sim_data)
    # print('stn', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=True, feature_transform=True, max_pool=False, num_points=num_points).cuda()
    # out = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # net_vlad = netvlad.NetVLAD(dim=1024).cuda()
    # out2 = net_vlad(out)
    # print('vlad', out2.size())

    pnv = PointNet2Vlad(global_feat=True, feature_transform=True, max_pool=False,
                        output_dim=256, num_points=num_points).cuda()
    pnv.train()
    out3 = pnv(sim_data2)
    print('pnv', out3.size())
'''

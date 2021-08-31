from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from utils import tools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class input_transform_net(nn.Module):
    def __init__(self, num_points = 2500, k = 3):
        super(input_transform_net, self).__init__()
        self.k = k
        self.num_points = num_points
        self.conv1 = torch.nn.Conv2d(k*2, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        nn.init.constant_(self.fc3.weight, 0.0)
        self.fc3.bias = torch.nn.Parameter(torch.eye(k).flatten())

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x, _ = x.max(-1, keepdim=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.squeeze()
        if batchsize == 1:
            x = x.unsqueeze(0)
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, self.k, self.k)
        return x

class FeatureTransform(nn.Module):
    def __init__(self, num_points = 2500):
        super(FeatureTransform, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64 * 64)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).view(1, 64 * 64).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 64, 64)
        return x


def knn(adj_matrix, k=20):
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k)
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int
    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.shape[0]
    point_cloud = point_cloud.squeeze()
    if og_batch_size == 1:  # BOOOH
        point_cloud = point_cloud.unsqueeze(0)

    point_cloud_central = point_cloud
    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    num_dims = point_cloud.shape[2]
    idx_ = torch.arange(batch_size) * num_points
    idx_ = idx_.view(batch_size, 1, 1).to(device)

    point_cloud = point_cloud.contiguous()
    point_cloud_flat = point_cloud.view(-1, num_dims)
    point_cloud_neighbors = point_cloud_flat[nn_idx + idx_]
    point_cloud_central = point_cloud_central.unsqueeze(-2)
    point_cloud_central = point_cloud_central.repeat(1, 1, k, 1)

    edge_feature = torch.cat((point_cloud_central, point_cloud_neighbors - point_cloud_central), dim=-1)
    edge_feature = edge_feature.permute(0, 3, 1, 2)
    return edge_feature


class EdgeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=30):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.conv = nn.Conv2d(2*in_dim, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, num_points, in_features)

        Returns:
            Tensor (batch_size, num_points, out_features)
        """
        batch_size = x.shape[0]
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        adj_matrix = tools.pairwise_batch_mse(x)
        nn_idx = knn(adj_matrix, self.k)
        edge_feature = get_edge_feature(x, nn_idx, k=self.k)
        x = F.relu(self.bn(self.conv(edge_feature)))
        x, _ = x.max(-1, True)
        return x


class EdgeConvBlockSeg(nn.Module):
    def __init__(self, in_dim, out_dim, k=30, second_conv=False):
        super(EdgeConvBlockSeg, self).__init__()
        self.k = k
        self.second_conv = second_conv
        self.conv1 = nn.Conv2d(2*in_dim, out_dim, 1)
        if second_conv:
            self.conv2 = nn.Conv2d(out_dim, out_dim, 1)
        self.conv3 = nn.Conv2d(2*out_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        if second_conv:
            self.bn2 = nn.BatchNorm2d(out_dim)
        self.bn3 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, num_points, in_features)

        Returns:
            Tensor (batch_size, num_points, out_features)
        """
        batch_size = x.shape[0]
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)

        adj_matrix = tools.pairwise_batch_mse(x)
        nn_idx = knn(adj_matrix, self.k)
        edge_feature = get_edge_feature(x, nn_idx, k=self.k)
        x = F.relu(self.bn1(self.conv1(edge_feature)))
        if self.second_conv:
            x = F.relu(self.bn2(self.conv2(x)))
        x_max, _ = x.max(-1, True)
        x_mean = x.mean(-1, True)
        x_cat = torch.cat((x_max, x_mean), 1)
        x = F.relu(self.bn3(self.conv3(x_cat)))

        return x, x_max, x_mean


class EdgeConv(nn.Module):
    def __init__(self, num_points, k=30, num_classes=None):
        """
        Args:
            num_points (int): Number of points per batch
            k (int): Number of neighborhoods
        """
        super(EdgeConv, self).__init__()
        self.num_points = num_points
        self.k = k
        self.num_classes = num_classes
        self.input_transform = input_transform_net(num_points, 3)
        self.EC1 = EdgeConvBlock(3, 64, k)
        self.EC2 = EdgeConvBlock(64, 64, k)
        self.EC3 = EdgeConvBlock(64, 64, k)
        self.EC4 = EdgeConvBlock(64, 128, k)

        self.conv1 = nn.Conv2d(64+64+64+128, 1024, 1)
        if num_classes is not None:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, num_classes)

            self.bn = nn.BatchNorm2d(1024)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(512)

            self.dropout = nn.Dropout(0.3)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud (Tensor): shape (batch_size, num_points, 3)

        Returns:
            Tensor (batch_size, 1024, num_points, 1)
        """
        batch_size = point_cloud.shape[0]
        adj_matrix = tools.pairwise_batch_mse(point_cloud)
        nn_idx = knn(adj_matrix, k=self.k)
        edge_feature = get_edge_feature(point_cloud, nn_idx, k=self.k)
        transform = self.input_transform(edge_feature)
        point_cloud_transformed = torch.bmm(point_cloud, transform)

        x = point_cloud_transformed.permute(0, 2, 1)
        x1 = self.EC1(x)
        x2 = self.EC2(x1)
        x3 = self.EC3(x2)
        x4 = self.EC4(x3)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        #x_cat = x_cat.permute(0,2,1)
        #x = F.relu(self.bn(self.conv1(x_cat)))
        x = self.conv1(x_cat)

        if self.num_classes is not None:
            x, _ = x.max(-2)
            x = x.view(batch_size, -1)

            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)

        return x


class EdgeConvSeg(nn.Module):
    def __init__(self, num_points, k=30, num_classes=None, shared_embeddings=False):
        """
        Args:
            num_points (int): Number of points per batch
            k (int): Number of neighborhoods
        """
        super(EdgeConvSeg, self).__init__()
        self.num_points = num_points
        self.k = k
        self.num_classes = num_classes
        self.shared_embeddings = shared_embeddings
        self.input_transform = input_transform_net(num_points, 3)
        self.EC1 = EdgeConvBlockSeg(3, 64, k, second_conv=True)
        self.EC2 = EdgeConvBlockSeg(64, 64, k)
        self.EC3 = EdgeConvBlockSeg(64, 64, k)

        self.conv1 = nn.Conv2d(64+64+64, 640, 1)
        self.bn1 = nn.BatchNorm2d(640)

        if not shared_embeddings:
            self.conv1_2 = nn.Conv2d(64+64+64, 640, 1)
            self.bn1_2 = nn.BatchNorm2d(640)

        if num_classes is not None:
            self.conv2 = nn.Conv2d(1792, 256, 1)
            self.conv3 = nn.Conv2d(256, 256, 1)
            self.conv4 = nn.Conv2d(256, 128, 1)
            self.conv5 = nn.Conv2d(128, num_classes, 1)

            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(128)

            self.dropout = nn.Dropout(0.3)

    def forward(self, x, *args):
        """
        Args:
            point_cloud (Tensor): shape (batch_size, num_points, 3)

        Returns:
            Tensor (batch_size, 1792, num_points, 1)
        """

        B, N, C = x.shape
        batch_dict = {}
        batch_idx = torch.arange(B, device=x.device).view(-1, 1).repeat(1, x.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), x.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords.clone()
        batch_dict['batch_size'] = B
        # x = x / 100.

        adj_matrix = tools.pairwise_batch_mse(x)
        nn_idx = knn(adj_matrix, k=self.k)
        edge_feature = get_edge_feature(x, nn_idx, k=self.k)
        transform = self.input_transform(edge_feature)
        point_cloud_transformed = torch.bmm(x, transform)

        x = point_cloud_transformed.permute(0, 2, 1).contiguous()
        x1, max_1, mean_1 = self.EC1(x)
        x2, max_2, mean_2 = self.EC2(x1)
        x3, max_3, mean_3 = self.EC3(x2)

        x_cat = torch.cat((x1, x2, x3), dim=1)
        #x_cat = x_cat.permute(0, 2, 1)
        #x = F.relu(self.bn1(self.conv1(x_cat)))
        x = self.conv1(x_cat)
        batch_dict['point_features_NV'] = x

        if self.shared_embeddings:
            batch_dict['point_features'] = x
        else:
            x1 = self.conv1_2(x_cat)
            batch_dict['point_features'] = x1
        # x, _ = x.max(-2, True)
        # x = x.repeat(1, 1, self.num_points, 1)
        #
        # concat = torch.cat((x, max_1, mean_1, x1,
        #                     max_2, mean_2, x2,
        #                     max_3, mean_3, x3, x_cat), dim=1)
        #
        # if self.num_classes is not None:
        #     x = F.relu(self.bn2(self.conv2(concat)))
        #     x = self.dropout(x)
        #     x = F.relu(self.bn2(self.conv3(x)))
        #     x = self.dropout(x)
        #     x = F.relu(self.bn4(self.conv4(x)))
        #     x = self.conv5(x)

        return batch_dict


if __name__ == '__main__':
    np.random.seed(666)
    a = np.random.randn(10, 2000, 3)
    a_tf = torch.from_numpy(a).float()
    #block = EdgeConvBlockSeg(3, 64, second_conv=True)
    #block_out = block(a_tf)
    fn = EdgeConvSeg(2000)
    out = fn(a_tf)
    print("ciao")

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader

from ...pointnet2.data import Indoor3DSemSeg
from ...pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(nn.Module):
    def __init__(self, shared_embeddings=False):
        super(PointNet2SemSegSSG, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.01,
                nsample=32,
                mlp=[0, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.02,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.05,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.1,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 0, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Conv1d(128, 640, kernel_size=1, bias=True)
        if not shared_embeddings:
            self.fc_layer2 = nn.Conv1d(128, 640, kernel_size=1, bias=True)

        # self.fc_lyaer = nn.Sequential(
        #     nn.Conv1d(128, 128, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(128, 13, kernel_size=1),
        # )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, x, *args):
        r"""
            Forward pass of the network

            Parameters
            ----------
            x: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        B, N, C = x.shape
        batch_dict = {}
        batch_idx = torch.arange(B, device=x.device).view(-1, 1).repeat(1, x.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), x.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords.clone()
        batch_dict['batch_size'] = B
        x = x / 100.

        xyz, features = self._break_up_pc(x)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # fc_out = self.fc_lyaer(l_features[0])
        # return self.fc_lyaer(l_features[0])

        x = self.fc_layer(l_features[0])
        x = x.unsqueeze(-1)

        batch_dict['point_features'] = x

        if self.shared_embeddings:
            batch_dict['point_features_NV'] = x
        else:
            x2 = self.fc_layer2(l_features[0])
            x2 = x2.unsqueeze(-1)
            batch_dict['point_features_NV'] = x2

        return batch_dict

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone3D.heads import PointNetHead, CorrelationHead, UOTHead
from models.backbone3D.transformer_head import TransformerHead
from models.backbone3D.pytransformer_head import PyTransformerHead
from models.backbone3D.pytransformer_head_v2 import PyTransformerHead2
from models.backbone3D.pytransformer_feature_multilayer_head_v2 import PyTransformerFeatureMultiLayerHead


#import models.Backbone3D.Pointnet2_PyTorch.models.pointnet2_msg_sem as PN2
# from models.Backbone3D.EdgeConv import EdgeConvSeg, EdgeConv
# from models.Backbone3D.MySONet import MySONet
# from models.Backbone3D.Pointnet2_PyTorch.pointnet2.models.pointnet2_msg_sem import Pointnet2MSGFeatTest2


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, h_n1=1024, h_n2=512):
        super().__init__()
        self.FC1 = nn.Linear(input_dim, h_n1)
        self.FC2 = nn.Linear(h_n1, h_n2)
        self.FC3 = nn.Linear(h_n2, output_dim)
        self.dropout = nn.Dropout(0.2, True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.FC1(x))
        x = self.dropout(x)
        x = F.relu(self.FC2(x))
        x = self.dropout(x)
        x = self.FC3(x)
        return x


class NetVlad(nn.Module):
    def __init__(self, backbone, NV, feature_norm=False):
        super().__init__()
        self.backbone = backbone
        self.NV = NV
        self.feature_norm = feature_norm

    def forward(self, x):
        x = self.backbone(x)
        if self.feature_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.NV(x)
        return x


class LCDNet(nn.Module):
    def __init__(self, backbone, NV, **kwargs):
        # feature_norm=False, fc_input_dim=256,
        #        points_num=4096, head='SuperGlue', rotation_parameters=2,
        #        sinkhorn_iter=5, use_svd=False, sinkhorn_type='unbalanced'):
        super().__init__()
        self.backbone = backbone
        self.NV = NV
        self.feature_norm = kwargs.get("feature_norm", False)
        self.head = kwargs.get("head") or "SuperGlue"

        fc_input_dim = kwargs.get("fc_input_dim") or 256
        points_num = kwargs.get("num_points") or 4096
        rotation_parameters = kwargs.get("rotation_parameters") or 2

        self._transformer_head_dict = {
            "Transformer": TransformerHead,
            "PyTransformer": PyTransformerHead,
            "PyTransformer2": PyTransformerHead2,
            "MLFeatTF": PyTransformerFeatureMultiLayerHead
        }

        #* PointNetHead
        if self.head == 'PointNet':
            self.pose_head = PointNetHead(fc_input_dim, points_num, rotation_parameters)
            self.mp1 = torch.nn.MaxPool2d((points_num, 1), 1)

        # self.pose_head = CorrelationHead(fc_input_dim, points_num, rotation_parameters)

        #* SuperGlueHead
        elif self.head == 'SuperGlue':
            self.pose_head = UOTHead(fc_input_dim, points_num, **kwargs)

        else:

            if self.head not in self._transformer_head_dict:
                raise ValueError(f"Invalid head ({self.head}). Valid values: {self._transformer_head_dict.keys()}.")

            self.pose_head = self._transformer_head_dict[self.head](**kwargs)

    def forward(self, batch_dict, metric_head=True, compute_embeddings=True, compute_transl=True,
                compute_rotation=True, compute_backbone=True, mode='pairs'):
        # time1 = time.time()
        if compute_backbone:
            batch_dict = self.backbone(batch_dict, compute_embeddings, compute_rotation)
        # time2 = time.time()
        # print("BACKBONE: ", time2-time1)
        if self.feature_norm:
            if compute_rotation:
                batch_dict['point_features'] = F.normalize(batch_dict['point_features'], p=2, dim=1)
            if compute_embeddings:
                batch_dict['point_features_NV'] = F.normalize(batch_dict['point_features_NV'], p=2, dim=1)
        # print(backbone_out.shape)
        if self.head == 'PointNet++':
            batch_dict['point_features'] = batch_dict['point_features'].permute(0, 2, 1, 3)
            batch_dict['point_features_NV'] = batch_dict['point_features_NV'].permute(0, 2, 1, 3)
        # print(backbone_out.shape)
        # time1 = time.time()
        if compute_embeddings: # and self.head != 'Transformer':
            embedding = self.NV(batch_dict['point_features_NV'])
            # time2 = time.time()
            # print("NetVlad: ", time2-time1)
        else:
            embedding = None
        batch_dict['out_embedding'] = embedding

        # time1 = time.time()
        if metric_head:
            #* PointNetHead
            if self.head == 'PointNet':
                B, C, NUM, _ = batch_dict['point_features'].shape
                if mode == 'pairs':
                    assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
                    B = B // 2
                    anchors_feature_maps = batch_dict['point_features'][:B, :, :]
                    positives_feature_maps = batch_dict['point_features'][B:, :, :]
                else:
                    assert B % 3 == 0, "Batch size must be multiple of 3: B anchor + B positive + B negative samples"
                    B = B // 3
                    anchors_feature_maps = batch_dict['point_features'][:B, :, :]
                    positives_feature_maps = batch_dict['point_features'][B:2*B, :, :]
                anchors_feature_maps = self.mp1(anchors_feature_maps)
                positives_feature_maps = self.mp1(positives_feature_maps)
                pose_head_in = torch.cat((anchors_feature_maps, positives_feature_maps), 1)
                transl, yaw = self.pose_head(pose_head_in, compute_transl, compute_rotation)
                batch_dict['out_rotation'] = yaw
                batch_dict['out_translation'] = transl

            #* SuperGlueHead
            elif self.head == 'SuperGlue':
                batch_dict = self.pose_head(batch_dict, compute_transl, compute_rotation, mode=mode)

            #* TransformerHead
            elif self.head in self._transformer_head_dict:
                batch_dict = self.pose_head(batch_dict, mode=mode)

            # time2 = time.time()
            # print("Pose Head: ", time2-time1)

        else:
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None

        return batch_dict

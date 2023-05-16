import torch.nn as nn
import torch.nn.functional as F

from models.backbone3D.heads import UOTHead
from models.backbone3D.transformer_head import TransformerHead
from models.backbone3D.pytransformer_head import PyTransformerHead
from models.backbone3D.tf_head import TFHead
from models.backbone3D.deepclosestpoint_head import DeepClosestPointHead as DCP
from models.backbone3D.deepclosestpointorig_head import DeepClosestPointHead as DCPOrig
from models.backbone3D.pytransformer_feature_multilayer_head_v2 import PyTransformerFeatureMultiLayerHead


class PADLoC(nn.Module):
    def __init__(self, backbone, NV, **kwargs):
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
            "TFHead": TFHead,
            "MLFeatTF": PyTransformerFeatureMultiLayerHead,
            "DeepClosestPoint": DCPOrig,
            "DeepClosestPointMine": DCP
        }

        if self.head == 'SuperGlue':
            self.pose_head = UOTHead(fc_input_dim, points_num, **kwargs)

        else:

            if self.head not in self._transformer_head_dict:
                raise ValueError(f"Invalid head ({self.head}). Valid values: {self._transformer_head_dict.keys()}.")

            self.pose_head = self._transformer_head_dict[self.head](**kwargs)

    def forward(self, batch_dict, metric_head=True, compute_embeddings=True, compute_transl=True,
                compute_rotation=True, compute_backbone=True, mode='pairs'):

        if compute_backbone:
            batch_dict = self.backbone(batch_dict, compute_embeddings, compute_rotation)

        if self.feature_norm:
            if compute_rotation:
                batch_dict['point_features'] = F.normalize(batch_dict['point_features'], p=2, dim=1)
            if compute_embeddings:
                batch_dict['point_features_NV'] = F.normalize(batch_dict['point_features_NV'], p=2, dim=1)

        if self.head == 'PointNet++':
            batch_dict['point_features'] = batch_dict['point_features'].permute(0, 2, 1, 3)
            batch_dict['point_features_NV'] = batch_dict['point_features_NV'].permute(0, 2, 1, 3)

        if compute_embeddings:
            embedding = self.NV(batch_dict['point_features_NV'])
        else:
            embedding = None
        batch_dict['out_embedding'] = embedding

        if not metric_head:
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None
            return batch_dict

        if self.head == 'SuperGlue':
            batch_dict = self.pose_head(batch_dict, compute_transl, compute_rotation, mode=mode)

        elif self.head in self._transformer_head_dict:
            batch_dict = self.pose_head(batch_dict, mode=mode)

        return batch_dict

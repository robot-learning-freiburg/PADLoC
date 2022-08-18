from collections import OrderedDict

from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg as pvrcnn_cfg
import torch

#from models.backbone2D import models_2d as mdl2d
#from models.backbone2D.originalNetvlad import vd16_tokyoTM_conv5_3_max_dag, weights_init
from models.backbone3D.EdgeConv import EdgeConvSeg
from models.backbone3D.PVRCNN import PVRCNN
from models.backbone3D.PointNetVlad import PointNetfeat, NetVLADLoupe
from models.backbone3D.Pointnet2_PyTorch.pointnet2.models import PointNet2SemSegMSG, PointNet2SemSegSSG
from models.backbone3D.RandLANet.RandLANet import RandLANet
from models.backbone3D.RandLANet.helper_tool import ConfigSemanticKITTI2
from models.backbone3D.models_3d import LCDNet
from models.backbone3D.pointnet2_pytorch_geometric import PointNet2Geometric


def get_model(exp_cfg, is_training=True):
    rotation_parameters = 1
    exp_cfg['use_svd'] = False
    if exp_cfg['rot_representation'].startswith('sincos'):
        rotation_parameters = 2
    elif exp_cfg['rot_representation'].startswith('ce'):
        token = exp_cfg['rot_representation'].split('_')
        rotation_parameters = int(token[1]) + 1
    elif exp_cfg['rot_representation'] == 'quat':
        rotation_parameters = 4
    elif exp_cfg['rot_representation'] == 'bingham':
        rotation_parameters = 10
    elif exp_cfg['rot_representation'] == '6dof':
        exp_cfg['use_svd'] = True

    if exp_cfg['training_type'] == 'RGB':
        # print('Original NetVlad')
        # NV = mdl2d.FCLayer((512 * exp_cfg['image_height'] // 16) * (exp_cfg['image_width'] // 16),
        #              exp_cfg['feature_output_dim_2D'])
        #NV = mdl2d.FCLayer((512 * 320 // 16) * (1216 // 16), exp_cfg['feature_output_dim_2D'])
        #backbone2D = vd16_tokyoTM_conv5_3_max_dag()
        #backbone2D.apply(weights_init)  # Cambiare con i pretrainati
        #model = mdl2d.NetVlad(backbone2D, NV)
        pass
    elif exp_cfg['training_type'] == '3D':
        if exp_cfg['3D_net'] == 'PointNet':
            point_net = PointNetfeat(exp_cfg['num_points'], global_feat=True, feature_transform=True,
                                     max_pool=False, normalization=exp_cfg['model_norm'],
                                     shared_embeddings=exp_cfg['shared_embeddings'])
            net_vlad = NetVLADLoupe(feature_size=exp_cfg['feature_size'], cluster_size=256,
                                    output_dim=exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=is_training)
            model = LCDNet(point_net, net_vlad, feature_norm=False, fc_input_dim=exp_cfg['feature_size'],
                           points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                           rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                           use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
        elif exp_cfg['3D_net'] == 'PointNet++':
            # params = {"model.use_xyz": True}
            # point_net2_cls_sgg = PointNet2ClassificationSSG(params)
            # point_net2_sem_seg = PointNet2SemSegSSG(point_net2_cls_sgg)
            # point_net2 = PointNet2SemSegMSG(point_net2_sem_seg)
            # point_net2 = PointNet2SemSegMSG(params)
            point_net2 = PointNet2SemSegSSG(exp_cfg['shared_embeddings'])
            net_vlad = NetVLADLoupe(feature_size=exp_cfg['feature_size'], cluster_size=exp_cfg['cluster_size'],
                                    output_dim=exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=is_training)
            model = LCDNet(point_net2, net_vlad, feature_norm=False, fc_input_dim=exp_cfg['feature_size'],
                           points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                           rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                           use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
        elif exp_cfg['3D_net'] == 'PointNet++_geometric':
            point_net2 = PointNet2Geometric()
            net_vlad = NetVLADLoupe(feature_size=exp_cfg['feature_size'], cluster_size=exp_cfg['cluster_size'],
                                    output_dim=exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=is_training)
            model = LCDNet(point_net2, net_vlad, feature_norm=False, fc_input_dim=exp_cfg['feature_size'],
                           points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                           rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                           use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
        elif exp_cfg['3D_net'] == 'EdgeConv':
            # print('EdgeConv')
            edge_conv = EdgeConvSeg(exp_cfg['num_points'], k=30)
            net_vlad = NetVLADLoupe(feature_size=640, cluster_size=exp_cfg['cluster_size'],
                                    output_dim=exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=is_training)
            model = LCDNet(edge_conv, net_vlad, feature_norm=False, fc_input_dim=1792,
                           points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                           rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                           use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
        elif exp_cfg['3D_net'] == 'RandLANet':
            # print('EdgeConv')
            edge_conv = RandLANet(ConfigSemanticKITTI2())
            net_vlad = NetVLADLoupe(feature_size=640, cluster_size=exp_cfg['cluster_size'],
                                    output_dim=exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=is_training)
            model = LCDNet(edge_conv, net_vlad, feature_norm=False, fc_input_dim=1792,
                           points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                           rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                           use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
        elif exp_cfg['3D_net'] == 'PVRCNN':
            pvrcnn_cfg_file = exp_cfg.get("pvrcnn_cfg_file", "./models/backbone3D/pv_rcnn.yaml")
            cfg_from_yaml_file(pvrcnn_cfg_file, pvrcnn_cfg)
            pvrcnn_cfg.MODEL.PFE.NUM_KEYPOINTS = exp_cfg['num_points']
            pvrcnn_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES = exp_cfg['feature_size']
            if 'PC_RANGE' in exp_cfg:
                pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE = exp_cfg['PC_RANGE']
            exp_cfg['PC_RANGE'] = pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE
            pvrcnn = PVRCNN(pvrcnn_cfg, is_training, exp_cfg['model_norm'], exp_cfg['shared_embeddings'],
                            exp_cfg['use_semantic'], exp_cfg['use_panoptic'])

            net_vlad = None

            if exp_cfg['head'] != "Transformer" or exp_cfg['desc_head'] == "NetVLAD":
                net_vlad = NetVLADLoupe(feature_size=pvrcnn_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES,
                                        cluster_size=exp_cfg['cluster_size'],
                                        output_dim=exp_cfg['feature_output_dim_3D'],
                                        gating=True, add_norm=True, is_training=is_training)

            lcd_net_kwargs = {}
            lcd_net_kwargs.update(exp_cfg)
            lcd_net_kwargs['feature_norm'] = False
            lcd_net_kwargs['fc_input_dim'] = 640
            lcd_net_kwargs['rotation_parameters'] = rotation_parameters

            model = LCDNet(pvrcnn, net_vlad, **lcd_net_kwargs)
        else:
            raise TypeError("Unknown 3D network")
    else:
        raise TypeError("Unknown training mod")
    return model


def load_model(weights_path, override_cfg_dict=None):

    saved_params = torch.load(weights_path, map_location='cpu')

    exp_cfg = saved_params['config']

    if override_cfg_dict is not None:
        exp_cfg.update(override_cfg_dict)

    model = get_model(exp_cfg, is_training=False)

    renamed_dict = OrderedDict()
    for key in saved_params['state_dict']:
        if not key.startswith('module'):
            renamed_dict = saved_params['state_dict']
            break
        else:
            renamed_dict[key[7:]] = saved_params['state_dict'][key]

    # Reshape weights to account for differences in implementation between OpenPCDet versions
    if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != \
            model.state_dict()['backbone.backbone.conv_input.0.weight'].shape:
        for key in renamed_dict:
            if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                if len(renamed_dict[key].shape) == 5:
                    renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)

    res = model.load_state_dict(renamed_dict, strict=False)
    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    return model, exp_cfg

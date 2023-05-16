from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg as pvrcnn_cfg
import torch

from models.backbone3D.PVRCNN import PVRCNN
from models.backbone3D.PointNetVlad import NetVLADLoupe
from models.backbone3D.models_3d import PADLoC


Model = torch.nn.Module
Config = Dict[str, Any]


def get_model(
        exp_cfg: Config,
        is_training: bool = True
) -> Model:
    """
    Method for constructing a model from a configuration dictionary.

    :param exp_cfg: Model configuration.
    :type exp_cfg: dict
    :param is_training: Flag for configuring the model for either training or inference.
        If set to true, results may not be deterministic.
    :type is_training: bool

    :return: Model
    :rtype: torch.nn.Module
    """

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

    if exp_cfg['training_type'] != '3D':
        raise ValueError(f"Invalid training type {exp_cfg['training_type']}. Only '3D' supported.")

    if exp_cfg['3D_net'] != 'PVRCNN':
        raise ValueError(f"Invalid 3D_net {exp_cfg['3d_net']}. Only 'PVRCNN' supported.")

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

    model = PADLoC(pvrcnn, net_vlad, **lcd_net_kwargs)

    return model


def load_model(
        weights_path: str,
        override_cfg_dict: Optional[Config] = None,
        is_training: bool = False,
        strict_load: bool = False
) -> Tuple[Model, Config]:
    """
    Method for loading a model from a saved checkpoint.

    :param weights_path: Path to the checkpoint file to be loaded.
    :type weights_path: str
    :param override_cfg_dict: Settings that will override the saved model configuration. Useful for changing the
        batch size and other parameters.
    :type override_cfg_dict: dict
    :param is_training: Flag for configuring the model for either training or inference.
        If set to true, results may not be deterministic.
    :type is_training: bool
    :param strict_load: If set to True, the saved weights must match the model's architecture. Otherwise, an exception
        will be raised. If set to False, missing and extra weights will be logged to standard output.
    :type strict_load: bool

    :return: Model
    :rtype: torch.nn.Module
    """

    saved_params = torch.load(weights_path, map_location='cpu')

    exp_cfg = saved_params['config']

    if override_cfg_dict is not None:
        exp_cfg.update(override_cfg_dict)

    model = get_model(exp_cfg, is_training=is_training)

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

    res = model.load_state_dict(renamed_dict, strict=strict_load)
    if not strict_load and len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    return model, exp_cfg

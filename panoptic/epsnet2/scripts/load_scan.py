import glob
from argparse import Namespace
from itertools import chain
from os import path
import torch
import numpy as np
import torch.utils.data as data
import umsgpack
from PIL import Image

from epsnet.data.laserscan_p import LaserScan
import epsnet.models as eps_models

from epsnet.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, \
    ProposalMatcher
from epsnet.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from epsnet.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from epsnet.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from epsnet.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
# from epsnet.models.panoptic import PanopticNet_range as PanopticNet
from epsnet.modules.fpn import FPN, FPNBody
from epsnet.modules.heads import FPNSemanticHeadDPCR as FPNSemanticHeadDPC
from epsnet.modules.heads import FPNMaskHead, RPNHead
from epsnet.utils.misc import config_to_string, norm_act_from_config
from epsnet.config import load_config, DEFAULTS as DEFAULT_CONFIGS

from panoptic.epsnet2.epsnet.models.panoptic import PanopticNet_range


def scan_to_epsnet(point_cloud):
    sensor_img_H = 64
    sensor_img_W = 2048
    sensor_fov_up = 3
    sensor_fov_down = -25
    max_points = 150000
    sensor_img_means = torch.tensor([12.12,10.88,0.23,-1.04,0.21],
                                    dtype=torch.float).view(-1,1,1)
    sensor_img_stds = torch.tensor([12.32,11.47,6.91,0.86,0.16],
                                        dtype=torch.float).view(-1,1,1)
    scan = LaserScan(project=True,
                     H=sensor_img_H,
                     W=sensor_img_W,
                     fov_up=sensor_fov_up,
                     fov_down=sensor_fov_down)
    scan.set_points(point_cloud[:, :3], point_cloud[:, 3])

    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    proj_x = torch.full([max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - sensor_img_means
            ) / sensor_img_stds
    proj = proj * proj_mask.float()
    rec = {"img": proj, "proj_msk": proj_mask}
    size = (proj_mask.shape[0], proj_mask.shape[1])
    rec["p_x"] = proj_x
    rec["p_y"] = proj_y
    rec["proj_range"] = proj_range
    rec["unproj_range"] = unproj_range
    rec["n_points"] = unproj_n_points

    rec["idx"] = None
    rec["size"] = size
    rec["rel_path"] = None
    rec["abs_path"] = None

    return rec

def make_model(config, num_thing, num_stuff):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    body_fn = eps_models.__dict__["net_" + body_config["body"]]
    body_fn_range = eps_models.__dict__["net_" + 'efficientnet-b0']
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    body_range = body_fn_range(norm_act=norm_act_static, **{'in_channels':1})

    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    #for n, m in body.named_modules():
    #    for mod_id in range(1, body_config.getint("num_frozen") + 1):
    #        if ("mod%d" % mod_id) in n:
    #            freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")

    if 'efficient' in body_config['body']:
        fpn = FPN([body.corresponding_channels[2],body.corresponding_channels[3],body.corresponding_channels[5],body.corresponding_channels[-1]],
                  [body_range.corresponding_channels[2],body_range.corresponding_channels[3],body_range.corresponding_channels[5],body_range.corresponding_channels[-1]],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config["interpolation"])
    else:
        fpn = FPN([body_channels[inp] for inp in fpn_inputs],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create instance segmentation network
    bbx_prediction_generator = BbxPredictionGenerator(roi_config.getfloat("nms_threshold"),
                                                      roi_config.getfloat("score_threshold"),
                                                      roi_config.getint("max_predictions"))
    msk_prediction_generator = MskPredictionGenerator()
    roi_size = roi_config.getstruct("roi_size")
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    bbx_loss = DetectionLoss(roi_config.getfloat("sigma"))
    msk_loss = InstanceSegLoss()
    lbl_roi_size = tuple(s * 2 for s in roi_size)
    roi_algo = InstanceSegAlgoFPN(
        bbx_prediction_generator, msk_prediction_generator, proposal_matcher, bbx_loss, msk_loss, classes,
        roi_config.getstruct("bbx_reg_weights"), roi_config.getint("fpn_canonical_scale"),
        roi_config.getint("fpn_canonical_level"), roi_size, roi_config.getint("fpn_min_level"),
        roi_config.getint("fpn_levels"), lbl_roi_size, roi_config.getboolean("void_is_background"))
    roi_head = FPNMaskHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["total"])
    sem_head = FPNSemanticHeadDPC(fpn_config.getint("out_channels"),
                                  sem_config.getint("fpn_min_level"),
                                  sem_config.getint("fpn_levels"),
                                  classes["total"],
                                  pooling_size=sem_config.getstruct("pooling_size"),
                                  norm_act=norm_act_static)

    # Create final network
    return PanopticNet_range(body, body_range, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, None, classes, msk_prediction_generator)

def load_meta(meta_file):
    with open(meta_file, "rb") as fid:
        data = umsgpack.load(fid, encoding="utf-8")
        meta = data["meta"]
    return meta


def make_config(args):

    conf = load_config(args.config, DEFAULT_CONFIGS["panoptic"])

    return conf


def resume_from_snapshot(model, snapshot, modules):
    snapshot = torch.load(snapshot, map_location="cpu")
    state_dict = snapshot["state_dict"]

    for module in modules:
        if module in state_dict:
            _load_pretraining_dict(getattr(model, module), state_dict[module])
        else:
            raise KeyError("The given snapshot does not contain a state_dict for module '{}'".format(module))

    return snapshot


def _load_pretraining_dict(model, state_dict):
    """Load state dictionary from a pre-training snapshot

    This is an even less strict version of `model.load_state_dict(..., False)`, which also ignores parameters from
    `state_dict` that don't have the same shapes as the corresponding ones in `model`. This is useful when loading
    from pre-trained models that are trained on different datasets.

    Parameters
    ----------
    model : torch.nn.Model
        Target model
    state_dict : dict
        Dictionary of model parameters
    """
    model_sd = model.state_dict()

    for k, v in model_sd.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                del state_dict[k]

    model.load_state_dict(state_dict, False)


def get_eps_model():
    args = Namespace()
    args.meta = '/home/cattaneo/deep_lcd/panoptic/epsnet2/scripts/metadata.bin'
    args.config = '/home/cattaneo/deep_lcd/panoptic/epsnet2/scripts/config/SemanticKITTI.ini'
    # args.min_area = 4096
    args.min_area = 64
    args.score_threshold = 0.5
    args.iou_threshold = 0.5
    config = make_config(args)
    meta = load_meta(args.meta)
    model = make_model(config, meta["num_thing"], meta["num_stuff"])
    resume_from_snapshot(model, '/home/cattaneo/deep_lcd/panoptic/epsnet2/scripts/downsample_model/model_best.pth.tar',
                         ["body", "rpn_head", "roi_head", "sem_head"])
    return model


def compact_labels(msk, cat, track_id=None, iscrowd=None):
    ids = np.unique(msk)
    if 0 not in ids:
        ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

    ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
    ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)
    #        print (cat,track_id)
    msk = ids_to_compact[msk]
    cat = cat[ids]

    #iscrowd = iscrowd[ids]
    #track_id = track_id[ids]
    #        for l,j in zip(cat,track_id):
    #            print (l,j)
    return msk, cat #, track_id, iscrowd

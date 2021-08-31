import argparse
import time
from collections import OrderedDict
from functools import partial
from os import path, mkdir
import os
import numpy as np
import torch
import torch.utils.data as data
import umsgpack
from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from torch import distributed
import cv2
import epsnet.models as models
from epsnet.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, \
    ProposalMatcher
from epsnet.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from epsnet.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from epsnet.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from epsnet.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from epsnet.config import load_config, DEFAULTS as DEFAULT_CONFIGS
from epsnet.data import ISSTestDataset_ as ISSTestDataset
from epsnet.data import ISSTestTransform, iss_collate_fn
from epsnet.data.sampler import DistributedARBatchSampler
from epsnet.modules.fpn import FPN, FPNBody
from epsnet.modules.heads import FPNSemanticHeadDPCR as FPNSemanticHeadDPC
from epsnet.modules.heads import FPNMaskHead, RPNHead
from epsnet.utils import logging
from epsnet.utils.meters import AverageMeter
from epsnet.utils.misc import config_to_string, norm_act_from_config
from epsnet.utils.panoptic import PanopticPreprocessing
from epsnet.utils.parallel import DistributedDataParallel
from epsnet.utils.snapshot import resume_from_snapshot
import sys
import yaml
from panoptic.epsnet2.scripts.KNN import KNN
import torch.nn.functional as functional
#sys.stdout = open(os.devnull, 'w')
from panoptic.epsnet2.epsnet.models.panoptic import PanopticNet_range

parser = argparse.ArgumentParser(description="Panoptic testing script")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("--meta", type=str, help="Path to metadata file of training dataset")
parser.add_argument("--score_threshold", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="Panoptic disambiguation IoU threshold")
parser.add_argument("--min_area", type=float, default=4096, help="Minimum pixel area for stuff predictions")
parser.add_argument("--raw", action="store_true", help="Save raw predictions instead of rendered images")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("model", metavar="FILE", type=str, help="Path to model file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")
parser.add_argument("out_dir", metavar="DIR", type=str, help="Path to output directory")


def mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_config(args):
    log_debug("Loading configuration from %s", args.config)

    conf = load_config(args.config, DEFAULT_CONFIGS["panoptic"])

    log_debug("\n%s", config_to_string(conf))
    return conf

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

def make_dataloader(args, config, rank, world_size):
    config = config["dataloader"]
    log_debug("Creating dataloaders for dataset in %s", args.data)

    # Validation dataloader
    test_tf = ISSTestTransform(config.getint("shortest_size"),
                               config.getstruct("rgb_mean"),
                               config.getstruct("rgb_std"))
    test_db = ISSTestDataset(args.data, test_tf)
    test_sampler = DistributedARBatchSampler(test_db, 1, world_size, rank, False)
    test_dl = data.DataLoader(test_db,
                              batch_sampler=test_sampler,
                              collate_fn=iss_collate_fn,
                              pin_memory=True,
                              num_workers=config.getint("num_workers"))

    return test_dl


def load_meta(meta_file):
    with open(meta_file, "rb") as fid:
        data = umsgpack.load(fid, encoding="utf-8")
        meta = data["meta"]
    return meta


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
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_fn_range = models.__dict__["net_" + 'efficientnet-b0']
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



def test(model, dataloader, **varargs):
    model.eval()
    cfg = yaml.safe_load(open('semantic-kitti-ours.yaml', 'r'))

    dataloader.batch_sampler.set_epoch(0)

    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())
    num_classes=19
    make_panoptic = varargs["make_panoptic"]
    num_stuff = varargs["num_stuff"]
    post = KNN(num_classes)

    save_function = varargs["save_function"]
    out = 'sequences'
    mkdir(out)

    data_time = time.time()
    for it, batch in enumerate(dataloader):
        with torch.no_grad():
            # Extract data
            idxs = batch["idx"]
            npoints = batch["n_points"][0]
            #print(batch['proj_range'][0].shape)
            p_x = batch["p_x"][0][:npoints]  
            p_y = batch["p_y"][0][:npoints]
            proj_range = batch["proj_range"][0][:npoints]  
            unproj_range = batch["unproj_range"][0][:npoints]  
#            proj_in = proj_in.cuda()
#            proj_mask = proj_mask.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()
            batch_sizes = [img.shape[-2:] for img in batch["img"]]
            original_sizes = batch["size"]

            img = batch["img"].cuda(device=varargs["device"], non_blocking=True)

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            _, pred, _ = model(img=img, do_loss=False, do_prediction=True)

            # Update meters
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            for i, (idy, sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits) in enumerate(zip(idxs,
                    pred["sem_pred"], pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"], pred["msk_pred"], pred["sem_logits"])):
                img_info = {
                    "batch_size": batch["img"][i].shape[-2:],
                    "original_size": batch["size"][i],
                    "rel_path": batch["rel_path"][i],
                    "abs_path": batch["abs_path"][i]
                }
                
                # Compute panoptic output
                panoptic_pred = make_panoptic(sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, num_stuff)
                sem_logits = sem_logits.unsqueeze(0) #functional.interpolate(sem_logits.unsqueeze(0), size=(64, 2048), mode="bilinear", align_corners=False)

                predcition_img = Image.fromarray(np.uint8(panoptic_pred[0].numpy()))
#                predcition_img = predcition_img.resize((2048,64), resample=Image.NEAREST)
                predcition_img = np.array(predcition_img, dtype=np.uint32, copy=False)
                cat = panoptic_pred[1].numpy()
                predcition_img, cat = compact_labels(predcition_img, cat)
                cat = torch.from_numpy(cat.astype(np.long))
                predcition_img = torch.from_numpy(predcition_img.astype(np.long))
                panop_logits = panoptic_pred[-1].unsqueeze(0) #functional.interpolate(panoptic_pred[-1].unsqueeze(0), size=(64, 2048), mode="bilinear", align_corners=False)

                proj_argmax = predcition_img.cuda()
                unproj_argmax = post(proj_range, unproj_range, proj_argmax, p_x, p_y, torch.max(proj_argmax)+1)
                unprojsem_argmax = cat[unproj_argmax] #post(proj_range, unproj_range, sem_argmax, p_x, p_y, 19)
                pred_np = unprojsem_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.uint32)
                pred_np = map(pred_np, cfg['learning_map_inv'])
                ################################################## Simple Mapping
                unproj_sem_logits = sem_logits[0, :, p_y, p_x]
                unproj_panoptic_logits = panop_logits[0, :, p_y, p_x]
                #############################################################
#                print (sem_logits.shape, unproj_sem_logits.shape, unproj_panoptic_logits.shape)
                pred_np = np.uint32(unproj_argmax.cpu().numpy().reshape((-1)).astype(np.uint32)<<16 | pred_np)
                ############################################# TO SAVE 
                #path = os.path.join(out,idy.split('_')[0])
                #mkdir(path)
                path = os.path.join(out, 'predictions')
                mkdir(path)

                #path = os.path.join(path,idy.split('_')[1]+'.label')
                path = os.path.join(path,idy+'.label')
                #pred_np.tofile(path)
                ####################################################################
                # Save prediction
#                raw_pred = (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred)
#                save_function(raw_pred, panoptic_pred, img_info)

            # Log batch
            if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
                logging.iteration(
                    None, "val", 0, 1, 1,
                    it + 1, len(dataloader),
                    OrderedDict([
                        ("data_time", data_time_meter),
                        ("batch_time", batch_time_meter)
                    ])
                )

            data_time = time.time()


def ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


def save_prediction_image(_, panoptic_pred, img_info, out_dir, colors, num_stuff):
    msk, cat, obj, iscrowd = panoptic_pred

    img = Image.open(img_info["abs_path"])

    # Prepare folders and paths
    folder, img_name = path.split(img_info["rel_path"])
    img_name, _ = path.splitext(img_name)
    out_dir = path.join(out_dir, folder)
    ensure_dir(out_dir)
    out_path = path.join(out_dir, img_name + ".jpg")

    # Render semantic
    sem = cat[msk].numpy()
    crowd = iscrowd[msk].numpy()
    sem[crowd == 1] = 255

    sem_img = Image.fromarray(colors[sem])
    sem_img = sem_img.resize(img_info["original_size"][::-1])

    # Render contours
    is_background = (sem < num_stuff) | (sem == 255)
    msk = msk.numpy()
    msk[is_background] = 0

    contours = find_boundaries(msk, mode="outer", background=0).astype(np.uint8) * 255
    contours = dilation(contours)

    contours = np.expand_dims(contours, -1).repeat(4, -1)
    contours_img = Image.fromarray(contours, mode="RGBA")
    contours_img = contours_img.resize(img_info["original_size"][::-1])

    # Compose final image and save
    out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
    out = Image.alpha_composite(out, contours_img)
    out.convert(mode="RGB").save(out_path)

def save_prediction_raw(raw_pred, _, img_info, out_dir):
    # Prepare folders and paths
    folder, img_name = path.split(img_info["rel_path"])
    img_name, _ = path.splitext(img_name)
    out_dir = path.join(out_dir, folder)
    ensure_dir(out_dir)
    out_path = path.join(out_dir, img_name + ".pth.tar")

    out_data = {
        "sem_pred": raw_pred[0],
        "bbx_pred": raw_pred[1],
        "cls_pred": raw_pred[2],
        "obj_pred": raw_pred[3],
        "msk_pred": raw_pred[4]
    }
    torch.save(out_data, out_path)


def main(args):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    if rank == 0:
        logging.init(args.log_dir, "test")

    # Load configuration
    config = make_config(args)

    # Create dataloader
    test_dataloader = make_dataloader(args, config, rank, world_size)
    meta = load_meta(args.meta)

    # Create model
    model = make_model(config, meta["num_thing"], meta["num_stuff"])

    # Load snapshot
    log_debug("Loading snapshot from %s", args.model)
    resume_from_snapshot(model, args.model, ["body", "rpn_head", "roi_head", "sem_head"])

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id)

    # Panoptic processing parameters
    panoptic_preprocessing = PanopticPreprocessing(args.score_threshold, args.iou_threshold, args.min_area)

    if args.raw:
        save_function = partial(save_prediction_raw, out_dir=args.out_dir)
    else:
        palette=[]
        for i in range(256):
            if i < len(meta["palette"]):
                palette.append(meta["palette"][i])
            else:
                palette.append((0, 0, 0))
        
        palette = np.array(palette, dtype=np.uint8)

        save_function = partial(
            save_prediction_image, out_dir=args.out_dir, colors=palette, num_stuff=meta["num_stuff"])
    test(model, test_dataloader, device=device, summary=None,
         log_interval=config["general"].getint("log_interval"), save_function=save_function,
         make_panoptic=panoptic_preprocessing, num_stuff=meta["num_stuff"])


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8899'
    main(parser.parse_args())


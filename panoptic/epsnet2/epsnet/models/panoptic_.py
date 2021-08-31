from collections import OrderedDict
import torch.nn.functional as functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from epsnet.utils.sequence import pad_packed_images
from epsnet.utils.parallel import PackedSequence

NETWORK_INPUTS = ["img", "msk", "cat", "iscrowd", "bbx", "track_id", "proj_msk"]


class PanopticNet(nn.Module):
    def __init__(self,
                 body,
                 rpn_head,
                 roi_head,
                 sem_head,
                 rpn_algo,
                 instance_seg_algo,
                 semantic_seg_algo,
                 pan_seg_algo,
                 classes ,
                 msk_predict
                 ):
        super(PanopticNet, self).__init__()
        self.num_stuff = classes["stuff"]

        # Modules
        self.body = body
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.instance_seg_algo = instance_seg_algo
        self.semantic_seg_algo = semantic_seg_algo
        self.pan_seg_algo = pan_seg_algo
        self.msk_predict = msk_predict 

    def _prepare_inputs(self, msk, cat, iscrowd, bbx, track_id):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, track_id_out = [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i, track_i in zip(msk, cat, iscrowd, bbx, track_id):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i>0)
           
            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                track_id_out.append(track_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)
                track_id_out.append(None)
            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i>0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])
        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out, track_id_out

    def forward(self, img, msk=None, cat=None, track_id=None, iscrowd=None, bbx=None,proj_msk=None, do_loss=False, do_prediction=True ):
        # Pad the input images
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]
#        print ('p', proj_msk[0].shape)
        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx, ids, sem, track_id = self._prepare_inputs(msk, cat, iscrowd, bbx, track_id)
#        print ('ss', track_id[0], track_id[1], track_id[2])
            
        # Run network body
        x = self.body(img, None, proj_msk)

        # RPN part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(
                self.rpn_head, x, bbx, iscrowd, valid_size, training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, x, valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = self.instance_seg_algo.training(
                self.roi_head, x, proposals, bbx, cat, iscrowd, ids, msk, track_id, img_size)
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred = self.instance_seg_algo.inference(
                self.roi_head, x, proposals, valid_size, img_size)
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation part
        if do_loss:
            sem_loss, conf_mat, sem_pred, sem_logits = self.semantic_seg_algo.training(self.sem_head, x, sem, valid_size, img_size)
        elif do_prediction:
            sem_pred, sem_logits = self.semantic_seg_algo.inference(self.sem_head, x, valid_size, img_size)
            sem_loss, conf_mat = None, None
        else:
            sem_loss, conf_mat, sem_pred = None, None, None  

        # Prepare outputs
        loss = OrderedDict([
            ("obj_loss", obj_loss),
            ("bbx_loss", bbx_loss),
            ("roi_cls_loss", roi_cls_loss),
            ("roi_bbx_loss", roi_bbx_loss),
            ("roi_msk_loss", roi_msk_loss),
            ("sem_loss", sem_loss),
        ])
        pred = OrderedDict([
            ("bbx_pred", bbx_pred),
            ("cls_pred", cls_pred),
            ("obj_pred", obj_pred),
            ("msk_pred", msk_pred),
            ("sem_pred", sem_pred),
            ("sem_logits", sem_logits)
        ])
        conf = OrderedDict([
            ("sem_conf", conf_mat)
        ])
        return loss, pred, conf


class PanopticNet_range(nn.Module):
    def __init__(self,
                 body,
                 body_range,
                 rpn_head,
                 roi_head,
                 sem_head,
                 rpn_algo,
                 instance_seg_algo,
                 semantic_seg_algo,
                 pan_seg_algo,
                 classes ,
                 msk_predict
                 ):
        super(PanopticNet_range, self).__init__()
        self.num_stuff = classes["stuff"]
#        body_range = None
        # Modules
        self.body = body
        self.body_range = body_range
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.instance_seg_algo = instance_seg_algo
        self.semantic_seg_algo = semantic_seg_algo
        self.pan_seg_algo = pan_seg_algo
        self.msk_predict = msk_predict 

    def _prepare_inputs(self, msk, cat, iscrowd, bbx, track_id):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, track_id_out = [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i, track_i in zip(msk, cat, iscrowd, bbx, track_id):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i>0)
           
            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                track_id_out.append(track_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)
                track_id_out.append(None)
            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i>0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])
        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out, track_id_out

    def forward(self, img, msk=None, cat=None, track_id=None, iscrowd=None, bbx=None, do_loss=False, do_prediction=True, proj_msk=None):
        # Pad the input images
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]
        
        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx, ids, sem, track_id = self._prepare_inputs(msk, cat, iscrowd, bbx, track_id)
#        print ('ss', track_id[0], track_id[1], track_id[2])
            
        # Run network body
        x_range = None
        if self.body_range is not None:
            x_range = self.body_range(img[:,0:1,...])
        x = self.body(img, x_range, proj_msk)
#        print ('sa',x_range['mod4'].shape) 
        # RPN part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(
                self.rpn_head, x, bbx, iscrowd, valid_size, training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, x, valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = self.instance_seg_algo.training(
                self.roi_head, x, proposals, bbx, cat, iscrowd, ids, msk, track_id, img_size)
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred = self.instance_seg_algo.inference(
                self.roi_head, x, proposals, valid_size, img_size)
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation part
        if do_loss:
            sem_loss, conf_mat, sem_pred, sem_logits = self.semantic_seg_algo.training(self.sem_head, x, sem, valid_size, img_size, x_range)
        elif do_prediction:
            sem_pred, sem_logits = self.semantic_seg_algo.inference(self.sem_head, x, valid_size, img_size)
            sem_loss, conf_mat = None, None
        else:
            sem_loss, conf_mat, sem_pred = None, None, None  

        # Prepare outputs
        loss = OrderedDict([
            ("obj_loss", obj_loss),
            ("bbx_loss", bbx_loss),
            ("roi_cls_loss", roi_cls_loss),
            ("roi_bbx_loss", roi_bbx_loss),
            ("roi_msk_loss", roi_msk_loss),
            ("sem_loss", sem_loss),
        ])
        pred = OrderedDict([
            ("bbx_pred", bbx_pred),
            ("cls_pred", cls_pred),
            ("obj_pred", obj_pred),
            ("msk_pred", msk_pred),
            ("sem_pred", sem_pred),
            ("sem_logits", sem_logits)
        ])
        conf = OrderedDict([
            ("sem_conf", conf_mat)
        ])
        return loss, pred, conf



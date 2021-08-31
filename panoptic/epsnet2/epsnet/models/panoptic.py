from collections import OrderedDict
import torch.nn.functional as functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from epsnet.utils.sequence import pad_packed_images
from epsnet.utils.parallel import PackedSequence
from skimage.segmentation import find_boundaries
from epsnet.utils.roi_sampling import roi_sampling
from epsnet.utils.bbx import invert_roi_bbx

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
#            print ('va',len(thing),len(iscrowd_i),len(cat_i))
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
        # img, valid_size = pad_packed_images(img)
        valid_size = []
        for i in range(img.shape[0]):
            valid_size.append(img.shape[2:])
        img_size = img.shape[-2:]
        
        # Convert ground truth to the internal format
        if do_loss:
            cat, iscrowd, bbx, ids, sem, track_id = self._prepare_inputs(msk, cat, iscrowd, bbx, track_id)
#        print ('ss', track_id[0], track_id[1], track_id[2])
  #      img.requires_grad = True
# 
 #       print(img.shape)    
        # Run network body
        x_range = None
        if self.body_range is not None:
            x_range = self.body_range(img[:,0:1,...])
        x = self.body(img, x_range, proj_msk)
         #x = self.body(img, None, proj_msk)
#        x[0].backward()
#        print(img.grad)

#        g = torch.autograd.grad(x[0].sum(), img, retain_graph=True)[0].data
#        print(g)

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

        '''          
            if (cat[0] is not None):
               bbx_pred, cls_pred, obj_pred, msk_pred = self.instance_seg_algo.inference(
                   self.roi_head, x, proposals, valid_size, img_size) 
            msk_boundary=[]
            reward =torch.tensor(0.).cuda()
            if cat[0] is not None:# and 1==2: 
             #print(msk[0].shape)
             #print('tata', proj_msk.contiguous[0].shape,len(img))
             #proj_range1 = ((img[0][0])*12.32 +12.12)*proj_msk.contiguous[0].float()
             for bbx_pred_i, msk_pred_i, pr, pmsk in zip(bbx_pred,msk_pred,img,proj_msk):
                 if msk_pred_i is not None:
                     proj_range1 = (pmsk) #*12.32+12.12)*pmsk.float()
                     #print(proj_range1.shape)  
                     bbx_inv1 = invert_roi_bbx(bbx_pred_i, list(msk_pred_i.shape[-2:]), (64,2048))
                     bbx_idx = torch.arange(0, msk_pred_i.size(0), dtype=torch.long, device=msk_pred_i.device)

                     msk_pred1 = roi_sampling(msk_pred_i.unsqueeze(1), bbx_inv1, bbx_idx, tuple(list([64,2048])), padding="zero")
                     msk_pred1 = msk_pred1.sigmoid()>0.5

                     msk_boundaries = np.zeros(msk_pred1.shape)
                     #print(msk_boundaries[0][0])
                     msk_pred11 = msk_pred1.detach().cpu().numpy().astype(np.uint8)
                     neighbors =[]
                     reward =torch.tensor(0.).cuda()
                     sum_=0
                     for k in range(msk_boundaries.shape[0]):
                        msk_boundaries[k][0] = find_boundaries(msk_pred11[k][0],mode='inner').astype(np.uint8)
                        invt1 = (1 - msk_pred1[k][0]).float() 
                        #invt = 1 - msk_pred11[k][0]
                        points = np.where( msk_boundaries[k][0]>0)
                        #print(points[0].shape)
                        px1 = points[0]+1
                        c = np.where(px1 >63)
                        px1[c] = 63

                        if(px1.shape[0]>0):
                            pxm1 = points[0]-1
                            c = np.where(pxm1 <0)
                            pxm1[c] = 0
                            py1 = points[1]+1
                            c = np.where(py1 >2047)
                            py1[c] = 2047

                            pym1 = points[1]-1
                            c = np.where(pxm1 <0)
                            pxm1[c] = 0
                            loss = torch.zeros((px1.shape[0],8))
                        #print(invt1[px1,py1].dtype,(proj_range1[points[0],points[1]]-proj_range1[px1,py1]).dtype)
                            loss[:,0] =  (invt1[px1,py1].float())*(proj_range1[points[0],points[1]]-proj_range1[px1,py1])**2
                            loss[:,1] =  invt1[pxm1,py1]*(proj_range1[points[0],points[1]]-proj_range1[pxm1,py1])**2
                            loss[:,2] =  invt1[pxm1,pym1]*(proj_range1[points[0],points[1]]-proj_range1[pxm1,pym1])**2
                            loss[:,3] =  invt1[px1,pym1]*(proj_range1[points[0],points[1]]-proj_range1[px1,pym1])**2

                            loss[:,4] =  invt1[px1,points[1]]*(proj_range1[points[0],points[1]]-proj_range1[px1,points[1]])**2
                            loss[:,5] =  invt1[pxm1,points[1]]*(proj_range1[points[0],points[1]]-proj_range1[pxm1,points[1]])**2
                            loss[:,6] =  invt1[points[0],py1]*(proj_range1[points[0],points[1]]-proj_range1[points[0],py1])**2
                            loss[:,7] =  invt1[points[0],pym1]*(proj_range1[points[0],points[1]]-proj_range1[points[0],pym1])**2

                            #print((torch.max(loss,0).values).shape)
                            main_loss = torch.sum(torch.max(loss,1).values)
                            sum_+=px1.shape[0]
                            reward= reward + (main_loss)#/px1.shape[0])

                     if msk_boundaries.shape[0] >0 and sum_!=0:
                        reward = reward/sum_    

             #print('r22', reward)    
             if reward.item()!=0:
                reward = -0.1*reward.cuda()
        '''




#        elif do_prediction:
#            proposals = self.rpn_algo.inference(self.rpn_head, x, valid_size, self.training)
#            obj_loss, bbx_loss = None, None
#        else:
#            obj_loss, bbx_loss, proposals = None, None, None

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
            sem_pred, sem_logits = self.semantic_seg_algo.inference(self.sem_head, x, valid_size, img_size, x_range)
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
#            ("reward", reward),
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



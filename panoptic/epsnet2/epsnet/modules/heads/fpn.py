from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN
import numpy as np
from epsnet.utils.misc import try_index
from epsnet.models.utils import (
    Swish,
    MemoryEfficientSwish,
)


class FPNROIHead(nn.Module):
    """ROI head module for FPN

    Parameters
    ----------
    in_channels : int
        Number of input channels
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    roi_size : tuple of int
        `(height, width)` of the ROIs extracted from the input feature map, these will be average-pooled 2x2 before
        feeding to the rest of the head
    hidden_channels : int
        Number of channels in the hidden layers
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, classes, roi_size, hidden_channels=1024, norm_act=ABN):
        super(FPNROIHead, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), hidden_channels, bias=False)),
            ("bn1", norm_act(hidden_channels)),
            ("fc2", nn.Linear(hidden_channels, hidden_channels, bias=False)),
            ("bn2", norm_act(hidden_channels)),
        ]))
        self.roi_cls = nn.Linear(hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(hidden_channels, classes["thing"] * 4)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.fc.bn1.activation, self.fc.bn1.activation_param)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "roi_cls" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x):
        """ROI head module for FPN

        Parameters
        ----------
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W

        Returns
        -------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape S x (num_thing + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape S x num_thing x 4
        """
        x = functional.avg_pool2d(x, 2)

        # Run head
        x = self.fc(x.view(x.size(0), -1))
        return self.roi_cls(x), self.roi_bbx(x).view(x.size(0), -1, 4)


class FPNMaskHead(nn.Module):
    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(FPNMaskHead._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation , padding=dilation, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
 
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x 


    """ROI head module for FPN

    Parameters
    ----------
    in_channels : int
        Number of input channels
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    roi_size : tuple of int
        `(height, width)` of the ROIs extracted from the input feature map, these will be average-pooled 2x2 before
        feeding to the fully-connected branch
    fc_hidden_channels : int
        Number of channels in the hidden layers of the fully-connected branch
    conv_hidden_channels : int
        Number of channels in the hidden layers of the convolutional branch
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, classes, roi_size, fc_hidden_channels=1024, conv_hidden_channels=256, norm_act=ABN):
        super(FPNMaskHead, self).__init__()

        # ROI section
        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), fc_hidden_channels, bias=False)),
            ("bn1", norm_act(fc_hidden_channels)),
            ("fc2", nn.Linear(fc_hidden_channels, fc_hidden_channels, bias=False)),
            ("bn2", norm_act(fc_hidden_channels)),
        ]))
        self.roi_cls = nn.Linear(fc_hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(fc_hidden_channels, classes["thing"] * 4)

        # Mask section

        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn1", norm_act(conv_hidden_channels)),
            ("conv2", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn2", norm_act(conv_hidden_channels)),
            ("conv3", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn3", norm_act(conv_hidden_channels)),
            ("conv4", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn4", norm_act(conv_hidden_channels)),
            ("conv_up", nn.ConvTranspose2d(conv_hidden_channels, conv_hidden_channels, 2, stride=2, bias=False)),
            ("bn_up", norm_act(conv_hidden_channels)),
        ]))

        
        self.roi_msk = nn.Conv2d(conv_hidden_channels, classes["thing"], 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.fc.bn1.activation, self.fc.bn1.activation_param)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
                if "roi_cls" in name or "roi_msk" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x, do_cls_bbx=True, do_msk=True):
        """ROI head module for FPN

        Parameters
        ----------
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W
        do_cls_bbx : bool
            Whether to compute or not the class and bounding box regression predictions
        do_msk : bool
            Whether to compute or not the mask predictions

        Returns
        -------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape S x (num_thing + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape S x num_thing x 4
        msk_logits : torch.Tensor
            A tensor of class-specific mask logits with shape S x num_thing x (H_roi * 2) x (W_roi * 2)
        """
        # Run fully-connected head
        if do_cls_bbx:
            x_fc = functional.avg_pool2d(x, 2)
            x_fc = self.fc(x_fc.view(x_fc.size(0), -1))

            cls_logits = self.roi_cls(x_fc)
            bbx_logits = self.roi_bbx(x_fc).view(x_fc.size(0), -1, 4)
        else:
            cls_logits = None
            bbx_logits = None

        # Run convolutional head
        if do_msk:
            x = self.conv(x)
            msk_logits = self.roi_msk(x)
        else:
            msk_logits = None

        return cls_logits, bbx_logits, msk_logits

class FPNSemanticHeadDPC(nn.Module):
    """Semantic segmentation head for FPN-style networks, extending DPC for FPN bodies"""
    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(FPNSemanticHeadDPC._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation , padding=dilation, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
 
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x 

    class _3x3box(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act):
            super(FPNSemanticHeadDPC._3x3box, self).__init__()
           
            self.conv1_3x3_1 = seperable_conv(in_channels, out_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(out_channels) 
            self.conv1_3x3_2 = seperable_conv(out_channels, out_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(out_channels)
            
        def forward(self, x):
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))      
            return x
           
    class _DPC(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act, include_range=False):
            super(FPNSemanticHeadDPC._DPC, self).__init__()
            self.include_range = include_range

            self.conv1_3x3_1 = seperable_conv(in_channels, in_channels, (1,6), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(in_channels) 
            self.conv1_3x3_2 = seperable_conv(in_channels, in_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(in_channels)  
            self.conv1_3x3_3 = seperable_conv(in_channels, in_channels, (6,21), norm_act, bias=False)
            self.conv1_3x3_3_bn = norm_act(in_channels) 
            self.conv1_3x3_4 = seperable_conv(in_channels, in_channels, (18,15), norm_act, bias=False)
            self.conv1_3x3_4_bn = norm_act(in_channels)  
            self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6,3), norm_act, bias=False)
            self.conv1_3x3_5_bn = norm_act(in_channels)  
            
            total_in_channels = in_channels*5
            if include_range:
                total_in_channels = in_channels * 6
                self.range_in = Parameter(torch.Tensor(
                                    in_channels, 1, *(3,3)))
                init.kaiming_uniform_(self.range_in, a=math.sqrt(5))
                self.range_out = Parameter(torch.Tensor(
                                    in_channels, 1, *(3,3)))
                init.kaiming_uniform_(self.range_out, a=math.sqrt(5))
                self.learn_lambda = Parameter(torch.Tensor(
                                    1, 1, *(1,1))) 
                init.constant_(self.learn_lambda, 1.0)
                
                #self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6,3), norm_act, bias=False)
                #self.conv1_3x3_range_out_bn = norm_act(in_channels)  
                #self.conv1_3x3_range_in_bn = norm_act(in_channels)  
                self.reduce_range_in = nn.Conv2d(in_channels+9, in_channels//2, 1, bias=False)
                self.reduce_range_in_bn = norm_act(in_channels//2)
                self.reduce_range_out = nn.Conv2d(in_channels+9, in_channels//2, 1, bias=False)
                self.reduce_range_out_bn = norm_act(in_channels//2)

            self.conv2 = nn.Conv2d(total_in_channels, out_channels, 1, bias=False)
            self.bn2 = norm_act(out_channels)
#            self._swish = MemoryEfficientSwish()

        def forward(self, x, x_range=None):
            if self.include_range:
                offset = 3.0*self.learn_lambda*torch.tanh(1/x_range)
                #print ('lambda', self.learn_lambda, self.learn_lambda.grad, np.min(offset.detach().cpu().numpy()),np.max(offset.detach().cpu().numpy()))
                x_off = torch.from_numpy(np.array([-1,0,1,-1,0,1,-1,0,1])).cuda(x.device)
                y_off = torch.from_numpy(np.array([-1,-1,-1,0,0,0,1,1,1])).cuda(x.device)
                v, _ = torch.meshgrid([torch.arange(0,(x.shape[-2]+2)*(x.shape[-1]+2)), torch.arange(0,9)])
                v = v.cuda(x.device) 
                pre_v = v+x_off+(y_off*x.shape[-1]+2)
                pre_v = pre_v.view(x.shape[-2]+2,x.shape[-1]+2,9)[1:x.shape[-2]+1,1:x.shape[-1]+1].contiguous().view(-1,9)
                pre_v = torch.cat(x.shape[0]*[pre_v.unsqueeze(0)],0)
                after_v = torch.round(pre_v.float()+offset.view(x.shape[0],x.shape[-1]*x.shape[-2],9)).long()
                #print ('prev_v', torch.round(pre_v.float()+offset.view(x.shape[0],x.shape[-1]*x.shape[-2],9)))
 #               after_v[:,:, 4] = pre_v[:,:,4]
                after_v = torch.min(torch.ones(after_v.shape).long().cuda(after_v.device)*((x.shape[-2]+2)*(x.shape[-1]+2)-1),torch.max(torch.zeros(after_v.shape).long().cuda(after_v.device), after_v)) 
#                print ('prev_v', after_v[0,32,3])
                x_padded = functional.pad(x, (1,1,1,1), mode='constant', value=0.5)
                #print ('sha', x_padded.shape)
                
                after_v = torch.cat(x.shape[1]*[after_v.view(-1,x.shape[-1]*x.shape[-2]*9).unsqueeze(1)],1)
                
                x_padded = torch.gather(x_padded.view(x_padded.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
                x_padded = torch.sum(x_padded.view(x_padded.shape[0],x.shape[1],x.shape[-2]*x.shape[-1],9)*self.range_out.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+self.learn_lambda/x_range
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
                x_padded = self.reduce_range_out(torch.cat((x_padded,offset), dim=1))
                x_padded = self.reduce_range_out_bn(x_padded)
#                print ('xp', np.min(x_padded.detach().cpu().numpy()),np.max(x_padded.detach().cpu().numpy()))
                
                #print ('done_1', x_padded.shape, x_padded.device)
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))

            if self.include_range:
                x_padded_ = functional.pad(x, (1,1,1,1), mode='constant', value=0)
#                print ('now_in', x_padded_.shape, after_v.shape)
                x_padded_ = torch.gather(x_padded_.view(x_padded_.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
                x_padded_ = torch.sum(x_padded_.view(x_padded_.shape[0],x.shape[1],x.shape[-2]*x.shape[-1],9)*self.range_in.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+(self.learn_lambda/x_range)
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
                x_padded_ = self.reduce_range_in(torch.cat((x_padded_,offset), dim=1))
                x_padded_ = self.reduce_range_in_bn(x_padded_)
                #print ('done_1', x_padded.shape, x_padded.device)

            x1 = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            x2 = self.conv1_3x3_3_bn(self.conv1_3x3_3(x))
            x3 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x))
            x4 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x3))    
            x = torch.cat([
                x,
                x1,
                x2,
                x3,
                x4,
            ], dim=1)
            if self.include_range:
                x = torch.cat([
                    x,          
                    x_padded,
                    x_padded_, 
                    ], dim=1)
                #print (x.shape)
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def __init__(self,
                 in_channels,
                 min_level,
                 levels,
                 num_classes,
                 hidden_channels=128,
                 dilation=0,
                 pooling_size=(64, 64),
                 norm_act=ABN,
                 interpolation="bilinear",):
        super(FPNSemanticHeadDPC, self).__init__()
        self.min_level = min_level
        self.levels = levels
        self.interpolation = interpolation
        self.include_range = False
        if dilation == 0:
            self.include_range = True  
#            print ('aya')
            self.lateral = nn.ModuleList([
                self._make_lateral(channels, 9, norm_act) for channels in [640,56]
                ])
        self.output_1 = nn.ModuleList([
            self._DPC(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act, self.include_range) for _ in range(levels-2)
        ])
        self.output_2 = nn.ModuleList([
            self._3x3box(self._seperable_conv ,in_channels, hidden_channels, dilation, norm_act) for _ in range(2)
        ])
        self.pre_process = nn.ModuleList([
            self._3x3box(self._seperable_conv ,128, 128, dilation, norm_act) for _ in range(2)
        ])  
        self.conv_sem = nn.Conv2d(hidden_channels * levels, num_classes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.output_1[0].bn2.activation,self.output_1[0].bn2.activation_param)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Conv2d):
                if "conv_sem" not in name:
                    nn.init.xavier_normal_(mod.weight, gain)
                else:
                    nn.init.xavier_normal_(mod.weight, .1)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)
    @staticmethod
    def _make_lateral(input_channels, hidden_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 1, bias=False)),
            ("bn", norm_act(hidden_channels))
        ]))
       
    def forward(self, xs, xs_range=None):
        xs = xs[self.min_level:self.min_level + self.levels]
 
        ref_size = xs[0].shape[-2:]
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False
      
        i = self.min_level + self.levels - 1
        js = 0 
        if self.include_range:   
         xs_range = [xs_range[inp] for inp in xs_range]
         #print (np.max(xs_range[i].detach().cpu().numpy()))
         for output,lateral in zip(self.output_1, self.lateral):
#            print ('aa',xs_range[i].shape)   
            xs[i] = output(xs[i], lateral(xs_range[i]))
#            if lateral.conv.weight.grad is not None:
#                print (lateral.conv.weight.grad[3,3,0,0])
         
            i = i - 1
        else:
         for output in self.output_1:
            xs[i] = output(xs[i])
            i = i - 1
         
        interm = self.pre_process[js](xs[i+1] + functional.interpolate(xs[i+2], size=xs[i+1].shape[-2:], **interp_params))
        for output in self.output_2:
            xs[i] = output(xs[i])
            if js==1:
                interm = self.pre_process[js](xs[i+1])
           
            xs[i] = xs[i] + functional.interpolate(interm, size=xs[i].shape[-2:], **interp_params)
            js += 1
            i = i - 1
        for i in range(self.min_level, self.min_level + self.levels):
            if i > 0:
                xs[i] = functional.interpolate(xs[i], size=ref_size, **interp_params)
 
        xs = torch.cat(xs, dim=1)
        xs = self.conv_sem(xs)
#        print (xs.shape) 
        return xs

class FPNSemanticHeadDPCR(nn.Module):
    """Semantic segmentation head for FPN-style networks, extending DPC for FPN bodies"""
    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(FPNSemanticHeadDPCR._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation , padding=dilation, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
 
        def forward(self, x):

            x = self.depthwise(x)
            x = self.pointwise(x)
            return x 

    class _3x3box(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act, include_range = False):
            super(FPNSemanticHeadDPCR._3x3box, self).__init__()
            self.include_range = include_range

            if include_range:

                self.range_out = Parameter(torch.Tensor(
                                    in_channels, 1, *(3,3)))
                init.kaiming_uniform_(self.range_out, a=math.sqrt(5))
                self.reduce_range_out = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.reduce_range_out_bn = norm_act(out_channels)
                in_channels = out_channels
            #else:    
            self.conv1_3x3_1 = seperable_conv(in_channels, out_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(out_channels) 
            self.conv1_3x3_2 = seperable_conv(out_channels, out_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(out_channels)
        def clamp(self,after_v, x_padded):
            after_v = torch.min(torch.ones(after_v.shape).cuda(after_v.device)*((x_padded.shape[-2])*(x_padded.shape[-1])-1),torch.max(torch.zeros(after_v.shape).cuda(after_v.device), after_v))
            return after_v

        def get_val(self,x_padded, after_v):
            B, C, H, W = x_padded.shape[0],x_padded.shape[1],x_padded.shape[-2], x_padded.shape[-1]
            after_v = torch.cat(x_padded.shape[1]*[after_v.view(-1,(x_padded.shape[-1]-2)*(x_padded.shape[-2]-2)*9).unsqueeze(1)],1).long()

            x_padded = torch.gather(x_padded.view(x_padded.shape[0],x_padded.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
            #print ('x', x_padded.shape, (H-2),(W-2))
            return x_padded.view(B,C,(H-2)*(W-2),9)
                    
        def forward(self, x, x_range=None):
          
            if self.include_range and x_range is not None:
                offset = 3*x_range.sigmoid() #0*((x_range-x_range.min())/(x_range.max()-x_range.min())) #[:,0:1,...], x_range[:,1:2,...]
#                torch.set_printoptions(profile="full")
#                print ('offset', offset) 
                B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
                #print ('off', torch.max(offset), torch.min(offset))  
          #      print ('lambda', self.learn_lambda, self.learn_lambda.grad, np.min(offset.detach().cpu().numpy()),np.max(offset.detach().cpu().numpy()))
                x_off = torch.from_numpy(np.array([-1,0,1,-1,0,1,-1,0,1])).float().cuda(x.device)
                y_off = torch.from_numpy(np.array([-1,-1,-1,0,0,0,1,1,1])).float().cuda(x.device)
                v, _ = torch.meshgrid([torch.arange(0,(x.shape[-2]+2)*(x.shape[-1]+2)), torch.arange(0,9)])
                v = v.float().cuda(x.device) 
                pre_v = v+x_off+(y_off*(x.shape[-1]+2))
                offset_v = torch.matmul(offset.view(B,H*W,1),x_off.view(1,9))+ y_off.view(1,9)*(x.shape[-1]+2) #torch.matmul(offset_y.view(B,H*W,1),y_off.view(1,9)*(x.shape[-1]+2))
                pre_v = pre_v.view(x.shape[-2]+2,x.shape[-1]+2,9)[1:x.shape[-2]+1,1:x.shape[-1]+1].contiguous().view(-1,9)
                pre_v = torch.cat(x.shape[0]*[pre_v.unsqueeze(0)],0).float()
                x_padded = functional.pad(x, (1,1,1,1), mode='constant')
                after_v = pre_v+offset_v
                after_v_f = self.clamp(pre_v+torch.floor(offset_v),x_padded)
                after_v_f1 = self.clamp(after_v_f+x_off,x_padded)
                after_v_c = self.clamp(pre_v+torch.ceil(offset_v),x_padded)
                after_v_c1 = self.clamp(after_v_c+x_off,x_padded)
                val_f = self.get_val(x_padded,after_v_f)
                val_f1 = self.get_val(x_padded,after_v_f1)
                val_c = self.get_val(x_padded,after_v_c)
                val_c1 = self.get_val(x_padded,after_v_c1)
                #print ('f', torch.max((torch.abs(after_v_c-after_v)-x_padded.shape[-1]))  , torch.min((torch.abs(after_v_c-after_v)-x_padded.shape[-1])))
                f1 = (torch.abs(after_v-after_v_f)).unsqueeze(1)*val_f +(torch.abs(after_v_f1-after_v)).unsqueeze(1)*val_f1
                #print ('s', f1.shape)   
                f2 = (torch.abs(after_v-after_v_c1)).unsqueeze(1)*val_c1 +(torch.abs(after_v_c-after_v)).unsqueeze(1)*val_c
                x_padded = torch.abs((after_v-after_v_f)/x_padded.shape[-1]).unsqueeze(1)*f1 +torch.abs((after_v_c1-after_v)/(x_padded.shape[-1])).unsqueeze(1)*f2
                
                #print ('prev_v', torch.round(pre_v.float()+offset.view(x.shape[0],x.shape[-1]*x.shape[-2],9)))
 #               after_v[:,:, 4] = pre_v[:,:,4]
                #after_v = torch.min(torch.ones(after_v.shape).long().cuda(after_v.device)*((x.shape[-2]+2)*(x.shape[-1]+2)-1),torch.max(torch.zeros(after_v.shape).long().cuda(after_v.device), after_v)) 
#                print ('prev_v', after_v[0,32,3])
                #print ('sha', x_padded.shape)
                
                #after_v = torch.cat(x.shape[1]*[after_v.view(-1,x.shape[-1]*x.shape[-2]*9).unsqueeze(1)],1)
                
                #x_padded = torch.gather(x_padded.view(x_padded.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
                x_padded = torch.sum(x_padded*self.range_out.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+self.learn_lambda/x_range
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
                x_padded = self.reduce_range_out(x_padded)
                x = self.reduce_range_out_bn(x_padded)
#                print ('xp', np.min(x_padded.detach().cpu().numpy()),np.max(x_padded.detach().cpu().numpy()))
            #print (x.shape)
            
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))      
            return x
           
    class _DPC(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act, include_range=False, t=True):
            super(FPNSemanticHeadDPCR._DPC, self).__init__()
            self.include_range = include_range

            self.conv1_3x3_1 = seperable_conv(in_channels, in_channels, (1,6), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(in_channels) 
            self.conv1_3x3_2 = seperable_conv(in_channels, in_channels, (1,1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(in_channels)  
            self.conv1_3x3_3 = seperable_conv(in_channels, in_channels, (6,21), norm_act, bias=False)
            self.conv1_3x3_3_bn = norm_act(in_channels) 
            self.conv1_3x3_4 = seperable_conv(in_channels, in_channels, (18,15), norm_act, bias=False)
            self.conv1_3x3_4_bn = norm_act(in_channels)  
            self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6,3), norm_act, bias=False)
            self.conv1_3x3_5_bn = norm_act(in_channels)  
            
            total_in_channels = in_channels*5
            if include_range:
                total_in_channels = in_channels * 6 
                self.range_out = Parameter(torch.Tensor(
                                    in_channels, 1, *(3,3)))
                init.kaiming_uniform_(self.range_out, a=math.sqrt(5))
                #self.learn_lambda = Parameter(torch.Tensor(
                #                    1, 1, *(1,1))) 
                #init.constant_(self.learn_lambda, 1.0)
                
                #self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6,3), norm_act, bias=False)
                #self.conv1_3x3_range_out_bn = norm_act(in_channels)  
                #self.conv1_3x3_range_in_bn = norm_act(in_channels) 
                if t: 
                    self.range_in = Parameter(torch.Tensor(
                                    in_channels, 1, *(3,3)))
                    init.kaiming_uniform_(self.range_in, a=math.sqrt(5))
                
                    self.reduce_range_in = nn.Conv2d(in_channels, in_channels//2, 1, bias=False)
                    self.reduce_range_in_bn = norm_act(in_channels//2)
                    self.reduce_range_out = nn.Conv2d(in_channels, in_channels//2, 1, bias=False)
                    self.reduce_range_out_bn = norm_act(in_channels//2)
                else:
                    self.reduce_range_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
                    self.reduce_range_out_bn = norm_act(in_channels)

            self.conv2 = nn.Conv2d(total_in_channels, out_channels, 1, bias=False)
            self.bn2 = norm_act(out_channels)
#            self._swish = MemoryEfficientSwish()
        def clamp(self,after_v, x_padded):
            after_v = torch.min(torch.ones(after_v.shape).cuda(after_v.device)*((x_padded.shape[-2])*(x_padded.shape[-1])-1),torch.max(torch.zeros(after_v.shape).cuda(after_v.device), after_v))
            return after_v
        def get_val(self,x_padded, after_v):
            B, C, H, W = x_padded.shape[0],x_padded.shape[1],x_padded.shape[-2], x_padded.shape[-1]
            after_v = torch.cat(x_padded.shape[1]*[after_v.view(-1,(x_padded.shape[-1]-2)*(x_padded.shape[-2]-2)*9).unsqueeze(1)],1).long()
                
            x_padded = torch.gather(x_padded.view(x_padded.shape[0],x_padded.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
            #print ('x', x_padded.shape, (H-2),(W-2))
            return x_padded.view(B,C,(H-2)*(W-2),9)
        def forward(self, x, x_range=None, in_=True, r=[6,24]):
            if self.include_range:
#                offset = 3*x_range.sigmoid() #torch.clamp(1/x_range,0,24) #((x_range-x_range.min())/(x_range.max()-x_range.min())) #[:,0:1,...], x_range[:,1:2,...]
                offset_y = r[0]*x_range.sigmoid()
                offset_x = r[1]*x_range.sigmoid()
#                torch.set_printoptions(profile="full")
#                if offset.shape[2]==8:
#                    print ('offset', offset[0,0,:,:])
                B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
                #print ('off', torch.max(offset), torch.min(offset))  
          #      print ('lambda', self.learn_lambda, self.learn_lambda.grad, np.min(offset.detach().cpu().numpy()),np.max(offset.detach().cpu().numpy()))
                x_off = torch.from_numpy(np.array([-1,0,1,-1,0,1,-1,0,1])).float().cuda(x.device)
                y_off = torch.from_numpy(np.array([-1,-1,-1,0,0,0,1,1,1])).float().cuda(x.device)
                v, _ = torch.meshgrid([torch.arange(0,(x.shape[-2]+2)*(x.shape[-1]+2)), torch.arange(0,9)])
                v = v.float().cuda(x.device) 
                pre_v = v+x_off+(y_off*(x.shape[-1]+2))
                #offset_v = torch.matmul(offset.view(B,H*W,1),x_off.view(1,9))+ y_off.view(1,9)*(x.shape[-1]+2) #torch.matmul(offset_y.view(B,H*W,1),y_off.view(1,9)*(x.shape[-1]+2))
                offset_v = torch.matmul(offset_x.view(B,H*W,1),x_off.view(1,9))+ torch.matmul(offset_y.view(B,H*W,1),y_off.view(1,9)*(x.shape[-1]+2)) #torch.matmul(offset_y.view(B,H*W,1),y_off.view(1,9)*(x.shape[-1]+2))
                pre_v = pre_v.view(x.shape[-2]+2,x.shape[-1]+2,9)[1:x.shape[-2]+1,1:x.shape[-1]+1].contiguous().view(-1,9)
                pre_v = torch.cat(x.shape[0]*[pre_v.unsqueeze(0)],0).float()
                x_padded = functional.pad(x, (1,1,1,1), mode='constant')
                after_v = pre_v+offset_v
                after_v_f = self.clamp(pre_v+torch.floor(offset_v),x_padded)
                after_v_f1 = self.clamp(after_v_f+x_off,x_padded)
                after_v_c = self.clamp(pre_v+torch.ceil(offset_v),x_padded)
                after_v_c1 = self.clamp(after_v_c+x_off,x_padded)
                val_f = self.get_val(x_padded,after_v_f)
                val_f1 = self.get_val(x_padded,after_v_f1)
                val_c = self.get_val(x_padded,after_v_c)
                val_c1 = self.get_val(x_padded,after_v_c1)
                #print ('f', torch.max((torch.abs(after_v_c-after_v)-x_padded.shape[-1]))  , torch.min((torch.abs(after_v_c-after_v)-x_padded.shape[-1])))
                f1 = (torch.abs(after_v-after_v_f)).unsqueeze(1)*val_f +(torch.abs(after_v_f1-after_v)).unsqueeze(1)*val_f1
                #print ('s', f1.shape)   
                f2 = (torch.abs(after_v-after_v_c1)).unsqueeze(1)*val_c1 +(torch.abs(after_v_c-after_v)).unsqueeze(1)*val_c
                x_padded = torch.abs((after_v-after_v_f)/x_padded.shape[-1]).unsqueeze(1)*f1 +torch.abs((after_v_c1-after_v)/(x_padded.shape[-1])).unsqueeze(1)*f2
                
                #print ('prev_v', torch.round(pre_v.float()+offset.view(x.shape[0],x.shape[-1]*x.shape[-2],9)))
 #               after_v[:,:, 4] = pre_v[:,:,4]
                #after_v = torch.min(torch.ones(after_v.shape).long().cuda(after_v.device)*((x.shape[-2]+2)*(x.shape[-1]+2)-1),torch.max(torch.zeros(after_v.shape).long().cuda(after_v.device), after_v)) 
#                print ('prev_v', after_v[0,32,3])
                #print ('sha', x_padded.shape)
                
                #after_v = torch.cat(x.shape[1]*[after_v.view(-1,x.shape[-1]*x.shape[-2]*9).unsqueeze(1)],1)
                
                #x_padded = torch.gather(x_padded.view(x_padded.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
                x_padded = torch.sum(x_padded*self.range_out.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+self.learn_lambda/x_range
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
                x_padded = self.reduce_range_out(x_padded)
                x_padded1 = self.reduce_range_out_bn(x_padded)
#                print ('xp', np.min(x_padded.detach().cpu().numpy()),np.max(x_padded.detach().cpu().numpy()))
                
                #print ('done_1', x_padded.shape, x_padded.device)
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))

            if self.include_range and in_:
                x_padded = functional.pad(x, (1,1,1,1), mode='constant')
                after_v = pre_v+offset_v
                after_v_f = self.clamp(pre_v+torch.floor(offset_v),x_padded)
                after_v_f1 = self.clamp(after_v_f+x_off,x_padded)
                after_v_c = self.clamp(pre_v+torch.ceil(offset_v),x_padded)
                after_v_c1 = self.clamp(after_v_c+x_off,x_padded)
                val_f = self.get_val(x_padded,after_v_f)
                val_f1 = self.get_val(x_padded,after_v_f1)
                val_c = self.get_val(x_padded,after_v_c)
                val_c1 = self.get_val(x_padded,after_v_c1)
                #print ('f', torch.max((torch.abs(after_v_c-after_v)-x_padded.shape[-1]))  , torch.min((torch.abs(after_v_c-after_v)-x_padded.shape[-1])))
                f1 = (torch.abs(after_v-after_v_f)).unsqueeze(1)*val_f +(torch.abs(after_v_f1-after_v)).unsqueeze(1)*val_f1
                #print ('s', f1.shape)   
                f2 = (torch.abs(after_v-after_v_c1)).unsqueeze(1)*val_c1 +(torch.abs(after_v_c-after_v)).unsqueeze(1)*val_c
                x_padded = torch.abs((after_v-after_v_f)/x_padded.shape[-1]).unsqueeze(1)*f1 +torch.abs((after_v_c1-after_v)/(x_padded.shape[-1])).unsqueeze(1)*f2
                
                #print ('prev_v', torch.round(pre_v.float()+offset.view(x.shape[0],x.shape[-1]*x.shape[-2],9)))
 #               after_v[:,:, 4] = pre_v[:,:,4]
                #after_v = torch.min(torch.ones(after_v.shape).long().cuda(after_v.device)*((x.shape[-2]+2)*(x.shape[-1]+2)-1),torch.max(torch.zeros(after_v.shape).long().cuda(after_v.device), after_v)) 
#                print ('prev_v', after_v[0,32,3])
                #print ('sha', x_padded.shape)
                
                #after_v = torch.cat(x.shape[1]*[after_v.view(-1,x.shape[-1]*x.shape[-2]*9).unsqueeze(1)],1)
                
                #x_padded = torch.gather(x_padded.view(x_padded.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
                x_padded = torch.sum(x_padded*self.range_in.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+self.learn_lambda/x_range
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
                x_padded = self.reduce_range_in(x_padded)
                x_padded2 = self.reduce_range_in_bn(x_padded)
                
#                x_padded_ = functional.pad(x, (1,1,1,1), mode='constant', value=0)
#                print ('now_in', x_padded_.shape, after_v.shape)
            #    x_padded_ = torch.gather(x_padded_.view(x_padded_.shape[0],x.shape[1],-1), 2, after_v) #.view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1])
            #    x_padded_ = torch.sum(x_padded_.view(x_padded_.shape[0],x.shape[1],x.shape[-2]*x.shape[-1],9)*self.range_in.view(1,x.shape[1],1,9),-1).view(x_padded.shape[0],x.shape[1],x.shape[-2],x.shape[-1]).cuda(x.device) #+(self.learn_lambda/x_range)
                #x_padded = self.conv1_3x3_range_out_bn(x_padded)
            #    x_padded_ = self.reduce_range_in(x_padded_)
            #    x_padded_ = self.reduce_range_in_bn(x_padded_)
                #print ('done_1', x_padded.shape, x_padded.device)

            x1 = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            x2 = self.conv1_3x3_3_bn(self.conv1_3x3_3(x))
            x3 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x))
            x4 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x3))    
            x = torch.cat([
                x,
                x1,
                x2,
                x3,
                x4,
            ], dim=1)
            if self.include_range and in_:
                x = torch.cat([
                    x,          
                    x_padded1,
                    x_padded2], dim=1)
                
                #print (x.shape)
            else:
                x = torch.cat([
                    x,          
                    x_padded1], dim=1)
                
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def __init__(self,
                 in_channels,
                 min_level,
                 levels,
                 num_classes,
                 hidden_channels=128,
                 dilation=0,
                 pooling_size=(64, 64),
                 norm_act=ABN,
                 interpolation="bilinear",):
        super(FPNSemanticHeadDPCR, self).__init__()
        self.min_level = min_level
        self.levels = levels
        self.interpolation = interpolation
        self.include_range = False
        if dilation == 0:
            self.include_range = True  
#            print ('aya')
            self.lateral = nn.ModuleList([
                self._make_lateral(channels, 1, norm_act) for channels in [40,24,16,8]    #[320,32,16,8]
                ])
#        print (self.include_range)
        self.output_1 = nn.ModuleList([
            self._DPC(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act, self.include_range, t) for _,t in zip(range(levels-2),[False, False])
        ])
        self.output_2 = nn.ModuleList([
            self._3x3box(self._seperable_conv ,in_channels, hidden_channels, dilation, norm_act, t) for _,t in zip(range(2),[True,False])
        ])
        self.pre_process = nn.ModuleList([
            self._3x3box(self._seperable_conv ,128, 128, dilation, norm_act) for _ in range(2)
        ])  
        self.conv_sem = nn.Conv2d(hidden_channels * levels, num_classes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.output_1[0].bn2.activation,self.output_1[0].bn2.activation_param)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Conv2d):
                if "conv_sem" not in name:
                    nn.init.xavier_normal_(mod.weight, gain)
                else:
                    nn.init.xavier_normal_(mod.weight, .1)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)
    @staticmethod
    def _make_lateral(input_channels, hidden_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 3, bias=False, padding=1)),
#            ("bn", norm_act(hidden_channels))
        ]))
       
    def forward(self, xs, xs_range=None):
        xs = xs[self.min_level:self.min_level + self.levels]
 
        ref_size = xs[0].shape[-2:]
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False
      
        i = self.min_level + self.levels - 1
        js = 0 
        if self.include_range:   
         xs_range = [xs_range[inp] for inp in xs_range]
         #print (np.max(xs_range[i].detach().cpu().numpy()))
         for output,lateral,pl,sl in zip(self.output_1, self.lateral[:2],[False,False],[[6,24],[12,32]]):
            
#            print ('aa',i, xs_range[i].shape, xs[i].shape)   
            xs[i] = output(xs[i], lateral(xs_range[i]), pl,sl)
           # if lateral.conv.weight.grad is not None:
                #print (torch.unique(output.conv1_3x3_1.depthwise.weight.grad))
           #     print(torch.unique(lateral.conv.weight.grad)) 
            i = i - 1
        else:
         for output in self.output_1:
            xs[i] = output(xs[i])
            i = i - 1
         
        interm = self.pre_process[js](xs[i+1] + functional.interpolate(xs[i+2], size=xs[i+1].shape[-2:], **interp_params))
        for output,lateral in zip(self.output_2, self.lateral[2:]):
#            print ('aa',i, xs_range[i].shape, xs[i].shape)
            pass_in = None   
            if i > 0:
                pass_in = lateral(xs_range[i])
            xs[i] = output(xs[i], pass_in)
            if js==1:
                interm = self.pre_process[js](xs[i+1])
           
            xs[i] = xs[i] + functional.interpolate(interm, size=xs[i].shape[-2:], **interp_params)
            js += 1
            i = i - 1
        for i in range(self.min_level, self.min_level + self.levels):
            if i > 0:
                xs[i] = functional.interpolate(xs[i], size=ref_size, **interp_params)
 
        xs = torch.cat(xs, dim=1)
        xs = self.conv_sem(xs)
#        print (xs.shape) 
        return xs



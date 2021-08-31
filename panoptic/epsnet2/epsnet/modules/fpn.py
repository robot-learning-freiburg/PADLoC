from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN
import cv2
import numpy as np
class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(_seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation , padding=dilation, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
 
        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x


class FPN_old(nn.Module):
    """Feature Pyramid Network module

    Parameters
    ----------
    in_channels : sequence of int
        Number of feature channels in each of the input feature levels
    out_channels : int
        Number of output feature channels (same for each level)
    extra_scales : int
        Number of extra low-resolution scales
    norm_act : callable
        Function to create normalization + activation modules
    interpolation : str
        Interpolation mode to use when up-sampling, see `torch.nn.functional.interpolate`
    """

    def __init__(self, in_channels, out_channels=256, extra_scales=0, norm_act=ABN, interpolation="nearest"):
        super(FPN, self).__init__()
        self.interpolation = interpolation

        # Lateral connections and output convolutions
        self.lateral = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])
        self.output = nn.ModuleList([
            self._make_output(out_channels, norm_act) for _ in in_channels
        ])

        if extra_scales > 0:
            self.extra = nn.ModuleList([
                self._make_extra(in_channels[-1] if i == 0 else out_channels, out_channels, norm_act)
                for i in range(extra_scales)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.lateral[0].bn.activation, self.lateral[0].bn.activation_param)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, gain)
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

    @staticmethod
    def _make_output(channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ("bn", norm_act(channels))
        ]))

    @staticmethod
    def _make_extra(input_channels, out_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, out_channels, 3, stride=2, padding=1, bias=False)),
            ("bn", norm_act(out_channels))
        ]))

    def forward(self, xs):
        """Feature Pyramid Network module

        Parameters
        ----------
        xs : sequence of torch.Tensor
            The input feature maps, tensors with shapes N x C_i x H_i x W_i

        Returns
        -------
        ys : sequence of torch.Tensor
            The output feature maps, tensors with shapes N x K x H_i x W_i
        """
        ys = []
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        # Build pyramid
        for x_i, lateral_i in zip(xs[::-1], self.lateral[::-1]):
            x_i = lateral_i(x_i)
            if len(ys) > 0:
                x_i = x_i + functional.interpolate(ys[0], size=x_i.shape[-2:], **interp_params)
            ys.insert(0, x_i)

        # Compute outputs
        ys = [output_i(y_i) for y_i, output_i in zip(ys, self.output)]

        # Compute extra outputs if necessary
        if hasattr(self, "extra"):
            y = xs[-1]
            for extra_i in self.extra:
                y = extra_i(y)
                ys.append(y)

        return ys

class FPN(nn.Module):
     

    """Feature Pyramid Network module

    Parameters
    ----------
    in_channels : sequence of int
        Number of feature channels in each of the input feature levels
    out_channels : int
        Number of output feature channels (same for each level)
    extra_scales : int
        Number of extra low-resolution scales
    norm_act : callable
        Function to create normalization + activation modules
    interpolation : str
        Interpolation mode to use when up-sampling, see `torch.nn.functional.interpolate`
    """

    def __init__(self, in_channels, ic ,out_channels=256, extra_scales=0, norm_act=ABN, interpolation="nearest"):
        super(FPN, self).__init__()
        self.interpolation = interpolation
        
        # Lateral connections and output convolutions
        self.lateral_r = nn.ModuleList([
            self._make_output_(out_channels, norm_act, 256+ci) for channels, ci in zip(in_channels,[8,16,24,40])
        ])
        self.lateral_w = nn.ModuleList([
            self._make_lateral_(256, 256, norm_act) for channels in ic
        ])
        self.lateral = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])
        self.lateral1 = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])

        self.output = nn.ModuleList([
            self._make_output(out_channels, norm_act) for _ in in_channels
        ])

        if extra_scales > 0:
            self.extra = nn.ModuleList([
                self._make_extra(in_channels[-1] if i == 0 else out_channels, out_channels, norm_act)
                for i in range(extra_scales)
            ]) 
          
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.lateral[0].bn.activation, self.lateral[0].bn.activation_param)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    @staticmethod
    def _make_lateral_(input_channels, hidden_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 1, bias=False)),
           # ("bn", norm_act(hidden_channels))
        ]))

    @staticmethod
    def _make_lateral(input_channels, hidden_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 1, bias=False)),
            ("bn", norm_act(hidden_channels))
        ]))

    @staticmethod
    def _make_output_(channels, norm_act, in_channels):
        if in_channels is None:
            in_channels = channels
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels, int(channels), 3, padding=1, bias=False)),
            ("bn", norm_act(int(channels))),
            ("conv1", nn.Conv2d(channels, int(channels), 3, padding=1, bias=False)),
            ("bn1", norm_act(int(channels)))
        ]))

    @staticmethod
    def _make_output(channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, int(channels), 3, padding=1, bias=False)),
            ("bn", norm_act(int(channels)))
        ]))
#        return nn.Sequential(OrderedDict([
#            ("conv",_seperable_conv(channels, int(channels), 1, None, bias=False)),
#            ("bn", norm_act(int(channels)))
#        ]))

    @staticmethod
    def _make_extra(input_channels, out_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, out_channels, 3, stride=2, padding=1, bias=False)),
            ("bn", norm_act(out_channels))
        ]))

    def forward(self, xs, x_range=None):
        """Feature Pyramid Network module

        Parameters
        ----------
        xs : sequence of torch.Tensor
            The input feature maps, tensors with shapes N x C_i x H_i x W_i

        Returns
        -------
        ys : sequence of torch.Tensor
            The output feature maps, tensors with shapes N x K x H_i x W_i
        """
        ys = []
        us = []
        fs = [] 
        zs = []  
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        # Build pyramid
        #for x_i, lateral_i in zip(x_range[::-1], self.lateral_r[::-1]):
        #    print (x_i.shape, lateral_i.conv.weight.shape)
        #    x_i = lateral_i(x_i)
        #    zs.insert(0, x_i)
        for x_i, lateral_i in zip(xs[::-1], self.lateral[::-1]):
            x_i = lateral_i(x_i)
            if len(ys) > 0:
                x_i = x_i + functional.interpolate(ys[0], size=x_i.shape[-2:], **interp_params)
            ys.insert(0, x_i)

        for x_i, lateral_i in zip(xs, self.lateral1):
            x_i = lateral_i(x_i)
            if len(us) > 0:
                x_i = x_i + functional.interpolate(us[0], size=x_i.shape[-2:], **interp_params)
            us.insert(0, x_i)
#        for a in x_range:
#            if a.shape[2]==32:
#                print(torch.unique(a))
        # Compute outputs
        ys_ = [output_i(y_i+u_i) for y_i,u_i,output_i in zip(ys, us[::-1], self.output)]
        ys_1 = [output_i(torch.cat((y_i+u_i,z_i),1)) for y_i,u_i,z_i, output_i in zip(ys, us[::-1], x_range, self.lateral_r)]
        ys = [u_i*output_i(u_i).sigmoid() + y_i*(1-output_i(u_i).sigmoid()) for y_i,u_i,output_i in zip(ys_, ys_1, self.lateral_w)]
#        print(ys[0][0,0,44,44].backward())
       

#        for y_i,u_i,output_i in zip(ys_, ys_1, self.lateral_w):
#           b = output_i(u_i).sigmoid()
#           a = u_i*b + y_i*(1-b)
#           print (torch.unique(1-b)) 
#           ys.append(a)
#           for i in range(256):
#            print (i)
#            op = ys[0][0,i,:,:]
#            op1 = ys_[0][0,i,:,:]
#            op2 = ys_1[0][0,i,:,:]
#            op = cv2.applyColorMap(np.uint8(((op-torch.min(op))/(torch.max(op)-torch.min(op))).detach().cpu().numpy()*255),cv2.COLORMAP_JET)
#            op1 = cv2.applyColorMap(np.uint8(((op1-torch.min(op1))/(torch.max(op1)-torch.min(op1))).detach().cpu().numpy()*255),cv2.COLORMAP_JET)
#            print (b.shape) 
#            op2 = cv2.applyColorMap(np.uint8(((op2-torch.min(op2))/(torch.max(op2)-torch.min(op2))).detach().cpu().numpy()*255),cv2.COLORMAP_JET)
#            cv2.imshow('img', op)
#            cv2.imwrite('op.png',op)
#            cv2.imshow('img1', op1)
#            cv2.imwrite('op1.png',op1)
#            cv2.imshow('img2', op2)
#            cv2.imwrite('op2.png',op2) 
#            cv2.waitKey(0)  


        # Compute exta outputs if necessary
        if hasattr(self, "extra"):
            
            y = xs[-1]
            for extra_i in self.extra:
                y = extra_i(y)
                ys.append(y)
        
        return ys

class FPN_(nn.Module):
     

    """Feature Pyramid Network module

    Parameters
    ----------
    in_channels : sequence of int
        Number of feature channels in each of the input feature levels
    out_channels : int
        Number of output feature channels (same for each level)
    extra_scales : int
        Number of extra low-resolution scales
    norm_act : callable
        Function to create normalization + activation modules
    interpolation : str
        Interpolation mode to use when up-sampling, see `torch.nn.functional.interpolate`
    """

    def __init__(self, in_channels ,out_channels=256, extra_scales=0, norm_act=ABN, interpolation="nearest"):
        super(FPN, self).__init__()
        self.interpolation = interpolation

        # Lateral connections and output convolutions
        self.lateral = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])
        self.lateral1 = nn.ModuleList([
            self._make_lateral(channels, out_channels, norm_act) for channels in in_channels
        ])

        self.output = nn.ModuleList([
            self._make_output(out_channels, norm_act) for _ in in_channels
        ])

        if extra_scales > 0:
            self.extra = nn.ModuleList([
                self._make_extra(in_channels[-1] if i == 0 else out_channels, out_channels, norm_act)
                for i in range(extra_scales)
            ]) 
          
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.lateral[0].bn.activation, self.lateral[0].bn.activation_param)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_normal_(mod.weight, gain)
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

    @staticmethod
    def _make_output(channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, int(channels), 3, padding=1, bias=False)),
            ("bn", norm_act(int(channels)))
        ]))
#        return nn.Sequential(OrderedDict([
#            ("conv",_seperable_conv(channels, int(channels), 1, None, bias=False)),
#            ("bn", norm_act(int(channels)))
#        ]))

    @staticmethod
    def _make_extra(input_channels, out_channels, norm_act):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, out_channels, 3, stride=2, padding=1, bias=False)),
            ("bn", norm_act(out_channels))
        ]))

    def forward(self, xs):
        """Feature Pyramid Network module

        Parameters
        ----------
        xs : sequence of torch.Tensor
            The input feature maps, tensors with shapes N x C_i x H_i x W_i

        Returns
        -------
        ys : sequence of torch.Tensor
            The output feature maps, tensors with shapes N x K x H_i x W_i
        """
        ys = []
        us = []
        fs = [] 
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        # Build pyramid
        
        for x_i, lateral_i in zip(xs[::-1], self.lateral[::-1]):
            x_i = lateral_i(x_i)
            if len(ys) > 0:
                x_i = x_i + functional.interpolate(ys[0], size=x_i.shape[-2:], **interp_params)
            ys.insert(0, x_i)

        for x_i, lateral_i in zip(xs, self.lateral1):
            x_i = lateral_i(x_i)
            if len(us) > 0:
                x_i = x_i + functional.interpolate(us[0], size=x_i.shape[-2:], **interp_params)
            us.insert(0, x_i)

        # Compute outputs
        ys = [output_i(y_i+u_i) for y_i,u_i,output_i in zip(ys, us[::-1], self.output)]
        
        # Compute exta outputs if necessary
        if hasattr(self, "extra"):
            
            y = xs[-1]
            for extra_i in self.extra:
                y = extra_i(y)
                ys.append(y)
        
        return ys



class FPNBody(nn.Module):
    """Wrapper for a backbone network and an FPN module

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone network, which takes a batch of images and produces a dictionary of intermediate features
    fpn : torch.nn.Module
        FPN module, which takes a list of intermediate features and produces a list of outputs
    fpn_inputs : iterable
        An iterable producing the names of the intermediate features to take from the backbone's output and pass
        to the FPN
    """

    def __init__(self, backbone, fpn, fpn_inputs=()):
        super(FPNBody, self).__init__()
        self.fpn_inputs = fpn_inputs
        self.backbone = backbone
        self.fpn = fpn
        
    def forward(self, x, x_range=None, proj_msk=None):
        x = self.backbone(x, proj_msk)
#        for a in x: 
#            print(x[a].shape,torch.unique(x[a])) 
        xs = [x[fpn_input] for fpn_input in self.fpn_inputs]
        if x_range is not None:
            x_range = [x_range[fpn_input] for fpn_input in self.fpn_inputs]
            return self.fpn(xs, x_range) 
        return self.fpn(xs) 
        

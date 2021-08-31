from math import ceil
import numpy as np
import torch
import torch.nn.functional as functional

from epsnet.utils.parallel import PackedSequence
from epsnet.utils.sequence import pack_padded_images
from epsnet.utils.bbx import invert_roi_bbx
from epsnet.utils.roi_sampling import roi_sampling
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

class SemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, ignore_index=255):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index

    def __call__(self, sem_logits, sem):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        for sem_logits_i, sem_i in zip(sem_logits, sem):
            sem_loss_i = functional.cross_entropy(
                sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0), ignore_index=self.ignore_index, reduction="none")
            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)

class cSemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, ignore_index=255):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index

    def __call__(self, sem_logits, sem):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        for sem_logits_i, sem_i in zip(sem_logits, sem):
#            print ('sem_shape', sem_logits_i.shape)
            loss = lovasz_softmax_flat(*flatten_probas(functional.softmax(sem_logits_i.unsqueeze(0),dim=1), sem_i.unsqueeze(0), self.ignore_index))
            sem_loss_i = functional.cross_entropy(
                sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0), ignore_index=self.ignore_index, reduction="none")
            sem_loss_i = sem_loss_i.view(-1)
            
            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)
            sem_loss.append(sem_loss_i.mean() + loss)

        return sum(sem_loss) / len(sem_logits)

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class SemanticSegAlgo:
    """Semantic segmentation algorithm

    Parameters
    ----------
    loss : SemanticSegLoss
    num_classes : int
        Number of classes
    """

    def __init__(self, loss, num_classes, ignore_index=255):
        self.loss = loss
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    @staticmethod
    def _pack_logits(sem_logits, valid_size, img_size):
        sem_logits = functional.interpolate(sem_logits, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(sem_logits, valid_size)

    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                confmat.index_add_(
                    0, sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1), confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    @staticmethod
    def _logits(head, x, valid_size, img_size, x_range=None):
        sem_logits = head(x, x_range)
        return sem_logits 

    def training(self, head, x, sem, valid_size, img_size, x_range=None):
        """Given input features and ground truth compute semantic segmentation loss, confusion matrix and prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        conf_mat : torch.Tensor
            A confusion matrix tensor with shape M x M, where M is the number of classes
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        # Compute logits and prediction
#        print (x_range['mod5'].shape, x_range['mod4'].shape)
        sem_logits_ = self._logits(head, x, valid_size, img_size, x_range)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)
        
#        sem_logits = self._pack_logits(sem_logits_, valid_size, [1024,2048])
        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])

        # Compute loss and confusion matrix
        sem_loss = self.loss(sem_logits, sem)
        conf_mat = self._confusion_matrix(sem_pred, sem)

        return sem_loss, conf_mat, sem_pred, sem_logits

    def inference(self, head, x, valid_size, img_size, x_range=None):
        """Given input features compute semantic segmentation prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        sem_logits_ = self._logits(head, x, valid_size, img_size, x_range)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)
        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        return sem_pred, sem_logits


def confusion_matrix(sem_pred, sem, num_classes, ignore_index=255):
    confmat = sem_pred.new_zeros(num_classes * num_classes, dtype=torch.float)

    valid = sem != ignore_index
    if valid.any():
        sem_pred = sem_pred[valid]
        sem = sem[valid]

        confmat.index_add_(0, sem.view(-1) * num_classes + sem_pred.view(-1), confmat.new_ones(sem.numel()))

    return confmat.view(num_classes, num_classes)


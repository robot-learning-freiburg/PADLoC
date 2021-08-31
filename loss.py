import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning import distances
from utils.tools import pairwise_mse


class TripletLoss(nn.Module):
    def __init__(self, margin: float, triplet_selector, distance: distances.BaseDistance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.distance = distance

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        if other_embeddings is None:
            other_embeddings = embeddings
        dist_mat = self.distance(embeddings, other_embeddings)
        triplets = self.triplet_selector(dist_mat, pos_mask, neg_mask, self.distance.is_inverted)
        distance_positive = dist_mat[triplets[0], triplets[1]]
        if triplets[-1] is None:
            if self.distance.is_inverted:
                return F.relu(1 - distance_positive).mean()
            else:
                return F.relu(distance_positive).mean()
        distance_negative = dist_mat[triplets[0], triplets[2]]
        curr_margin = self.distance.margin(distance_positive, distance_negative)
        loss = F.relu(curr_margin + self.margin)
        return loss.mean()


class MyCircleLoss(CircleLoss):
    def __init__(self, version='PML', *args, **kwargs):
        """
        Modified CircleLoss
        Args:
            version: Either 'PML': (similar to the one implemented in pytorch_metric_learning,
                     or 'TZM': adapted from https://github.com/TinyZeaMays/CircleLoss
            *args:
            **kwargs:
        """
        super(MyCircleLoss, self).__init__(*args, **kwargs)
        self.version = version
        if self.version not in ['PML', 'TZM', 'TZMGrad']:
            raise AttributeError(f"Circle loss with version {version} unknown")


    def logsumexp(self, x, keep_mask=None, add_one=True, dim=1):
        max_vals = torch.zeros(x.shape[0], device=x.device)
        keep_mask = keep_mask.bool()
        for i in range(x.shape[0]):
            max_vals[i] = torch.max(x[i, keep_mask[i]])
        max_vals = max_vals.unsqueeze(1)
        inside_exp = x - max_vals
        inside_exp[~keep_mask] = 0.
        exp = torch.exp(inside_exp)
        if keep_mask is not None:
            exp = exp*keep_mask
        inside_log = torch.sum(exp, dim=dim, keepdim=True)
        if add_one:
            inside_log = inside_log + torch.exp(-max_vals)
        else:
            # add one only if necessary
            inside_log[inside_log==0] = torch.exp(-max_vals[inside_log==0])
        return torch.log(inside_log) + max_vals


    def _compute_loss(self, dist_mat, pos_mask, neg_mask):
        pos_mask_bool = pos_mask.bool()
        neg_mask_bool = neg_mask.bool()
        anchor_positive = dist_mat[pos_mask_bool]
        anchor_negative = dist_mat[neg_mask_bool]

        if self.version == 'PML':
            new_mat = torch.zeros_like(dist_mat)
            new_mat[pos_mask_bool] = -self.gamma * torch.relu(self.op - anchor_positive.detach()) * (anchor_positive - self.delta_p)
            new_mat[neg_mask_bool] = self.gamma * torch.relu(anchor_negative.detach() - self.on) * (anchor_negative - self.delta_n)


            losses = self.soft_plus(self.logsumexp(new_mat, keep_mask=pos_mask, add_one=False, dim=1) + self.logsumexp(new_mat, keep_mask=neg_mask, add_one=False, dim=1))

            zero_rows = torch.where((torch.sum(pos_mask, dim=1)==0) | (torch.sum(neg_mask, dim=1) == 0))[0]
            final_mask = torch.ones_like(losses)
            final_mask[zero_rows] = 0
            losses = losses*final_mask
            return losses
            # return {"loss": {"losses": losses, "indices": c_f.torch_arange_from_size(new_mat), "reduction_type": "element"}}
        elif self.version == 'TZM':
            ap = -self.gamma * torch.relu(self.op - anchor_positive.detach()) * (anchor_positive - self.delta_p)
            an = self.gamma * torch.relu(anchor_negative.detach() - self.on) * (anchor_negative - self.delta_n)
            loss = self.soft_plus(torch.logsumexp(ap, dim=0) + torch.logsumexp(an, dim=0))
            return loss
        elif self.version == 'TZMGrad':
            ap = torch.clamp_min(- anchor_positive.detach() + 1 + self.m, min=0.)
            an = torch.clamp_min(anchor_negative.detach() + self.m, min=0.)
            logit_p = - ap * (anchor_positive - self.delta_p) * self.gamma
            logit_n = an * (anchor_negative - self.delta_n) * self.gamma

            loss = torch.log(1 + torch.clamp_max(torch.exp(logit_n).sum() * torch.exp(logit_p).sum(), max=1e38))
            z = - torch.exp(- loss) + 1

            anchor_positive.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma, retain_graph=True)
            anchor_negative.backward(gradient=z * an * torch.softmax(logit_n, dim=0) * self.gamma, retain_graph=True)
            return loss.detach()


class smooth_metric_lossv2(nn.Module):
    def __init__(self, margin):
        super(smooth_metric_lossv2, self).__init__()
        self.margin = margin

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        """

        Args:
            embeddings: Embedding of shape 2*N, i and i+N should be positive pairs
            neg_mask:
            other_embeddings:

        Returns:

        """
        if other_embeddings is None:
            other_embeddings = embeddings
        # CARE: when using dot, embedding should be normalized (i guess hehe)
        # D = pairwise_mse(embeddings, other_embeddings) + 1e-5
        D = torch.cdist(embeddings, other_embeddings, p=2)
        # D = torch.sqrt(D)
        # marg_D = self.margin - D
        batch_size = embeddings.shape[0]
        J_all = []

        for i in range(embeddings.shape[0]//2):

            matching_idx = i+embeddings.shape[0]//2
            ap_distance = D[i, matching_idx]#.sqrt()

            neg_d_1 = self.margin - D[i, neg_mask[i]]#.sqrt()
            neg_d_2 = self.margin - D[matching_idx, neg_mask[matching_idx]]#.sqrt()
            # J_ij = neg_d - neg_d.max()  # Why did i add this?
            # J_ij = torch.exp(J_ij).sum()
            J_ij_1 = torch.exp(neg_d_1).sum()
            J_ij_2 = torch.exp(neg_d_2).sum()
            J_ij = (J_ij_1+J_ij_2).log() + ap_distance
            if torch.any(torch.isnan(J_ij)):
                print("NaN found")
            else:
                J_all.append(J_ij)

        J_all = torch.stack(J_all)
        loss = F.relu(J_all).pow(2).mean().div(2)
        return loss


class NPair_loss(nn.Module):
    def __init__(self):
        super(NPair_loss, self).__init__()

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        """

        Args:
            embeddings: Embedding of shape 2*N, i and i+N should be positive pairs
            neg_mask:
            pos_mask:
            other_embeddings:

        Returns:

        """
        if other_embeddings is None:
            other_embeddings = embeddings
        # CARE: when using dot, embedding should be normalized (i guess hehe)
        # D = pairwise_mse(embeddings, other_embeddings) + 1e-5
        D = torch.mm(embeddings, torch.transpose(other_embeddings,0,1))
        # D = torch.sqrt(D)
        # marg_D = self.margin - D
        batch_size = embeddings.shape[0]
        J_all = []

        for i in range(embeddings.shape[0]//2):

            matching_idx = i+embeddings.shape[0]//2
            ap_distance = D[i, matching_idx]#.sqrt()

            # expm = torch.exp(ap_distance - D)
            expm = torch.exp(D - ap_distance)
            J_ij = expm[i, neg_mask[i]].sum()

            J_ij = (J_ij + 1).log()
            if torch.any(torch.isnan(J_ij)):
                print("NaN found")
            else:
                J_all.append(J_ij)

        J_all = torch.stack(J_all)
        loss = J_all.mean()
        return loss


class Circle_Loss(nn.Module):
    def __init__(self, version='PML', m=0.25, gamma=256):
        super(Circle_Loss, self).__init__()
        self.version = version
        self.loss_fn = MyCircleLoss(version, m=m, gamma=gamma)

    def forward(self, embeddings, pos_mask, neg_mask, other_embeddings=None):
        """

        Args:
            embeddings: Embedding of shape 2*N, i and i+N should be positive pairs
            neg_mask:
            pos_mask:
            other_embeddings:

        Returns:

        """
        if other_embeddings is None:
            other_embeddings = embeddings
        batch_size = embeddings.shape[0] // 2
        # a1_idx, p_idx = torch.where(torch.eye(batch_size).repeat(1, 2))
        # a2_idx, n_idx = torch.where(neg_idxs)
        # a1_idx, p_idx = a1_idx.to(embeddings.device), p_idx.to(embeddings.device)
        # a2_idx, n_idx = a2_idx.to(embeddings.device), n_idx.to(embeddings.device)
        D = self.loss_fn.distance(embeddings, other_embeddings)
        loss = self.loss_fn._compute_loss(D, pos_mask, neg_mask)
        if self.version == 'PML':
            nonzero_idx = loss > 0
            if nonzero_idx.sum() == 0.:
                return loss.mean()*0
            return loss[nonzero_idx].mean()
        else:
            return loss


def sinkhorn_matches_loss(batch_dict, delta_pose, mode='pairs'):
    sinkhorn_matches = batch_dict['sinkhorn_matches']
    src_coords = batch_dict['point_coords']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    B, N_POINT, _ = src_coords.shape
    if mode == 'pairs':
        B = B // 2
    else:
        B = B // 3
    src_coords = src_coords[:B, :, [1, 2, 3, 0]]
    src_coords[:, :, -1] = 1.
    gt_dst_coords = torch.bmm(delta_pose.inverse(), src_coords.permute(0, 2, 1))
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    return loss


def pose_loss(batch_dict, delta_pose, mode='pairs'):
    src_coords = batch_dict['point_coords']
    src_coords = src_coords.clone().view(batch_dict['batch_size'], -1, 4)
    B, N_POINT, _ = src_coords.shape
    if mode == 'pairs':
        B = B // 2
    else:
        B = B // 3
    src_coords = src_coords[:B, :, [1, 2, 3, 0]]
    src_coords[:, :, -1] = 1.
    delta_pose_inv = delta_pose.double().inverse()
    gt_dst_coords = torch.bmm(delta_pose_inv, src_coords.permute(0,2,1).double()).float()
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    # loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    transformation = batch_dict['transformation']
    pred_dst_coords = torch.bmm(transformation, src_coords.permute(0,2,1))
    pred_dst_coords = pred_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = torch.mean(torch.abs(pred_dst_coords - gt_dst_coords))
    # loss = (pred_dst_coords - gt_dst_coords).norm(dim=2).mean()
    return loss


def rpm_loss_for_rpmnet(points_src, transformations, delta_pose, mode='pairs'):
    src_coords = points_src
    src_coords[:, :, -1] = 1.
    gt_dst_coords = torch.bmm(delta_pose.inverse(), src_coords.permute(0, 2, 1))
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    # loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    loss = torch.tensor([0.], device=points_src.device, dtype=points_src.dtype)
    for i in range(len(transformations)):
        transformation = transformations[i]
        pred_dst_coords = torch.bmm(transformation, src_coords.permute(0, 2, 1))
        pred_dst_coords = pred_dst_coords.permute(0, 2, 1)[:, :, :3]
        discount = 0.5
        discount = discount ** (len(transformations) - i - 1)
        loss += torch.mean(torch.abs(pred_dst_coords - gt_dst_coords)) * discount
    return loss

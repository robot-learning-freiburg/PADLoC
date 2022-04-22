import numpy as np
from pytorch_metric_learning.losses import CircleLoss as PyCircleLoss
from pytorch_metric_learning import distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Callable

from models.backbone3D.functional import hard_kronecker
from models.backbone3D.matchers import normalize_matching
from triple_selector import hardest_negative_selector, random_negative_selector, \
    semihard_negative_selector
from utils.qcqp_layer import QuadQuatFastSolver
from utils.rotation_conversion import quaternion_atan_loss
from utils.tools import NaNLossError


_REVERSE_L_SUFFIX = "(Reverse)"


# TODO: Documentation!!!
def _get_point_hcoords(batch_dict, *, point_set, mode):
    points_key_dict = {
        "anc": "anc_hcoords",
        "pos": "pos_hcoords",
        "neg": "neg_hcoords"
    }

    if point_set not in points_key_dict:
        raise KeyError(f"Invalid point set ({point_set}). Valid values [{points_key_dict.keys()}].")

    points_key = points_key_dict[point_set]

    if points_key in batch_dict:
        return batch_dict[points_key]

    point_coords = batch_dict["point_coords"]
    point_coords = point_coords.clone().view(batch_dict["batch_size"], -1, 4)
    bt, _, _ = point_coords.shape
    t = 2 if mode == "pairs" else 3
    b = bt // t
    assert b * t == bt

    # Move reflectivity to last position
    point_coords = point_coords[:, :, [1, 2, 3, 0]]
    # Set last position to 1 to make them homogeneous coordinates
    point_coords[:, :, -1] = 1.

    batch_dict[points_key_dict["anc"]] = point_coords[:b]
    batch_dict[points_key_dict["pos"]] = point_coords[b:2*b]
    if t == 3:
        batch_dict[points_key_dict["neg"]] = point_coords[2*b:3*b]

    return batch_dict[points_key]


def _loss_sinkhorn_matches(*, sinkhorn_matches, src_coords, delta_pose):
    gt_dst_coords = torch.bmm(delta_pose, src_coords.permute(0, 2, 1))
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = (gt_dst_coords - sinkhorn_matches).norm(dim=2).mean()
    return loss


def loss_sinkhorn_matches(batch_dict, *, mode, reverse_loss=False, **_):
    sinkhorn_matches = batch_dict["sinkhorn_matches"]
    delta_pose = batch_dict["delta_pose"]

    anc_coords = _get_point_hcoords(batch_dict, point_set="anc", mode=mode)

    loss_dict = {
        "": _loss_sinkhorn_matches(sinkhorn_matches=sinkhorn_matches, src_coords=anc_coords,
                                   delta_pose=delta_pose)
    }

    if reverse_loss:
        sinkhorn_matches2 = batch_dict["sinkhorn_matches2"]
        pos_coords = _get_point_hcoords(batch_dict, point_set="pos", mode=mode)
        delta_pose_inv = delta_pose.inverse()
        loss_dict[_REVERSE_L_SUFFIX] = _loss_sinkhorn_matches(sinkhorn_matches=sinkhorn_matches2, src_coords=pos_coords,
                                                              delta_pose=delta_pose_inv)

    return loss_dict


def _loss_pose(*, transformation, src_coords, delta_pose_inv):
    gt_dst_coords = torch.bmm(delta_pose_inv, src_coords.permute(0, 2, 1).double()).float()
    gt_dst_coords = gt_dst_coords.permute(0, 2, 1)[:, :, :3]
    pred_dst_coords = torch.bmm(transformation, src_coords.permute(0, 2, 1))
    pred_dst_coords = pred_dst_coords.permute(0, 2, 1)[:, :, :3]
    loss = torch.mean(torch.abs(pred_dst_coords - gt_dst_coords))
    return loss


def loss_pose(batch_dict, *, mode, reverse_loss=False, **_):

    transformation = batch_dict["transformation"]
    delta_pose = batch_dict["delta_pose"].double()

    anc_coords = _get_point_hcoords(batch_dict, point_set="anc", mode=mode)
    delta_pose_inv = delta_pose.inverse()

    loss_dict = {
        "": _loss_pose(transformation=transformation, src_coords=anc_coords, delta_pose_inv=delta_pose_inv)
    }

    if reverse_loss:
        transformation2 = batch_dict["transformation2"]
        # Coordinates should be the same as in the non-reverse loss,
        # since we are only using them to compare the predicted and true transformations
        loss_dict[_REVERSE_L_SUFFIX] = _loss_pose(transformation=transformation2, src_coords=anc_coords,
                                                  delta_pose_inv=delta_pose)

    return loss_dict


def rpm_loss_for_rpmnet(points_src, transformations, delta_pose, **_):
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


def _get_normalized_matching(batch_dict, pos2anc=True):
    pos2anc_key = "norm_match_p2a"
    anc2pos_key = "norm_match_a2p"
    match_key = pos2anc_key if pos2anc else anc2pos_key

    if match_key in batch_dict:
        return batch_dict[match_key]

    batch_dict[pos2anc_key] = normalize_matching(batch_dict["transport"])
    batch_dict[anc2pos_key] = normalize_matching(batch_dict["transport2"])

    return batch_dict[match_key]


def _remap_label(value, mapping_dict: Optional[Dict[int, int]] = None):
    if mapping_dict is None:
        return value

    remapped_value = torch.zeros_like(value, dtype=value.dtype, device=value.device)

    for src, tgt in mapping_dict.items():
        idx = torch.nonzero(value == src)
        remapped_value[idx] = tgt

    return remapped_value


def _loss_miss_classification(*, match_p2a, labels1_one_hot, labels2_one_hot, cce=True):
    pred_labels1_one_hot = torch.bmm(match_p2a, labels2_one_hot)

    # Categorical Cross-Entropy loss between the one-hot encoded predicted tensor and the true indices
    if cce:
        masked_pred_labels_1 = pred_labels1_one_hot * labels1_one_hot
        masked_pred_labels_1 = masked_pred_labels_1.sum(-1)  # Only select the entry for the true label
        # masked_pred_labels_1 = pred_labels1_one_hot[labels1]  # Only select the entry for the true label
        masked_pred_labels_1 = 1e-8 + masked_pred_labels_1  # To avoid getting NANs if the prediction is 0
        masked_pred_labels_1 = masked_pred_labels_1.clamp(0., 1.)  # To avoid having values larger than 1
        loss = - torch.log(masked_pred_labels_1)  # CCE Loss
        loss = loss.mean()
        return loss

    # Mean Square Error between the predicted and true one-hot encoded tensors
    err_mat = labels1_one_hot - pred_labels1_one_hot
    err_vec = err_mat.square().sum(-1)
    loss = err_vec.mean()
    return loss


def _loss_meta_semantic(batch_dict, *, meta, reverse_loss=False, **_):
    match_p2a = _get_normalized_matching(batch_dict, pos2anc=True)

    lbl_suffix = "semantic"
    map_key_prefix = "class"
    if meta:
        lbl_suffix = "supersem"
        map_key_prefix = "superclass"

    b = match_p2a.shape[0]  # Batch size

    # Original labels for All points in each point cloud
    lbl1_orig_all_points = batch_dict["anchor_" + lbl_suffix]
    lbl2_orig_all_points = batch_dict["positive_" + lbl_suffix]

    # Extract the original labels of only the sampled points
    keypoint_idx = batch_dict['keypoint_idxs']  # Indices of the sampled points
    lbl1_sampled = [s[i] for s, i in zip(lbl1_orig_all_points, keypoint_idx[:b])]
    lbl2_sampled = [s[i] for s, i in zip(lbl2_orig_all_points, keypoint_idx[b:2*b])]

    # Remap labels from sparse to contiguous so that the one-hot encoding doesn't have unnecessary columns
    oh_mapping = batch_dict[map_key_prefix + "_one_hot_map"]
    lbl1_remap = [_remap_label(t, m) for t, m in zip(lbl1_sampled, oh_mapping)]
    lbl2_remap = [_remap_label(t, m) for t, m in zip(lbl2_sampled, oh_mapping)]
    # Turn list of label index tensors into single tensor by stacking them
    lbl1_remap = torch.stack(lbl1_remap).view(b, -1)
    lbl2_remap = torch.stack(lbl2_remap).view(b, -1)
    # Cast as int64, otherwise one_hot gives an error
    lbl1_remap = lbl1_remap.to(torch.int64)
    lbl2_remap = lbl2_remap.to(torch.int64)
    # Detach to prevent gradient from flowing through here, since both are Ground Truth
    lbl1_remap = lbl1_remap.detach()
    lbl2_remap = lbl2_remap.detach()

    # Encode labels as one-hot
    n_classes = max([max(mapping.values()) for mapping in oh_mapping]) + 1  # Number of different labels
    lbl1_oh = F.one_hot(lbl1_remap, n_classes)
    lbl2_oh = F.one_hot(lbl2_remap, n_classes)
    # Recast as floats, otherwise BMM in loss complains :S
    lbl1_oh = lbl1_oh.to(torch.float32)
    lbl2_oh = lbl2_oh.to(torch.float32)
    # Detach to get rid of any gradient, since it is the ground truth
    lbl1_oh = lbl1_oh.detach()
    lbl2_oh = lbl2_oh.detach()

    loss_dict = {
        # "": _loss_miss_classification(match_p2a=match_p2a, labels1=lbl1_oh, labels2_one_hot=lbl2_oh, cce=False) # MSE
        "": _loss_miss_classification(match_p2a=match_p2a, labels1_one_hot=lbl1_oh, labels2_one_hot=lbl2_oh, cce=True)
    }

    if reverse_loss:
        match_a2p = _get_normalized_matching(batch_dict, pos2anc=False)

        # loss_dict[_REVERSE_L_SUFFIX] = _loss_miss_classification(match_p2a=match_a2p,
        #                                                    labels1=lbl2_oh, labels2_one_hot=lbl1_oh, cce=False) # MSE
        loss_dict[_REVERSE_L_SUFFIX] = _loss_miss_classification(match_p2a=match_a2p,
                                                                 labels1_one_hot=lbl2_oh, labels2_one_hot=lbl1_oh,
                                                                 cce=True)

    return loss_dict


def loss_semantic(batch_dict, *, reverse_loss=False, **_):
    return _loss_meta_semantic(batch_dict, meta=False, reverse_loss=reverse_loss)


def loss_metasemantic(batch_dict, *, reverse_loss=False, **_):
    return _loss_meta_semantic(batch_dict, meta=True, reverse_loss=reverse_loss)


def _loss_panoptic(*, match_a2p, match_p2a, obj1, obj2):
    mask = 1 - obj1
    pred_obj1 = torch.bmm(match_p2a, obj2)
    pred_obj1 = torch.bmm(pred_obj1, match_a2p)
    err = obj1 - pred_obj1
    masked_err = mask * err
    # loss = masked_err.square() # Don't square them, it makes them tiny
    loss = masked_err.abs()  # to get bigger values and more resolution
    # loss = loss.sum([1, 2]) # Don't sum them per matrix, otherwise huge values (~400'000) that depend on points_num
    # loss = loss.mean()  # Don't average of all values, there are a ton of zeros in the matrix, lowering the mean
    batch_dim = match_a2p.shape[0]
    batch_mean_losses = torch.zeros(batch_dim, device=match_a2p.device)
    for i in range(batch_dim):
        batch_loss = loss[i]
        batch_loss_non_zero = batch_loss[batch_loss != 0.]  # Keep only non-zero entries.
        # Compute mean over the matrices, since each matrix will have diff. number of non-zero edges
        batch_mean_loss = batch_loss_non_zero.mean()
        batch_mean_losses[i] = batch_mean_loss
    loss = batch_mean_losses.mean()  # mean over the batches, since each matrix mean will have different denominators
    return loss


def loss_panoptic(batch_dict, *, reverse_loss=False, **_):
    # Panoptic Label mismatch loss
    match_p2a = _get_normalized_matching(batch_dict, pos2anc=True)
    match_a2p = _get_normalized_matching(batch_dict, pos2anc=False)

    b = match_p2a.shape[0]
    keypoint_idx = batch_dict['keypoint_idxs']

    # Get the Panoptic Labels for each point cloud
    panoptic_1 = torch.stack([s[i] for s, i in zip(batch_dict['anchor_panoptic'], keypoint_idx[:b])]).view(b, -1)
    panoptic_2 = torch.stack([s[i] for s, i in zip(batch_dict['positive_panoptic'], keypoint_idx[b:2*b])]).view(b, -1)

    # Build the object connectivity graph matrices
    obj1 = hard_kronecker(panoptic_1)
    obj2 = hard_kronecker(panoptic_2)

    # Disable the gradient since they are the Ground Truth, just in case
    obj1 = obj1.detach()
    obj2 = obj2.detach()

    loss_dict = {
        "": _loss_panoptic(match_a2p=match_a2p, match_p2a=match_p2a, obj1=obj1, obj2=obj2)
    }

    if reverse_loss:
        loss_dict[_REVERSE_L_SUFFIX] = _loss_panoptic(match_a2p=match_p2a, match_p2a=match_a2p, obj1=obj2, obj2=obj1)

    return loss_dict


def inverse_tf_loss(batch_dict):
    """
        Loss of multiplying the transformation of PC1 -> PC2 and that of the reverse (PC2 -> PC1)
        and comparing it to the identity matrix.

        DON'T USE.
        Instead use the reverse_pose_loss and reverse_sinkhorn_matches_loss.
        Better to learn the actual inverse transformation,
        than to learn that they must be inverse matrices to each other.
    """

    tf_mat_a = batch_dict['transformation']
    b = tf_mat_a.shape[0]
    device = tf_mat_a.device

    tf_mat = torch.zeros((b, 4, 4), device=device)
    tf_mat[:, :3, :] = tf_mat_a
    tf_mat[:, 3, 3] = 1.0

    tf_mat_inv = torch.zeros((b, 4, 4), device=device)
    tf_mat_inv[:, :3, :] = batch_dict['transformation2']
    tf_mat_inv[:, 3, 3] = 1.0

    loss = torch.bmm(tf_mat, tf_mat_inv)
    loss = torch.eye(4, 4, device=device) - loss
    loss = torch.square(loss)
    loss = torch.mean(loss)

    return loss


def rottrace_loss(batch_dict, delta_pose):
    """
    Rotation Loss, based on the fact that:

    tr(R) = 1 + 2 cos(theta)

    for any given 3D rotation matrix representing a rotation of theta around an arbitrary axis.

    :param batch_dict:
    :param delta_pose:
    :return:
    """

    predicted_pose = batch_dict['transformation']
    homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(predicted_pose.shape[0], 1, 1).to(predicted_pose.device)
    predicted_pose = torch.cat((predicted_pose, homogeneous), dim=1)

    # Invert the ground truth, so that the gradient does not have to propagate back through the inversion process.
    delta_pose_inv = delta_pose.double().inverse().float()

    err_pose = torch.bmm(delta_pose_inv, predicted_pose)

    # Computing the trace, since torch.trace() doesn't work as it expects a 2D Matrix.
    rot_err = err_pose.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    rot_err = (1 - rot_err) / 2  # rot_err = - cos(theta)
    tra_err = torch.norm(err_pose[:, :3, 3:], dim=1)

    mean_rot_err = torch.mean(rot_err)
    mean_tra_err = torch.mean(tra_err)

    # mean_rot_err_deg = torch.zeros_like(rot_err)
    # mean_rot_err_deg[rot_err <= -1.] = torch.tensor(0., device=mean_rot_err.device)
    # mean_rot_err_deg[rot_err >= 1.] = torch.tensor(np.pi, device=mean_rot_err.device)
    # valid_idx = torch.logical_and(rot_err > -1., rot_err < 1.)
    mean_rot_err_deg = - torch.clip(rot_err, -1., 1.)
    mean_rot_err_deg = torch.arccos(mean_rot_err_deg)

    # Circular mean of the error angles
    mean_rot_err_deg_c = torch.mean(torch.cos(mean_rot_err_deg))
    mean_rot_err_deg_s = torch.mean(torch.sin(mean_rot_err_deg))
    mean_rot_err_deg = torch.atan2(mean_rot_err_deg_s, mean_rot_err_deg_c)

    # Normalize angle between -pi and pi
    # Not needed, since all outputs of acos are between -pi and pi ???

    # Convert to degrees
    mean_rot_err_deg = torch.abs(mean_rot_err_deg)
    mean_rot_err_deg = mean_rot_err_deg * 180 / np.pi

    return mean_rot_err, mean_rot_err_deg, mean_tra_err


def _loss_sinkhorn_inlier(transport):
    inlier_loss = (1 - transport.sum(dim=1)).mean()
    inlier_loss += (1 - transport.sum(dim=2)).mean()
    return inlier_loss


def loss_sinkhorn_inlier(batch_dict, reverse_loss=False, **_):
    loss_dict = {
        "": _loss_sinkhorn_inlier(batch_dict["transport"])
    }

    if reverse_loss:
        loss_dict[_REVERSE_L_SUFFIX] = _loss_sinkhorn_inlier(batch_dict["transport2"])

    return loss_dict


def loss_transl(batch_dict):
    transl_out = batch_dict["out_translation"]
    transl_diff = batch_dict["transl_diff"]
    reg_loss = torch.nn.SmoothL1Loss(reduction='none')
    # loss_transl = L1loss(transl_diff, transl_out).sum(1).mean() * exp_cfg['weight_transl']
    loss = reg_loss(transl_out, transl_diff).sum(1).mean()

    return loss


def _loss_quat(batch_dict):
    delta_quat = batch_dict["delta_quat"]
    yaws_out = batch_dict["out_rotation"]

    yaws_out = F.normalize(yaws_out, dim=1)
    loss_rot = quaternion_atan_loss(yaws_out, delta_quat).mean()
    return loss_rot


def _loss_bingham(batch_dict):
    delta_quat = batch_dict["delta_quat"]
    yaws_out = batch_dict["out_rotation"]

    to_quat = QuadQuatFastSolver()
    quat_out = to_quat.apply(yaws_out)
    loss_rot = quaternion_atan_loss(quat_out, delta_quat[:, [3, 0, 1, 2]]).mean()
    return loss_rot


class QuatRotationLoss:

    _LOSS_FN_DICT = {
        "quat": _loss_quat,
        "bingham": _loss_bingham,
    }

    def __init__(self, cfg):
        self. reg_loss = torch.nn.SmoothL1Loss(reduction='none')
        # self.reg_loss = torch.nn.MSELoss(reduction='none')

        self.loss_fn = None
        self.num_classes = None

        rot_repr = cfg['rot_representation']

        loss_fn_dict = {

        }
        if rot_repr not in loss_fn_dict:
            raise NotImplementedError(f"No loss function configured for {rot_repr}.")

        self.loss_fn = loss_fn_dict[rot_repr]

    def __call__(self, batch_dict, **_):
        return self.loss_fn(batch_dict)


class RotationLoss:
    def __init__(self, cfg):
        self. reg_loss = torch.nn.SmoothL1Loss(reduction='none')
        # self.reg_loss = torch.nn.MSELoss(reduction='none')

        self.loss_fn = None
        self.num_classes = None

        rot_repr = cfg['rot_representation']

        loss_fn_dict = {
            "sincos": self._loss_sincos,
            "sincos_atan": self._loss_sincos_atan,
            "yaw": self._loss_yaw,
        }

        if rot_repr in loss_fn_dict:
            self.loss_fn = loss_fn_dict[rot_repr]
            return

        if rot_repr.startswith("ce"):
            token = rot_repr.split("_")
            self.num_classes = int(token[1])
            self.loss_fn = self._loss_cross_entropy
            return

        raise NotImplementedError(f"No loss function configured for {rot_repr}.")

    def _loss_sincos(self, batch_dict):
        delta_rot = batch_dict["delta_rot"]
        yaws_out = batch_dict["out_rotation"]

        # diff_rot = (anchor_yaws - positive_yaws)
        diff_rot = delta_rot[:, 2]
        diff_rot_cos = torch.cos(diff_rot)
        diff_rot_sin = torch.sin(diff_rot)
        yaws_out_cos = yaws_out[:, 0]
        yaws_out_sin = yaws_out[:, 1]
        loss_rot = self.reg_loss(yaws_out_sin, diff_rot_sin).mean()
        loss_rot = loss_rot + self.reg_loss(yaws_out_cos, diff_rot_cos).mean()
        return loss_rot

    def _loss_sincos_atan(self, batch_dict):
        delta_rot = batch_dict["delta_rot"]
        yaws_out = batch_dict["out_rotation"]

        # diff_rot = (anchor_yaws - positive_yaws)
        diff_rot = delta_rot[:, 2]
        yaws_out_cos = yaws_out[:, 0]
        yaws_out_sin = yaws_out[:, 1]
        yaws_out_final = torch.atan2(yaws_out_sin, yaws_out_cos)
        diff_rot_atan = torch.atan2(diff_rot.sin(), diff_rot.cos())
        loss_rot = self.reg_loss(yaws_out_final, diff_rot_atan).mean()
        return loss_rot

    @staticmethod
    def _loss_yaw(batch_dict):
        delta_rot = batch_dict["delta_rot"]
        yaws_out = batch_dict["out_rotation"]

        # diff_rot = (anchor_yaws - positive_yaws) % (2*np.pi)
        diff_rot = delta_rot[:, 2] % (2*np.pi)
        yaws_out = yaws_out % (2*np.pi)
        loss_rot = torch.abs(diff_rot - yaws_out)
        loss_rot[loss_rot > np.pi] = 2*np.pi - loss_rot[loss_rot > np.pi]
        loss_rot = loss_rot.mean()
        return loss_rot

    def _loss_cross_entropy(self, batch_dict):
        delta_rot = batch_dict["delta_rot"]
        yaws_out = batch_dict["out_rotation"]

        yaw_out_bins = yaws_out[:, :-1]
        yaw_out_delta = yaws_out[:, -1]
        bin_size = 2*np.pi / self.num_classes
        # diff_rot = (anchor_yaws - positive_yaws) % (2*np.pi)
        diff_rot = delta_rot[:, 2] % (2*np.pi)
        gt_bins = torch.zeros(diff_rot.shape[0], dtype=torch.long, device=yaws_out.device)
        for i in range(self.num_classes):
            lower_bound = i * bin_size
            upper_bound = (i+1) * bin_size
            indexes = (diff_rot >= lower_bound) & (diff_rot < upper_bound)
            gt_bins[indexes] = i
        gt_delta = diff_rot - bin_size*gt_bins

        loss_rot_fn = torch.nn.CrossEntropyLoss()
        loss_rot = loss_rot_fn(yaw_out_bins, gt_bins) + self.reg_loss(yaw_out_delta, gt_delta).mean()
        return loss_rot

    def __call__(self, batch_dict, **_):
        return self.loss_fn(batch_dict)


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


class MyCircleLoss(PyCircleLoss):
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

    @staticmethod
    def logsumexp(x: torch.Tensor, keep_mask=None, add_one: bool = True, dim: int = 1):
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
            inside_log[inside_log == 0] = torch.exp(-max_vals[inside_log == 0])
        return torch.log(inside_log) + max_vals

    def _compute_loss(self, dist_mat, pos_mask, neg_mask):
        pos_mask_bool = pos_mask.bool()
        neg_mask_bool = neg_mask.bool()
        anchor_positive = dist_mat[pos_mask_bool]
        anchor_negative = dist_mat[neg_mask_bool]

        if self.version == 'PML':
            new_mat = torch.zeros_like(dist_mat)
            new_mat[pos_mask_bool] = -self.gamma * torch.relu(self.op - anchor_positive.detach()) * \
                (anchor_positive - self.delta_p)
            new_mat[neg_mask_bool] = self.gamma * torch.relu(anchor_negative.detach() - self.on) * \
                (anchor_negative - self.delta_n)

            # losses = self.soft_plus(
            #     self.logsumexp(new_mat, keep_mask=pos_mask, add_one=False, dim=1) + \
            #     self.logsumexp(new_mat, keep_mask=neg_mask, add_one=False, dim=1)
            # )

            losses = self.soft_plus(self.logsumexp(new_mat, keep_mask=pos_mask, add_one=False, dim=1) +
                                    self.logsumexp(new_mat, keep_mask=neg_mask, add_one=False, dim=1))

            zero_rows = torch.where((torch.sum(pos_mask, dim=1) == 0) | (torch.sum(neg_mask, dim=1) == 0))[0]
            final_mask = torch.ones_like(losses)
            final_mask[zero_rows] = 0
            losses = losses*final_mask
            return losses
            # return {"loss":
            #   {"losses": losses, "indices": c_f.torch_arange_from_size(new_mat), "reduction_type": "element"}
            # }
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

            anchor_positive.backward(gradient=z * (- ap) * torch.softmax(logit_p, dim=0) * self.gamma,
                                     retain_graph=True)
            anchor_negative.backward(gradient=z * an * torch.softmax(logit_n, dim=0) * self.gamma, retain_graph=True)
            return loss.detach()

    def my_compute_loss(self, dist_mat, pos_mask, neg_mask):
        return self._compute_loss(dist_mat, pos_mask, neg_mask)


class SmoothMetricLossV2(nn.Module):
    def __init__(self, margin):
        super(SmoothMetricLossV2, self).__init__()
        self.margin = margin

    def forward(self, embeddings, _, neg_mask, other_embeddings=None):
        """

        Args:
            embeddings: Embedding of shape 2*N, i and i+N should be positive pairs
            _: Positive mask, unused
            neg_mask:
            other_embeddings:

        Returns:

        """

        if other_embeddings is None:
            other_embeddings = embeddings
        # CARE: when using dot, embedding should be normalized (i guess hehe)
        # d = pairwise_mse(embeddings, other_embeddings) + 1e-5
        d = torch.cdist(embeddings, other_embeddings, p=2)
        # d = torch.sqrt(D)
        # marg_d = self.margin - d
        # batch_size = embeddings.shape[0]
        j_all = []

        for i in range(embeddings.shape[0]//2):

            matching_idx = i+embeddings.shape[0]//2
            ap_distance = d[i, matching_idx]  # .sqrt()

            neg_d_1 = self.margin - d[i, neg_mask[i]]  # .sqrt()
            neg_d_2 = self.margin - d[matching_idx, neg_mask[matching_idx]]  # .sqrt()
            # j_ij = neg_d - neg_d.max()  # Why did i add this?
            # j_ij = torch.exp(j_ij).sum()
            j_ij_1 = torch.exp(neg_d_1).sum()
            j_ij_2 = torch.exp(neg_d_2).sum()
            j_ij = (j_ij_1 + j_ij_2).log() + ap_distance
            if torch.any(torch.isnan(j_ij)):
                print("NaN found")
            else:
                j_all.append(j_ij)

        j_all = torch.stack(j_all)
        loss = F.relu(j_all).pow(2).mean().div(2)
        return loss


class NPairLoss(nn.Module):
    def __init__(self):
        super(NPairLoss, self).__init__()

    @staticmethod
    def forward(embeddings, _, neg_mask, other_embeddings=None):
        """

        Args:
            embeddings: Embedding of shape 2*N, i and i+N should be positive pairs
            _: Positive mask, unused.
            neg_mask:
            other_embeddings:

        Returns:

        """

        if other_embeddings is None:
            other_embeddings = embeddings
        # CARE: when using dot, embedding should be normalized (i guess hehe)
        # d = pairwise_mse(embeddings, other_embeddings) + 1e-5
        d = torch.mm(embeddings, torch.transpose(other_embeddings, 0, 1))
        # d = torch.sqrt(d)
        # marg_D = self.margin - d
        # batch_size = embeddings.shape[0]
        j_all = []

        for i in range(embeddings.shape[0]//2):

            matching_idx = i+embeddings.shape[0]//2
            ap_distance = d[i, matching_idx]  # .sqrt()

            # expm = torch.exp(ap_distance - d)
            expm = torch.exp(d - ap_distance)
            j_ij = expm[i, neg_mask[i]].sum()

            j_ij = (j_ij + 1).log()
            if torch.any(torch.isnan(j_ij)):
                print("NaN found")
            else:
                j_all.append(j_ij)

        j_all = torch.stack(j_all)
        loss = j_all.mean()
        return loss


class CircleLoss(nn.Module):
    def __init__(self, version='PML', m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
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
        # batch_size = embeddings.shape[0] // 2
        # a1_idx, p_idx = torch.where(torch.eye(batch_size).repeat(1, 2))
        # a2_idx, n_idx = torch.where(neg_idxs)
        # a1_idx, p_idx = a1_idx.to(embeddings.device), p_idx.to(embeddings.device)
        # a2_idx, n_idx = a2_idx.to(embeddings.device), n_idx.to(embeddings.device)
        d = self.loss_fn.distance(embeddings, other_embeddings)
        loss = self.loss_fn.my_compute_loss(d, pos_mask, neg_mask)
        if self.version == 'PML':
            nonzero_idx = loss > 0
            if nonzero_idx.sum() == 0.:
                return loss.mean() * 0
            return loss[nonzero_idx].mean()
        else:
            return loss


class MetricLoss:
    def __init__(self, cfg):

        if cfg['loss_type'].startswith('triplet'):
            neg_selector = random_negative_selector
            if 'hardest' in cfg['loss_type']:
                neg_selector = hardest_negative_selector
            if 'semihard' in cfg['loss_type']:
                neg_selector = semihard_negative_selector
            loss_fn = TripletLoss(cfg['margin'], neg_selector, distances.LpDistance())
        elif cfg['loss_type'] == 'lifted':
            loss_fn = SmoothMetricLossV2(cfg['margin'])
        elif cfg['loss_type'] == 'npair':
            loss_fn = NPairLoss()
        elif cfg['loss_type'].startswith('circle'):
            version = cfg['loss_type'].split('_')[1]
            loss_fn = CircleLoss(version)
        else:
            raise NotImplementedError(f"Loss {cfg['loss_type']} not implemented")

        self.loss_function = loss_fn
        self.norm_embeddings = cfg['norm_embeddings']

    def __call__(self, batch_dict, *, mode, **_):

        model_out = batch_dict["out_embedding"]
        neg_mask = batch_dict["neg_mask"]

        if self.norm_embeddings:
            model_out = model_out / model_out.norm(dim=1, keepdim=True)

        pos_mask = torch.zeros((model_out.shape[0], model_out.shape[0]), device=model_out.device)
        if mode == 'triplets':
            batch_size = (model_out.shape[0]//3)
            for i in range(batch_size):
                pos_mask[i, i + batch_size] = 1
        elif mode == 'pairs':
            batch_size = (model_out.shape[0]//2)
            for i in range(batch_size):
                pos_mask[i, i + batch_size] = 1
                pos_mask[i+batch_size, i] = 1

        return self.loss_function(model_out, pos_mask, neg_mask)


class LossFunction:
    def __init__(self, function: Callable, *, label: str, weight: float, batch_keys: List,
                 rev_batch_keys: Optional[List] = None, reverse_loss: bool = False):

        self._function = function
        self.label = label
        self.weight = float(weight)

        self.batch_keys = batch_keys.copy()
        if reverse_loss and rev_batch_keys is not None:
            self.batch_keys.extend(rev_batch_keys.copy())

    def _tensors_in_batch(self, batch_dict: Dict):
        for k in self.batch_keys:
            if k not in batch_dict:
                raise KeyError(f"Unable to compute loss {self.label}. "
                               f"Required Tensor {k} not found in Batch dictionary.")

            if batch_dict[k] is None:
                raise ValueError(f"Unable to compute loss {self.label}. Required Tensor {k} has a value of None.")

    def __call__(self, batch_dict, *, mode, reverse_loss=False):
        # Check if the required tensors for the loss function exist and are not None in the batch dict.
        self._tensors_in_batch(batch_dict)

        return self._function(batch_dict, mode=mode, reverse_loss=reverse_loss)


class TotalLossFunction:

    def __init__(self, cfg):

        reverse_loss = cfg.get("inv_tf_weight", False)
        if isinstance(reverse_loss, bool):
            self.reverse_loss = reverse_loss
        else:
            self.reverse_loss = reverse_loss > 0.

        self.rot_representation = cfg.get("rot_representation", "6dof")
        self.batch_size = cfg["batch_size"]
        self.mode = cfg["mode"]
        self.tuple_size = 3 if self.mode == "triplets" else 2

        self.reg_loss = torch.nn.SmoothL1Loss(reduction='none')

        self.loss_functions = []

        if cfg['weight_transl'] > 0. and cfg['rot_representation'] != '6dof':
            self.loss_functions.append(
                LossFunction(loss_transl, label="Translation", weight=cfg["weight_transl"],
                             batch_keys=["out_translation", "transl_diff"])
            )

        if cfg["weight_rot"] > 0.:
            rot_repr = cfg["rot_representation"]

            if rot_repr in ["sincos", "sincos_atan", "yaw"] or rot_repr.startswith("ce"):
                loss_rot = RotationLoss(cfg)
                rot_batch_keys = ["delta_rot", "out_rotation"]
                rot_rev_batch_keys = []
            elif rot_repr in ["quat", "bingham"]:
                loss_rot = QuatRotationLoss(cfg)
                rot_batch_keys = [ "delta_quat", "out_rotation"]
                rot_rev_batch_keys = []
            elif rot_repr == "6dof":
                loss_rot = loss_pose
                rot_batch_keys = ["transformation", "delta_pose"]
                rot_rev_batch_keys = ["transformation2"]
            else:
                raise NotImplementedError(f"No loss function configured for {rot_repr}.")

            self.loss_functions.append(
                LossFunction(loss_rot, label="Rotation", weight=cfg["weight_rot"],
                             batch_keys=rot_batch_keys, rev_batch_keys=rot_rev_batch_keys,
                             reverse_loss=self.reverse_loss)
            )

            if cfg['head'] in ["SuperGlue", "Transformer", "PyTransformer", "TFHead", "MLFeatTF"] \
                    and cfg['sinkhorn_aux_loss']:
                self.loss_functions.append(
                    LossFunction(loss_sinkhorn_matches, label="Aux Matches", weight=0.05 * cfg["weight_rot"],
                                 batch_keys=["sinkhorn_matches", "delta_pose", "point_coords", "batch_size"],
                                 rev_batch_keys=["sinkhorn_matches2"], reverse_loss=self.reverse_loss)
                )

            if cfg['head'] == "SuperGlue" and cfg['sinkhorn_type'] == 'slack':
                self.loss_functions.append(
                    LossFunction(loss_sinkhorn_inlier, label="Sinkhorn Inlier", weight=0.01 * cfg["weight_rot"],
                                 batch_keys=["transport"], rev_batch_keys=["transport2"],
                                 reverse_loss=self.reverse_loss)
                )

        if cfg['panoptic_weight'] > 0.:
            self.loss_functions.append(
                LossFunction(loss_panoptic, label="Panoptic Mismatch", weight=cfg["panoptic_weight"],
                             batch_keys=["transport", "transport2", "keypoint_idxs",
                                         "anchor_panoptic", "positive_panoptic"], reverse_loss=self.reverse_loss)
            )

        if cfg["semantic_weight"] > 0.:
            self.loss_functions.append(
                LossFunction(loss_semantic, label="Semantic Mismatch", weight=cfg["semantic_weight"],
                             batch_keys=["transport", "class_one_hot_map", "anchor_semantic", "positive_semantic"],
                             rev_batch_keys=["transport2"], reverse_loss=self.reverse_loss)
            )

        if cfg["supersem_weight"] > 0.:
            self.loss_functions.append(
                LossFunction(loss_metasemantic, label="Meta-Semantic Mismatch", weight=cfg["supersem_weight"],
                             batch_keys=["transport", "superclass_one_hot_map", "anchor_supersem", "positive_supersem"],
                             rev_batch_keys=["transport2"], reverse_loss=self.reverse_loss)
            )

        if cfg["weight_metric_learning"] > 0.:
            metric_loss_fn = MetricLoss(cfg)
            self.loss_functions.append(
                LossFunction(metric_loss_fn, label="Metric Learning", weight=cfg["weight_metric_learning"],
                             batch_keys=["out_embedding", "neg_mask"]
                )
            )

    @staticmethod
    def loss_has_nan(loss, *, label):
        if torch.any(torch.isnan(loss)):
            raise NaNLossError(f"{label} contains a NaN.")

    def compute_losses(self, batch_dict):
        loss_dict = {
           "total": 0,
           "aux": {}
        }

        for loss_func in self.loss_functions:

            # Compute the loss(es).
            losses = loss_func(batch_dict, mode=self.mode, reverse_loss=self.reverse_loss)
            if not isinstance(losses, dict):
                losses = {"": losses}

            subtotal_loss = 0
            num_losses = float(len(losses.keys()))

            for suffix, loss in losses.items():
                loss_label = "Loss: " + loss_func.label
                if suffix != "":
                    loss_label += " " + suffix

                # Check for NANs. Raise exception if so, to avoid backpropagating it
                self.loss_has_nan(loss, label=loss_label)
                loss_dict["aux"][loss_label] = loss
                subtotal_loss += loss

            # Average losses in case there was more than one (e.g. for reverse losses)
            if num_losses > 1:
                subtotal_loss /= num_losses
            loss_dict["total"] += loss_func.weight * subtotal_loss

        return loss_dict

    def __call__(self, batch_dict):
        return self.compute_losses(batch_dict)

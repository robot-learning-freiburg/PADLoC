import random
import torch
import torch.nn.functional as F
import numpy as np
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
# from panoptic.epsnet2.scripts.load_scan import scan_to_epsnet


class SVDNonConvergenceError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NaNLossError(Exception):
    pass


def gather_nd(params, indices, name=None):
    """
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    """

    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    return torch.take(params, idx)


def pairwise_mse(x, y=None):
    """
    Input: x is a Nxd tensor
           y is an optional Mxd tensor
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def pairwise_L2(x, y=None):
    if y is None:
        y = x
    dist = pairwise_mse(x, y)
    dist = torch.sqrt(dist + 1e-5)
    return dist


def pairwise_dot(x, y=None):
    if y is None:
        y = x
    dot = torch.mm(x, torch.transpose(y, 0, 1))
    #return 1. - dot
    return dot


def pairwise_cosine(x, y=None):
    if y is None:
        y = x
    x_norm = x / x.norm(dim=1).unsqueeze(1)
    y_norm = y / y.norm(dim=1).unsqueeze(1)
    res = 1. - torch.mm(x_norm, y_norm.transpose(0, 1))
    return res


def softmax_cross_entropy(logits, target, mean=True):
    loss = torch.sum(- target * F.log_softmax(logits, -1), -1)
    if mean:
        loss = loss.mean()
    return loss


def pairwise_batch_mse(x, y=None):
    """
    Input: x is a Nxd tensor
           y is an optional Mxd tensor
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = x.pow(2).sum(2, True)
    if y is not None:
        y_norm = y.pow(2).sum(2, True)
    else:
        y = x
        y_norm = x_norm.transpose(1, 2)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
    return dist


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def update_bn_swa(model, sample, exp_cfg, device):
    model.train()

    if exp_cfg['training_type'] == "3D":
        anchor_list = []
        for i in range(len(sample['anchor'])):
            anchor = sample['anchor'][i].to(device)

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                a = anchor_set[0, :].long()
                anchor_i = anchor[a]
            else:
                anchor_i = anchor

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor_list.append(anchor_i[:, :3].unsqueeze(0))
            else:
                if exp_cfg['use_semantic'] or exp_cfg['use_panoptic']:
                    anchor_i = torch.cat((anchor_i, sample['anchor_logits'][i].to(device)), dim=1)
                anchor_list.append(model.module.module.backbone.prepare_input(anchor_i))
                del anchor_i

        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor = torch.cat(tuple(anchor_list), 0)
            model_in = anchor
            model_in = model_in / 100.
        else:
            model_in = KittiDataset.collate_batch(anchor_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

        batch_dict = model(model_in, metric_head=True)


def run_panoptic(panoptic_model, sample, device, make_panoptic, post, exp_cfg):
    imgs = []
    p_xs = []
    p_ys = []
    proj_ranges = []
    unproj_ranges = []
    n_points = []

    sample['anchor_logits'] = []
    for sample_id in range(len(sample['anchor'])):
        anchor = sample['anchor'][sample_id]
        res_a = scan_to_epsnet(anchor.cpu().numpy())
        imgs.append(res_a['img'])
        p_xs.append(res_a['p_x'])
        p_ys.append(res_a['p_y'])
        proj_ranges.append(res_a['proj_range'])
        unproj_ranges.append(res_a['unproj_range'])
        n_points.append(res_a['n_points'])
    if 'positive' in sample:
        sample['positive_logits'] = []
        for sample_id in range(len(sample['positive'])):
                positive = sample['positive'][sample_id]
                res_p = scan_to_epsnet(positive.cpu().numpy())
                imgs.append(res_p['img'])
                p_xs.append(res_p['p_x'])
                p_ys.append(res_p['p_y'])
                proj_ranges.append(res_p['proj_range'])
                unproj_ranges.append(res_p['unproj_range'])
                n_points.append(res_p['n_points'])
    imgs = torch.stack(imgs).to(device)
    p_xs = torch.stack(p_xs).to(device)
    p_ys = torch.stack(p_ys).to(device)
    proj_ranges = torch.stack(proj_ranges).to(device)
    unproj_ranges = torch.stack(unproj_ranges).to(device)
    with torch.no_grad():
        _, pred, _ = panoptic_model(imgs)
    logits_list = []
    for i, (sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits) in enumerate(zip(
            pred["sem_pred"], pred["bbx_pred"], pred["cls_pred"], pred["obj_pred"], pred["msk_pred"], pred["sem_logits"])):

        # Compute panoptic output
        panoptic_pred = make_panoptic(sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, 11)
        sem_logits = sem_logits.unsqueeze(0) #functional.interpolate(sem_logits.unsqueeze(0), size=(64, 2048), mode="bilinear", align_corners=False)

        panop_logits = panoptic_pred[-1].unsqueeze(0) #functional.interpolate(panoptic_pred[-1].unsqueeze(0), size=(64, 2048), mode="bilinear", align_corners=False)

        ################################################## Simple Mapping
        unproj_sem_logits = sem_logits[0, :, p_ys[i, :n_points[i]], p_xs[i, :n_points[i]]]
        unproj_panoptic_logits = panop_logits[0, :, p_ys[i, :n_points[i]], p_xs[i, :n_points[i]]]
        unproj_panoptic_logits = torch.nn.functional.pad(unproj_panoptic_logits, [0, 0, 0, 56-unproj_panoptic_logits.shape[0]])

        if exp_cfg['use_semantic'] and exp_cfg['use_panoptic']:
            logits = torch.cat((unproj_sem_logits, unproj_panoptic_logits))
        elif exp_cfg['use_semantic']:
            logits = unproj_sem_logits
        elif exp_cfg['use_panoptic']:
            logits = unproj_panoptic_logits
        logits_list.append(logits.T.clone())
    for sample_id in range(len(sample['anchor'])):
        sample['anchor_logits'].append(logits_list[sample_id])
    if 'positive' in sample:
        for sample_id in range(len(sample['positive'])):
            sample['positive_logits'].append(logits_list[len(sample['anchor'])+sample_id])
    return sample


def set_seed(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2**32 - 1)
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

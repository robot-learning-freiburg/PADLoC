import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import soft_kronecker
from utils.tools import SVDNonConvergenceError


class PointNetHead(nn.Module):

    def __init__(self, input_dim, points_num, rotation_parameters=2):
        super().__init__()
        # self.FC1 = nn.Linear(input_dim * points_num * 2, int(input_dim / 4))
        self.relu = nn.ReLU()
        self.FC1 = nn.Linear(input_dim * 2, 1024)
        self.FC2 = nn.Linear(1024, 512)
        # self.FC2_transl = nn.Linear(int(input_dim / 4), 3)  # x,y,z
        # self.FC2_rot = nn.Linear(int(input_dim / 4), 1)  # yaw
        self.FC_transl = nn.Linear(512, 3)
        self.FC_rot = nn.Linear(512, rotation_parameters)

    def forward(self, x, compute_transl=True, compute_rotation=True):
        # x = torch.squeeze(x, 3)
        x = x.view(x.shape[0], -1)

        x = self.relu(self.FC1(x))
        x = self.relu(self.FC2(x))
        # transl = self.FC2_transl(x)
        # yaw = self.FC2_rot(x)
        if compute_transl:
            transl = self.FC_transl(x)
        else:
            transl = None

        if compute_rotation:
            yaw = self.FC_rot(x)
        else:
            yaw = None
        # print(transl, yaw)
        return transl, yaw


class CorrelationHead(nn.Module):

    def __init__(self, input_dim, points_num, rotation_parameters=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.FC1 = nn.Conv1d(points_num, 128, kernel_size=1)
        self.FC2 = nn.Conv1d(128, 128, kernel_size=1)
        self.FC3 = nn.Conv1d(128, 96, kernel_size=1)
        self.FC4 = nn.Conv1d(96, 64, kernel_size=1)
        self.FC5 = nn.Conv1d(64, 32, kernel_size=1)
        self.FC6 = nn.Conv1d(32, 1, kernel_size=1)

        self.MFC1 = nn.Conv1d(1289, 256, kernel_size=1)
        self.MFC2 = nn.Conv1d(256, 256, kernel_size=1)
        self.MFC3 = nn.Conv1d(256, 3, kernel_size=1)

        self.FC7 = nn.Linear(points_num*3, 256)
        self.FC8 = nn.Linear(256, 256)
        self.FC9 = nn.Linear(256, rotation_parameters)

    def forward(self, feat1, feat2, batch_dict, compute_transl=True, compute_rotation=True):
        # x = torch.squeeze(x, 3)
        B, C, NUM, _ = feat1.shape
        corr = torch.bmm(feat1.squeeze().permute(0, 2, 1), feat2.squeeze()) / math.sqrt(C)

        x = self.relu(self.FC1(corr))
        x = self.relu(self.FC2(x))
        x = self.relu(self.FC3(x))
        x = self.relu(self.FC4(x))
        x = self.relu(self.FC5(x))
        flow = self.FC6(x)

        H = NUM
        W = 1
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)

        grid = torch.cat((xx, yy), 1)
        vgrid = grid.float().clone().to(feat1.device)
        # vgrid[:,1,:,0] = flow[:,0,:]

        vgrid[:, 0, :, :] = 0.
        vgrid[:, 1, :, 0] = flow[:,0,:]
        vgrid = vgrid.permute(0, 2, 3, 1)

        warped_feat2 = nn.functional.grid_sample(feat2, vgrid, 'nearest')
        coords = batch_dict['point_coords'].view(2*B, -1, 4)
        coords1 = coords[:B, :, 1:].permute(0,2,1).unsqueeze(-1)
        coords2 = coords[B:, :, 1:]
        warped_coords2 = nn.functional.grid_sample(coords2.permute(0,2,1).unsqueeze(-1), vgrid, 'nearest')

        x = torch.cat((coords1, warped_coords2, warped_coords2-coords1, feat1, warped_feat2), dim=1).squeeze()

        x = self.relu(self.MFC1(x))
        x = self.relu(self.MFC2(x))
        x = self.relu(self.MFC3(x))

        x = x.view(B, -1)
        # transl = self.FC2_transl(x)
        # yaw = self.FC2_rot(x)
        transl = None

        if compute_rotation:
            yaw = self.relu(self.FC7(x))
            yaw = self.relu(self.FC8(yaw))
            yaw = self.FC9(yaw)
        else:
            yaw = None
        # print(transl, yaw)
        return transl, yaw


# This function is from https://github.com/yewzijian/RPMNet/
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets
    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    # try:
    #     u, s, v = torch.svd(cov, some=False, compute_uv=True)
    # except:
    #     # Add some small random turbulence to improve convergence
    #     u, s, v = torch.svd(cov + 1e-4 * cov.mean() * torch.rand_like(cov), some=False, compute_uv=True)
    u, s, v = torch.svd(cov, some=False, compute_uv=True)

    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


# This function is from https://github.com/yewzijian/RPMNet/
def sinkhorn_slack_variables(feature1, feature2, beta, alpha, n_iters: int = 5, slack: bool = True) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)
    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    B, N, _ = feature1.shape
    _, M, _ = feature2.shape

    # Feature normalization
    feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
    feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)

    dist = -2 * torch.matmul(feature1, feature2.permute(0, 2, 1))
    dist += torch.sum(feature1 ** 2, dim=-1)[:, :, None]
    dist += torch.sum(feature2 ** 2, dim=-1)[:, None, :]

    log_alpha = -beta * (dist - alpha)

    # Sinkhorn iterations
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

    return log_alpha


# This function is from https://github.com/valeoai/FLOT/
def sinkhorn_unbalanced(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter,
                        semantic1=None, semantic2=None, semantic_weight=0.5,
                        supersem1=None, supersem2=None, supersem_weight=1):
    """
    Sinkhorn algorithm
    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost.
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost.
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.
    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    # Squared l2 distance between points points of both point clouds
    # distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    # distance_matrix = distance_matrix + torch.sum(
    #     pcloud2 ** 2, -1, keepdim=True
    # ).transpose(1, 2)
    # distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    # Force transport to be zero for points further than 10 m apart
    # support = (distance_matrix < 10 ** 2).float()

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    normalizer = 1
    # Add a semantic class mismatch cost
    if semantic1 is not None and semantic2 is not None:
        semanticC = 1 - soft_kronecker(semantic1, semantic2)

        C = C + semantic_weight * semanticC
        normalizer += semantic_weight

    if supersem1 is not None and supersem2 is not None:
        supersemC = 1 - soft_kronecker(supersem1, supersem2)

        C = C + supersem_weight * supersemC
        normalizer += supersem_weight

    if normalizer != 1:
        normalizer = 1 / normalizer
        C = C / normalizer

    # Entropic regularisation
    K = torch.exp(-C / epsilon) #* support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob1 = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
            )
            / K.shape[1]
    )
    prob2 = (
            torch.ones(
                (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
            )
            / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T


class UOTHead(nn.Module):

    def __init__(self, input_dim, points_num, **kwargs):
        #rotation_parameters=2, nb_iter=5, use_svd=False, sinkhorn_type='unbalanced',
         #        semantic_cost=False, semantic_weight=None):
        super().__init__()
        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.nb_iter = kwargs.get("nb_iter") or 5
        self.use_svd = kwargs.get("use_svd", False)
        self.semantic_cost = kwargs.get("semantic_matching_cost", False)

        self.semantic_weight = None
        self.supersem_weight = None
        if self.semantic_cost:
            self.semantic_weight = kwargs.get("semantic_weight") or torch.nn.Parameter(torch.zeros(1))
            self.supersem_weight = kwargs.get("supersem_weight") or torch.nn.Parameter(torch.zeros(1))

        self.sinkhorn_type = kwargs.get("sinkhorn_type") or "unbalanced"
        self.rotation_parameters = kwargs.get("rotation_parameters") or 2

        if not self.use_svd:
            self.FC1 = nn.Linear(points_num*9, 512)
            self.FC2 = nn.Linear(512, 256)
            self.FC3 = nn.Linear(256, self.rotation_parameters)
            self.relu = nn.ReLU()

    def forward(self, batch_dict, compute_transl=True, compute_rotation=True, src_coords=None, mode='pairs'):

        # torch.cuda.synchronize()
        # time1 = time.time()
        feats = batch_dict['point_features'].squeeze(-1)
        B, C, NUM = feats.shape

        semantic1 = semantic2 = None
        supersem1 = supersem2 = None
        keypoint_idx = None

        if self.semantic_cost:
            keypoint_idx = batch_dict['keypoint_idxs']

        if mode == 'pairs':
            assert B % 2 == 0, "Batch size must be multiple of 2: B anchor + B positive samples"
            B = B // 2
            feat1 = feats[:B]
            feat2 = feats[B:]

            coords = batch_dict['point_coords'].view(2*B, -1, 4)
            coords1 = coords[:B, :, 1:]
            coords2 = coords[B:, :, 1:]

            if self.semantic_cost:
                semantic1 = torch.stack([s[i] for s, i in zip(batch_dict['anchor_semantic'], keypoint_idx[:B])]).view(B, -1)
                semantic2 = torch.stack([s[i] for s, i in zip(batch_dict['positive_semantic'], keypoint_idx[B:])]).view(B, -1)
                supersem1 = torch.stack([s[i] for s, i in zip(batch_dict['anchor_supersem'], keypoint_idx[:B])]).view(B, -1)
                supersem2 = torch.stack([s[i] for s, i in zip(batch_dict['positive_supersem'], keypoint_idx[B:])]).view(B, -1)
        else:
            assert B % 3 == 0, "Batch size must be multiple of 3: B anchor + B positive + B negative samples"
            B = B // 3
            feat1 = feats[:B]
            feat2 = feats[B:2*B]

            coords = batch_dict['point_coords'].view(3*B, -1, 4)
            coords1 = coords[:B, :, 1:]
            coords2 = coords[B:2*B, :, 1:]

            if self.semantic_cost:
                semantic1 = [s[i] for s, i in zip(batch_dict['anchor_semantic'], keypoint_idx[:B])]
                semantic2 = [s[i] for s, i in zip(batch_dict['positive_semantic'], keypoint_idx[B:2*B])]
                supersem1 = torch.stack([s[i] for s, i in zip(batch_dict['anchor_supersem'], keypoint_idx[:B])]).view(B, -1)
                supersem2 = torch.stack([s[i] for s, i in zip(batch_dict['positive_supersem'], keypoint_idx[B:2*B])]).view(B, -1)

        if self.sinkhorn_type == 'unbalanced':

            semantic_weight = self.semantic_weight
            if isinstance(semantic_weight, nn.Parameter):
                semantic_weight = torch.exp(semantic_weight)

            supersem_weight = self.supersem_weight
            if isinstance(supersem_weight, nn.Parameter):
                supersem_weight = torch.exp(supersem_weight)

            transport = sinkhorn_unbalanced(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                coords1,
                coords2,
                epsilon=torch.exp(self.epsilon) + 0.03,
                gamma=torch.exp(self.gamma),
                max_iter=self.nb_iter,
                semantic1=semantic1,
                semantic2=semantic2,
                semantic_weight=semantic_weight,
                supersem1=supersem1,
                supersem2=supersem2,
                supersem_weight=supersem_weight
            )
        else:
            transport = sinkhorn_slack_variables(
                feat1.permute(0, 2, 1),
                feat2.permute(0, 2, 1),
                F.softplus(self.epsilon),
                F.softplus(self.gamma),
                self.nb_iter,
            )
            transport = torch.exp(transport)

        row_sum = transport.sum(-1, keepdim=True)

        sinkhorn_matches = (transport @ coords2) / (row_sum + 1e-8)
        ot_flow = sinkhorn_matches - coords1

        batch_dict['sinkhorn_matches'] = sinkhorn_matches
        batch_dict['transport'] = transport

        if not self.use_svd:
            x = torch.cat((coords1, sinkhorn_matches, ot_flow), dim=2)
            x = x.view(B, -1)

            x = self.relu(self.FC1(x))
            x = self.relu(self.FC2(x))
            x = self.FC3(x)

            batch_dict['out_rotation'] = x
            batch_dict['out_translation'] = None
        else:
            if src_coords is None:
                src_coords = coords1
            try:
                transformation = compute_rigid_transform(src_coords, sinkhorn_matches, row_sum.squeeze(-1))
            except RuntimeError as e:
                print("\n\n\n" + "="*80)
                print("SVD did not converge!!!!!")
                print("exp(epsilon):", torch.exp(self.epsilon))
                print("exp(gamma):", torch.exp(self.gamma))
                print("\n\n\ntransport: ", transport)
                print("\n\n\nsrc_coords: ", src_coords)
                print("\n\n\nsinkhorn matches:   ", sinkhorn_matches)
                print("\n\n\nrow_sum.squeeze(-1):   ", row_sum.squeeze(-1))
                print("\n\n\n")

                raise SVDNonConvergenceError("SVD did not converge!")

            batch_dict['transformation'] = transformation
            batch_dict['out_rotation'] = None
            batch_dict['out_translation'] = None

        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("Took: ", time2-time1)

        return batch_dict

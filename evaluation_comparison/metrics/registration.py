from functools import partial
from typing import Callable, List, Optional, Tuple

import open3d as o3d
try:
    reg_module = o3d.pipelines.registration
    o3d_feat_ransac = partial(reg_module.registration_ransac_based_on_feature_matching, mutual_filter=True)
except AttributeError:
    reg_module = o3d.registration
    o3d_feat_ransac = reg_module.registration_ransac_based_on_feature_matching
import torch
import torch.nn.functional as F

# Type hints
Transform = torch.Tensor
O3DRegResult = reg_module.RegistrationResult
RegResult = Tuple[Transform, O3DRegResult]
BatchRegResult = Tuple[Transform, List[O3DRegResult]]


def ransac_registration(*,
                        anc_coords: torch.Tensor,
                        anc_feats: torch.Tensor,
                        pos_coords: torch.Tensor,
                        pos_feats: torch.Tensor,
                        initial_transformation: Optional[torch.Tensor] = None,
                        ) -> RegResult:

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anc_coords.cpu().numpy())
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pos_coords.cpu().numpy())
    pcd1_feat = reg_module.Feature()
    pcd1_feat.data = anc_feats.permute(0, 1).detach().cpu().numpy()
    pcd2_feat = reg_module.Feature()
    pcd2_feat.data = pos_feats.permute(0, 1).detach().cpu().numpy()

    torch.cuda.synchronize()

    estimation_method = reg_module.TransformationEstimationPointToPoint(False)
    convergence_criteria = reg_module.RANSACConvergenceCriteria(5000)
    # time_ransac.tic()
    result = o3d_feat_ransac(
        source=pcd2, target=pcd1,
        source_feature=pcd2_feat, target_feature=pcd1_feat,
        max_correspondence_distance=0.6,
        estimation_method=estimation_method,
        ransac_n=3, checkers=[],
        criteria=convergence_criteria
    )

    # time_ransac.toc()
    transformation = torch.tensor(result.transformation.copy())
    return transformation, result


def icp_registration(*,
                     anc_coordinates: torch.Tensor,
                     pos_coordinates: torch.Tensor,
                     initial_transformation: torch.Tensor
                     ) -> RegResult:

    p1 = o3d.geometry.PointCloud()
    p1.points = o3d.utility.Vector3dVector(anc_coordinates.cpu().numpy())
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(pos_coordinates.cpu().numpy())

    # time_icp.tic()
    result = reg_module.registration_icp(
        p2, p1, 0.1, initial_transformation.cpu().numpy(),
        reg_module.TransformationEstimationPointToPoint())
    # time_icp.toc()

    transformation = torch.tensor(result.transformation.copy())

    return transformation, result


def batch_coord_feat_registration(*,
                                  reg_func: Callable,
                                  batch_coords: torch.Tensor,
                                  batch_feats: torch.Tensor,
                                  batch_size: int,
                                  initial_transformations: Optional[torch.Tensor] = None,
                                  ) -> BatchRegResult:
    transformations = []
    results = []
    for i in range(batch_size // 2):
        coords1 = batch_coords[i]
        coords2 = batch_coords[i + batch_size // 2]
        feat1 = batch_feats[i]
        feat2 = batch_feats[i + batch_size // 2]

        if initial_transformations is not None:
            initial_transformation = initial_transformations[i]
        else:
            initial_transformation = None

        transformation, result = reg_func(anc_coords=coords1[:, 1:], anc_feats=feat1,
                                          pos_coords=coords2[:, 1:], pos_feats=feat2,
                                          initial_transformation=initial_transformation)

        transformations.append(transformation)
        results.append(result)
    return torch.stack(transformations), results


def batch_coord_registration(*,
                             reg_func: Callable,
                             batch_coords: torch.Tensor,
                             batch_size: int,
                             initial_transformations: Optional[torch.Tensor] = None,
                             ) -> BatchRegResult:
    transformations = []
    results = []
    for i in range(batch_size // 2):
        coords1 = batch_coords[i]
        coords2 = batch_coords[i + batch_size // 2]

        if initial_transformations is not None:
            initial_transformation = initial_transformations[i]
        else:
            initial_transformation = None

        transformation, result = reg_func(anc_coords=coords1[:, 1:], pos_coords=coords2[:, 1:],
                                          initial_transformation=initial_transformation)

        transformations.append(transformation)
        results.append(result)
    return torch.stack(transformations), results


def batch_ransac_registration(*,
                              batch_coords: torch.Tensor,
                              batch_feats: torch.Tensor,
                              batch_size: int,
                              initial_transformations: Optional[torch.Tensor] = None,
                              ) -> BatchRegResult:
    return batch_coord_feat_registration(
        reg_func=ransac_registration,
        batch_coords=batch_coords, batch_feats=batch_feats,
        batch_size=batch_size,
        initial_transformations=initial_transformations,
    )


def batch_icp_registration(*,
                           batch_coords: torch.Tensor,
                           batch_size: int,
                           initial_transformations: Optional[torch.Tensor] = None,
                           ) -> BatchRegResult:
    return batch_coord_registration(
        reg_func=icp_registration, batch_coords=batch_coords,
        batch_size=batch_size,
        initial_transformations=initial_transformations
    )


def get_ransac_features(
        batch_dict: dict,
        model: torch.nn.Module,
        use_qk: bool = True
) -> torch.Tensor:
    """ Hacky way of extracting the Q and K from transformer based models."""

    features = batch_dict['point_features'].squeeze(-1)

    if not(model.head in ["TFHead"] and use_qk):
        return features

    batch_size = batch_dict["transformation"].shape[0]

    features = features.permute(2, 0, 1)
    features = F.normalize(features, dim=2)

    mha = model.pose_head.mod.matcher.matcher.tf.layers[0].self_attn
    wq = mha.q_proj_weight
    wk = mha.k_proj_weight
    b = mha.in_proj_bias

    e = mha.embed_dim
    h = mha.head_dim

    bq = b[:e]
    bk = b[e:2*e]

    s = h ** -0.5

    q = features[:, :batch_size, :]
    q = s * F.linear(q, weight=wq, bias=bq)
    q = q.permute(1, 2, 0)
    k = features[:, batch_size:2*batch_size, :]
    k = F.linear(k, weight=wk, bias=bk)
    k = k.permute(1, 2, 0)

    features = torch.cat([q, k], dim=0)

    return features

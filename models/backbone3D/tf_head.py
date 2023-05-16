from torch import nn
import torch.nn.functional as F

from .matchers import Matcher
from .registration import SVDRegistration
from .match_weighter import MatchWeighter
from .utils import split_apn_data


class TFMatchingRegistration(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()

		self.matcher = Matcher(**kwargs)

		self.match_weighter = MatchWeighter(weighting_method=kwargs.get("point_weighting_method"),
											order=kwargs.get("point_weighting_method_order", 2),
											normalize=kwargs.get("normalize_point_weights", False))
		self.register = SVDRegistration(**kwargs)

	def forward(self, *, src_features, src_coords, tgt_features, tgt_coords):
		"""

		:param src_features: Shape (P, BxT, F)
		:param src_coords: Shape (BxT, P, 3)
		:param tgt_features: Shape (P, BxT, F)
		:param tgt_coords: Shape (BxT, P, 3)
		:return:
		"""

		tgt_coords_proj, matching = self.matcher(src_features=src_features,
											 tgt_features=tgt_features, tgt_coords=tgt_coords.permute(1, 0, 2))
		tgt_coords_weights = self.match_weighter(matching)
		tgt_coords_proj = tgt_coords_proj.permute(1, 0, 2)  # Shape: (B x T, P, 3)
		transform = self.register(src_coords=src_coords, tgt_coords=tgt_coords_proj, weights=tgt_coords_weights)

		return transform, tgt_coords_proj, matching, tgt_coords_weights


class TFHead(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()

		self.mod = TFMatchingRegistration(**kwargs)

		self.compute_inverse_tf = kwargs.get("panoptic_weight", -1.) > 0. or \
								  kwargs.get("inv_tf_weight", -1.) > 0.

	def forward(self, batch_dict, *,
				mode="pairs", **_):

		features = batch_dict['point_features_NV'].squeeze(-1)
		d_bt, d_f, d_p = features.shape  # Shapes: Batch*Tuple, Features, Points
		coords = batch_dict['point_coords']
		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]  # Shape: (B x T, P, 3)

		features = features.permute(2, 0, 1)  # Shape: (P, B x T, F)

		# Critical!: Normalize Features so that all have the same magnitude.
		features = F.normalize(features, dim=2)

		# Split into anchor and positive features/coordinates
		features_anchor, features_positive = split_apn_data(features, mode=mode, slice_dim=1, get_negatives=False)
		coords_anchor, coords_positive = split_apn_data(coords, mode=mode, slice_dim=0, get_negatives=False)

		tf, tgt_coords_proj, matching, weights = self.mod(src_features=features_anchor, src_coords=coords_anchor,
														  tgt_features=features_positive, tgt_coords=coords_positive)

		batch_dict['transformation'] = tf
		batch_dict['transport'] = matching
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None
		batch_dict["sinkhorn_matches"] = tgt_coords_proj
		batch_dict["conf_weights"] = weights

		if self.compute_inverse_tf:
			tf2, tgt_coords_proj2, matching2, weights2 = self.mod(src_features=features_positive, src_coords=coords_positive,
																  tgt_features=features_anchor, tgt_coords=coords_anchor)

			batch_dict['transformation2'] = tf2
			batch_dict['transport2'] = matching2
			batch_dict["sinkhorn_matches2"] = tgt_coords_proj2
			batch_dict["conf_weights2"] = weights2

		return batch_dict

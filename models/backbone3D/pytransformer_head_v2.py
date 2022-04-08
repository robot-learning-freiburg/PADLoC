from torch import nn

from .matchers import TFEncMatcher
from .registration import SVDRegistration
from .attention_aggregator import AttentionAggregator
from .match_weighter import MatchWeighter
from .utils import split_apn_data


class EncTFMatchingRegistration(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()

		# kw_args = locals().copy()
		# kw_args.pop('self')  # Get rid of self and other *args
		# kw_args.pop('__class__')
		# kw_args.pop("kwargs")
		# kw_args.update(kwargs)

		self.matcher = TFEncMatcher(**kwargs)
		self.attn_agg = AttentionAggregator(agg_method=kwargs.get("attn_agg_method"))
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

		tgt_coords_proj, attn = self.matcher(src_features=src_features,
											 tgt_features=tgt_features, tgt_coords=tgt_coords.permute(1, 0, 2))
		agg_attn = self.attn_agg(attn)
		tgt_coords_weights = self.match_weighter(agg_attn)
		tgt_coords_proj = tgt_coords_proj.permute(1, 0, 2)  # Shape: (B x T, P, 3)
		transform = self.register(src_coords=src_coords, tgt_coords=tgt_coords_proj, weights=tgt_coords_weights)

		return transform, tgt_coords_proj, agg_attn, tgt_coords_weights


class PyTransformerHead2(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()

		self.mod = EncTFMatchingRegistration(**kwargs)

		self.compute_inverse_tf = kwargs.get("panoptic_loss", 0) > 0 or \
								  kwargs.get("inv_tf_weight", 0) > 0 or \
								  kwargs.get("semantic_weight", 0) > 0 or \
								  kwargs.get("supersem_weight", 0) > 0

	def forward(self, batch_dict, *,
				mode="pairs", **_):

		features = batch_dict['point_features_NV'].squeeze(-1)
		d_bt, d_f, d_p = features.shape  # Shapes: Batch*Tuple, Features, Points
		coords = batch_dict['point_coords']
		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]  # Shape: (B x T, P, 3)

		features = features.permute(2, 0, 1)  # Shape: (P, B x T, F)

		# Normalize Features for some reason. Probably not required, since the KQV weights will mess them up anyways
		# features = F.normalize(features, dim=2)

		# Split into anchor and positive features/coordinates
		features_anchor, features_positive = split_apn_data(features, mode=mode, slice_dim=1, get_negatives=False)
		coords_anchor, coords_positive = split_apn_data(coords, mode=mode, slice_dim=0, get_negatives=False)

		tf, tgt_coords_proj, matching, weights = self.mod(src_features=features_anchor, src_coords=coords_anchor,
														  tgt_features=features_positive, tgt_coords=coords_positive)

		batch_dict['transformation'] = tf
		batch_dict['transport'] = matching.permute(1, 0, 2)
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None
		batch_dict["sinkhorn_matches"] = tgt_coords_proj.permute(1, 0, 2)

		if self.compute_inverse_tf:
			tf2, tgt_coords_proj2, matching2, weights2 = self.mod(src_feat=features_positive, src_coord=coords_positive,
																  tgt_feat=features_anchor, tgt_coord=coords_anchor)

			batch_dict['transformation_2'] = tf2
			batch_dict['transport'] = matching2.permute(1, 0, 2)
			batch_dict["sinkhorn_matches_2"] = tgt_coords_proj2.permute(1, 0, 2)

		return batch_dict

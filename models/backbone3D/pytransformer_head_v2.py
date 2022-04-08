import torch
from torch import nn
import torch.nn.functional as F


# class PyTransformerHead2(nn.Module):
# 	"""
# 	TODO
# 	"""
#
# 	def __init__(self, *, feature_size,
# 				 tf_xa_enc_nheads=1, tf_xa_enc_layers=1, tf_xa_hiddn_size=None,
# 				 point_weighting_method="weight_sum", point_weighting_method_order=2,
# 				 inv_tf_weight=0,
# 				 **_):
#
# 		super(PyTransformerHead2, self).__init__()
#
# 		# Cross Attention Layer Parameters
# 		xa_enc_layers = tf_xa_enc_layers or 1
# 		# Default hidden size as done in the original paper "Attention is all you need": 4 times d_model
# 		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size
#
# 		sa_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
# 												 kdim=feature_size, vdim=3,
# 												 dim_feedforward=xa_hiddn_size)
#
# 		weighting_methods = {
# 			"hill": {"f": norm_hill_number, "kwargs": {"dim": -1}},
# 			"berger": {"f": berger_parker_index, "kwargs": {"dim": -1}},
# 			"weight_sum": {"f": weight_sum, "kwargs": {"dim": 1}}
# 		}
#
# 		if point_weighting_method not in weighting_methods:
# 			raise ValueError(f"Invalid point weighting method ({point_weighting_method}). Valid values: {weighting_methods.keys()}.")
# 		weighting_method = weighting_methods[point_weighting_method]
# 		weighting_method_kwargs = weighting_method["kwargs"].copy()
# 		weighting_method_kwargs["q"] = point_weighting_method_order
# 		self._weighting_method = weighting_method["f"]
# 		self._weighting_method_kwargs = weighting_method_kwargs
#
# 		self._inv_tf_weight = inv_tf_weight
#
# 		self.xa_encoder = XATransformerEncoder(sa_enc_layer, num_layers=xa_enc_layers)
# 		self.linear = nn.Linear(feature_size, 3)
#
# 	def forward(self, batch_dict, *,
# 				mode="pairs", **_):
# 		"""
# 		TODO
# 		:param batch_dict:
# 		:param mode:
# 		:return:
# 		"""
#
# 		features = batch_dict['point_features_NV'].squeeze(-1)
# 		coords = batch_dict['point_coords']
#
# 		# Dimensions
# 		d_bt, d_f, d_p = features.shape  # Dimensions: Batch*Tuple, Features, Points
# 		d_t = 2 if mode == "pairs" else 3  # Dimension: Tuple size (2 <- pairs / 3 <- triplets)
# 		d_b = d_bt // d_t  # Dimension: Batch size
#
# 		assert d_b * d_t == d_bt
#
# 		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]
#
# 		features = features.permute(2, 0, 1)
# 		coords = coords.permute(1, 0, 2)
#
# 		# Normalize Features for some reason. Probably not required, since the KQV weights will mess them up anyways
# 		features_norm = F.normalize(features, dim=2)
#
# 		# Split into anchor and positive features/coordinates
# 		features1 = features_norm[:, :d_b, :]
# 		features2 = features_norm[:, d_b:2*d_b, :]
# 		coords1 = coords[:, :d_b, :]  # Coordinates of PC1
# 		coords2 = coords[:, d_b:2*d_b, :]  # Coordinates of PC2
#
# 		in_features1, in_features2 = features1, features2
# 		in_coords1, in_coords2 = coords1, coords2
# 		if self._inv_tf_weight:
# 			in_features1 = torch.cat([features1, features2])
# 			in_features2 = torch.cat([features2, features1])
# 			in_coords1 = torch.cat([coords1, coords2])
# 			in_coords2 = torch.cat([coords2, coords1])
#
# 		tf_out = self.xa_encoder(q=in_features1, k=in_features2, v=in_coords2)
# 		attn_matrix = self.xa_encoder.attention[-1]
#
# 		# Return the output of the transformer encoder back to coordinate size (3) to get the virtual points.
# 		matches = self.linear(tf_out)
#
# 		# Compute the weight of each virtual point
# 		wm_kwargs = self._weighting_method_kwargs.copy()
# 		wm_kwargs["p"] = attn_matrix
# 		svd_weights = self._weighting_method(**wm_kwargs)
#
# 		src_coords = in_coords1.permute(1, 0, 2)
# 		tgt_coords = matches.permute(1, 0, 2)
#
# 		batch_dict['transport'] = attn_matrix
# 		batch_dict['sinkhorn_matches'] = tgt_coords
#
# 		try:
# 			transformation = compute_rigid_transform(src_coords, tgt_coords, svd_weights)
# 		except RuntimeError as e:
# 			print("SVD did not converge!!!!!")
# 			print(e)
# 			print("Debug Info:")
# 			print("\n\n\nattn: ", attn_matrix)
# 			print("\n\n\nsrc_coords: ", src_coords)
# 			print("\n\n\ntgt_coords:   ", tgt_coords)
# 			print("\n\n\nsvd_weights:   ", svd_weights)
# 			print("\n\n\n")
#
# 			raise SVDNonConvergenceError("SVD did not converge!")
#
# 		batch_dict['transformation'] = transformation
# 		batch_dict['out_rotation'] = None
# 		batch_dict['out_translation'] = None
#
# 		return batch_dict

from modules.matchers import EncTFMatcher
from modules.registrators import SVDRegistrator
from modules.attention_aggregator import AttentionAggregator
from modules.match_weighter import MatchWeighter
from modules.utils import split_apn_data


class EncTFMatchingRegistration(nn.Module):

	def __init__(self, **kwargs):
		super(EncTFMatchingRegistration, self).__init__()

		kw_args = locals().copy()
		kw_args.pop('self')  # Get rid of self and other *args
		kw_args.update(kwargs)

		self.matcher = EncTFMatcher(**kwargs)
		self.attn_agg = AttentionAggregator(agg_method=kwargs.get("attn_agg_method"))
		self.match_weighter = MatchWeighter(weighting_method=kwargs.get("point_weighting_method"),
											order=kwargs.get("point_weighting_method_order", 2),
											normalize=kwargs.get("normalize_point_weights", False))
		self.registrator = SVDRegistrator(**kwargs)

	def forward(self, *, src_features, src_coords, tgt_features, tgt_coords):

		tgt_coords_proj, attn = self.matcher(src_features=src_features, tgt_features=tgt_features, tgt_coords=tgt_coords)
		agg_attn = self.attn_agg.forward(attn)
		tgt_coords_weights = self.match_weighter(agg_attn)
		transform = self.registrator(src_coords=src_coords, tgt_coords=tgt_coords_proj, weights=tgt_coords_weights)

		return transform, tgt_coords_proj, agg_attn, tgt_coords_weights


class PyTransformerHead2(nn.Module):

	def __init__(self, **kwargs):
		super(PyTransformerHead2, self).__init__()

		self.mod = EncTFMatchingRegistration(**kwargs)

		self.compute_inverse_tf = kwargs.get("panoptic_loss", 0) > 0 or \
								  kwargs.get("inv_tf_weight", 0) > 0 or \
								  kwargs.get("semantic_weight", 0) > 0 or \
								  kwargs.get("supersem_weight", 0) > 0

	def forward(self, batch_dict, *,
				mode="pairs", **_):

		features = batch_dict['point_features_NV'].squeeze(-1)
		d_bt, d_f, d_p = features.shape  # Dimensions: Batch*Tuple, Features, Points
		coords = batch_dict['point_coords']
		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]

		features = features.permute(2, 0, 1)
		coords = coords.permute(1, 0, 2)

		# Normalize Features for some reason. Probably not required, since the KQV weights will mess them up anyways
		# features = F.normalize(features, dim=2)

		# Split into anchor and positive features/coordinates
		features_anchor , features_positive = split_apn_data(features, mode=mode, slice_dim=1, get_negatives=False)
		coords_anchor, coords_positive = split_apn_data(coords, mode=mode, slice_dim=1, get_negatives=False)

		tf, tgt_coords_proj, matching, weights = self.mod(src_features=features_anchor, src_coords=coords_anchor,
														  tgt_features=features_positive, tgt_coords=coords_positive)

		batch_dict['transformation'] = tf
		batch_dict['transport'] = matching.permute(1, 0, 2)
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None
		batch_dict["sinkhorn_matches"] = tgt_coords_proj.permute(1, 0, 2)

		if self.inverse_tf:
			tf2, tgt_coords_proj2, matching2, weights2 = self.mod(src_feat=features_positive, src_coord=coords_positive,
																  tgt_feat=features_anchor, tgt_coord=coords_anchor)

			batch_dict['transformation_2'] = tf2
			batch_dict['transport'] = matching2.permute(1, 0, 2)
			batch_dict["sinkhorn_matches_2"] = tgt_coords_proj2.permute(1, 0, 2)

		return batch_dict
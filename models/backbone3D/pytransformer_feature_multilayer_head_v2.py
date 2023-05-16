import torch
from torch import nn
import torch.nn.functional as F

from .heads import compute_rigid_transform
from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer
from .match_weighter import norm_hill_number, berger_parker_index, weight_sum
from utils.tools import SVDNonConvergenceError


class PyTransformerFeatureMultiLayerHead(nn.Module):
	"""
	TODO
	"""

	def __init__(self, *, feature_size,
				 tf_xa_enc_nheads=1, tf_xa_enc_layers=1, tf_xa_hiddn_size=None,
				 point_weighting_method="weight_sum", point_weighting_method_order=2,
				 attn_matrix_method="last",
				 inv_tf_weight=0,
				 **_):

		super().__init__()

		# Cross Attention Layer Parameters
		xa_enc_layers = tf_xa_enc_layers or 1
		# Default hidden size as done in the original paper "Attention is all you need": 4 times d_model
		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size

		xa_feature_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
												 kdim=feature_size, vdim=feature_size,
												 dim_feedforward=xa_hiddn_size)
		xa_point_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
													   kdim=feature_size, vdim=3,
													   dim_feedforward=xa_hiddn_size)

		weighting_methods = {
			"hill": {"f": norm_hill_number, "kwargs": {"dim": -1}},
			"berger": {"f": berger_parker_index, "kwargs": {"dim": -1}},
			"weight_sum": {"f": weight_sum, "kwargs": {"dim": 1}}
		}

		attn_matrix_methods = ["last", "sum", "product", "hadamard"]

		if attn_matrix_method not in attn_matrix_methods:
			raise ValueError(f"Invalid attention matrix aggregation method ({attn_matrix_method})." +
					f"Valid values: {attn_matrix_methods}.")

		self.attn_matrix_method = attn_matrix_method

		if point_weighting_method not in weighting_methods:
			raise ValueError(f"Invalid point weighting method ({point_weighting_method})." +
					f"Valid values: {weighting_methods.keys()}.")
		weighting_method = weighting_methods[point_weighting_method]
		weighting_method_kwargs = weighting_method["kwargs"].copy()
		weighting_method_kwargs["q"] = point_weighting_method_order
		self._weighting_method = weighting_method["f"]
		self._weighting_method_kwargs = weighting_method_kwargs

		self._inv_tf_weight = inv_tf_weight

		self.xa_feature_encoder = XATransformerEncoder(xa_feature_enc_layer, num_layers=xa_enc_layers)
		self.xa_point_encoder = XATransformerEncoder(xa_point_enc_layer, num_layers=1)
		self.linear = nn.Linear(feature_size, 3)

	def forward(self, batch_dict, *,
				mode="pairs", **_):
		"""
		TODO
		:param batch_dict:
		:param mode:
		:return:
		"""

		features = batch_dict['point_features_NV'].squeeze(-1)
		coords = batch_dict['point_coords']

		# Dimensions
		d_bt, d_f, d_p = features.shape  # Dimensions: Batch*Tuple, Features, Points
		d_t = 2 if mode == "pairs" else 3  # Dimension: Tuple size (2 <- pairs / 3 <- triplets)
		d_b = d_bt // d_t  # Dimension: Batch size

		assert d_b * d_t == d_bt

		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]

		features = features.permute(2, 0, 1)
		coords = coords.permute(1, 0, 2)

		# Normalize Features for some reason. Probably not required, since the KQV weights will mess them up anyways
		features_norm = F.normalize(features, dim=2)

		# Split into anchor and positive features/coordinates
		features1 = features_norm[:, :d_b, :]
		features2 = features_norm[:, d_b:2*d_b, :]
		coords1 = coords[:, :d_b, :]  # Coordinates of PC1
		coords2 = coords[:, d_b:2*d_b, :]  # Coordinates of PC2

		in_features1, in_features2 = features1, features2
		in_coords1, in_coords2 = coords1, coords2
		if self._inv_tf_weight:
			in_features1 = torch.cat([features1, features2])
			in_features2 = torch.cat([features2, features1])
			in_coords1 = torch.cat([coords1, coords2])
			in_coords2 = torch.cat([coords2, coords1])

		xa_features1 = self.xa_feature_encoder(q=in_features1, k=in_features2, v=in_features2)

		xaf1_attn = None
		for a in self.xa_feature_encoder.attention:
			# Compute the attention matrix as
			# 		A = A_l * A_l-1 * ... * A_2 * A_1 * A_0
			# So that the point coordinates passed to the second, single-layer encoder are:
			# 		^P_2 = A * P_2
			#            = A_l * ( A_l-1 * ( ... * A_2 * ( A_1 * ( A_0 * P_2 ) ) ... ) )
			if xaf1_attn is None:
				xaf1_attn = a
			else:
				xaf1_attn = torch.bmm(a, xaf1_attn)

		in_coords2_hat = torch.bmm(xaf1_attn, in_coords2.permute(1, 0, 2)).permute(1, 0, 2)

		xa_hd_points = self.xa_point_encoder(q=xa_features1, k=in_features2, v=in_coords2_hat)

		# Return the output of the transformer encoder back to coordinate size (3) to get the virtual points.
		matches = self.linear(xa_hd_points)

		# Compute the attention matrix
		last_attn_matrix = self.xa_point_encoder.attention[-1]
		if self.attn_matrix_method == "sum":
			attn_matrix = last_attn_matrix
			for a in self.xa_feature_encoder.attention:
				attn_matrix = attn_matrix + a
		elif self.attn_matrix_method == "product":
			attn_matrix = torch.bmm(last_attn_matrix, xaf1_attn)
		elif self.attn_matrix_method == "hadamard":
			attn_matrix = last_attn_matrix
			for a in self.xa_feature_encoder.attention:
				attn_matrix = attn_matrix * a
		else:
			attn_matrix = last_attn_matrix

		# Re-normalize the attention matrix after summing or multiplying,
		# since the weights are no longer going to add up to one
		if self.attn_matrix_method in ["sum", "hadamard"]:
			attn_matrix = torch.softmax(attn_matrix, dim=-1)

		# Compute the weight of each virtual point
		wm_kwargs = self._weighting_method_kwargs.copy()
		wm_kwargs["p"] = attn_matrix
		svd_weights = self._weighting_method(**wm_kwargs)

		src_coords = in_coords1.permute(1, 0, 2)
		tgt_coords = matches.permute(1, 0, 2)

		batch_dict['transport'] = attn_matrix
		batch_dict['sinkhorn_matches'] = tgt_coords

		try:
			transformation = compute_rigid_transform(src_coords, tgt_coords, svd_weights)
		except RuntimeError as e:
			print("SVD did not converge!!!!!")
			print(e)
			print("Debug Info:")
			print("\n\n\nattn: ", attn_matrix)
			print("\n\n\nsrc_coords: ", src_coords)
			print("\n\n\ntgt_coords:   ", tgt_coords)
			print("\n\n\nsvd_weights:   ", svd_weights)
			print("\n\n\n")

			raise SVDNonConvergenceError("SVD did not converge!")

		batch_dict['transformation'] = transformation
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		return batch_dict

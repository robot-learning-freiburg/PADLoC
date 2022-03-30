import torch
from torch import nn
import torch.nn.functional as F

from .heads import compute_rigid_transform
from torch.nn.modules.transformer import Transformer
from .xatransformer import XATransformerEncoderLayer, XATransformerEncoder
from .pytransformer_head_v2 import norm_hill_number, berger_parker_index, weight_sum
from utils.tools import SVDNonConvergenceError


class DeepClosestPointHead(nn.Module):
	"""
	Based on

	Deep Closest Point: Learning Representations for Point Cloud Registration
	https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf
	"""

	def __init__(self, *, feature_size,
				 tf_enc_layers=2,
				 tf_dec_layers=2,
				 tf_nheads=1,
				 tf_hiddn_size=None,
				 tf_coord_enc_nheads=1,
				 tf_coord_enc_hiddn_size=None,
				 point_weighting_method="weight_sum", point_weighting_method_order=2,
				 inv_tf_weight=0,
				 phi_weight=0.1,
				 dropout=0.1,
				 **_):

		super(DeepClosestPointHead, self).__init__()

		# Feature Transformer phi() Parameters
		tf_enc_layers = tf_enc_layers or 1
		tf_dec_layers = tf_dec_layers or 1
		tf_nheads = tf_nheads or 1
		# Default hidden size as done in the original paper "Attention is all you need": 4 times d_model
		tf_hiddn_size = (feature_size * 4) if tf_hiddn_size is None else tf_hiddn_size

		self.phi_tf = Transformer(d_model=feature_size, nhead=tf_nheads,
							   num_encoder_layers=tf_enc_layers, num_decoder_layers=tf_dec_layers,
							   dim_feedforward=tf_hiddn_size)
		self.phi_weight = phi_weight

		self.dropout1 = nn.Dropout(dropout)

		# Coordinate Transformer Encoder (Pointer) Parameters
		tf_coord_enc_nheads = tf_coord_enc_nheads or 1
		tf_coord_enc_hiddn_size = (feature_size * 4) if tf_coord_enc_hiddn_size is None else tf_coord_enc_hiddn_size
		enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_coord_enc_nheads,
											  kdim=feature_size, vdim=3, dim_feedforward=tf_coord_enc_hiddn_size)
		self.coord_enc = XATransformerEncoder(enc_layer, 1)

		weighting_methods = {
			"hill": {"f": norm_hill_number, "kwargs": {"dim": -1}},
			"berger": {"f": berger_parker_index, "kwargs": {"dim": -1}},
			"weight_sum": {"f": weight_sum, "kwargs": {"dim": 1}}
		}

		if point_weighting_method not in weighting_methods:
			raise ValueError(f"Invalid point weighting method ({point_weighting_method}). "
							 f"Valid values: {weighting_methods.keys()}.")
		weighting_method = weighting_methods[point_weighting_method]
		weighting_method_kwargs = weighting_method["kwargs"].copy()
		weighting_method_kwargs["q"] = point_weighting_method_order
		self._weighting_method = weighting_method["f"]
		self._weighting_method_kwargs = weighting_method_kwargs

		self._inv_tf_weight = inv_tf_weight

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

		tgt = torch.cat([features2, features1], dim=1)

		phi = self.phi_tf(src=features_norm, tgt=tgt)
		phi = F.normalize(phi, dim=2)
		phi = self.phi_weight * phi

		phi = features_norm + self.dropout1(phi)

		phi_1 = phi[:, :d_b, :]
		phi_2 = phi[:, d_b:2*d_b, :]

		tf_out = self.coord_enc(q=phi_1, k=phi_2, v=coords2)
		attn_matrix = self.coord_enc.attention[-1]

		# Return the output of the transformer encoder back to coordinate size (3) to get the virtual points.
		matches = self.linear(tf_out)

		# Compute the weight of each virtual point
		wm_kwargs = self._weighting_method_kwargs.copy()
		wm_kwargs["p"] = attn_matrix
		svd_weights = self._weighting_method(**wm_kwargs)

		src_coords = coords1.permute(1, 0, 2)
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

			del batch_dict
			raise SVDNonConvergenceError("SVD did not converge!")

		batch_dict['transformation'] = transformation
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		return batch_dict

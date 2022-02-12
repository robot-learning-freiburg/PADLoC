from torch import nn
import torch.nn.functional as F

from .heads import compute_rigid_transform
from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer
from .positional_encoder import PositionalEncodingCart3D
from utils.tools import SVDNonConvergenceError


class PyTransformerHead2(nn.Module):
	"""
	TODO
	"""

	def __init__(self, **kwargs):

		super(PyTransformerHead2, self).__init__()

		feat_size = kwargs['feature_size']

		# Cross Attention Layer Parameters
		xa_enc_nheads = kwargs.get("tf_xa_enc_nheads") or 1
		xa_enc_layers = kwargs.get("tf_xa_enc_layers") or 1
		# Default hidden size as done in the original paper "Attention is all you need": 4 times d_model
		xa_hiddn_size = kwargs.get("tf_xa_hiddn_size") or (feat_size * 4)

		sa_enc_layer = XATransformerEncoderLayer(d_model=feat_size, nhead=xa_enc_nheads,
												 kdim=feat_size, vdim=3,
												 dim_feedforward=xa_hiddn_size)

		self.xa_encoder = XATransformerEncoder(sa_enc_layer, num_layers=xa_enc_layers)
		self.linear = nn.Linear(feat_size, 3)

	def forward(self, batch_dict, **kwargs):
		"""
		TODO
		:param batch_dict:
		:param kwargs:
		:return:
		"""

		features = batch_dict['point_features_NV'].squeeze(-1)
		coords = batch_dict['point_coords']

		# Dimensions
		d_bt, d_f, d_p = features.shape  # Dimensions: Batch*Tuple, Features, Points
		mode = kwargs.get("mode") or "pairs"
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

		tf_out = self.xa_encoder(k=features1, q=features2, v=coords2)
		attn_matrix = self.xa_encoder.attention[-1]

		# Return the output of the transformer encoder back to coordinate size (3)
		matches = self.linear(tf_out)

		svd_weights = attn_matrix.sum(1)

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

			raise SVDNonConvergenceError("SVD did not converge!")

		batch_dict['transformation'] = transformation
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		return batch_dict

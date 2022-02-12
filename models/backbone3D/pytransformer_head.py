from torch import nn
import torch.nn.functional as F

from .heads import compute_rigid_transform
from .xatransformer import SATransformerEncoder, XATransformerDecoder,\
	SATransformerEncoderLayer, XATransformerDecoderLayer
from .positional_encoder import PositionalEncodingCart3D
from utils.tools import SVDNonConvergenceError


class PyTransformerHead(nn.Module):
	"""
	TODO
	"""

	def __init__(self, **kwargs):

		super(PyTransformerHead, self).__init__()

		self._pe_weight = kwargs.get("position_encoder_weight", 1)
		feat_size = kwargs['feature_size']

		self.descriptor_head = kwargs['desc_head']

		# Self Attention Layer Parameters
		sa_enc_nheads = kwargs.get("tf_sa_enc_nheads") or 1
		sa_enc_layers = kwargs.get("tf_sa_enc_layers") or 1
		# sa_dec_nheads = kwargs.get("tf_sa_dec_nheads") or 3
		# sa_dec_layers = kwargs.get("tf_sa_dec_layers") or 3
		sa_hiddn_size = kwargs.get("tf_sa_hiddn_size") or 1024

		# Cross Attention Layer Parameters
		# xa_enc_nheads = kwargs.get("tf_xa_enc_nheads") or 3
		# xa_enc_layers = kwargs.get("tf_xa_enc_layers") or 3
		xa_dec_sa_nheads = kwargs.get("tf_xa_dec_sa_nheads") or 1
		xa_dec_mha_nheads = kwargs.get("tf_xa_dec_mha_nheads") or 1
		xa_dec_layers = kwargs.get("tf_xa_dec_layers") or 1
		xa_hiddn_size = kwargs.get("tf_xa_hiddn_size") or 1024

		self._positional_encoding = None
		if self._pe_weight:
			self._positional_encoding = PositionalEncodingCart3D(feat_size, **kwargs)

		sa_enc_layer = SATransformerEncoderLayer(d_model=feat_size, nhead=sa_enc_nheads,
												  dim_feedforward=sa_hiddn_size)
		xa_dec_layer = XATransformerDecoderLayer(sa_d_model=3, mha_d_model=feat_size,
												 sa_nhead=xa_dec_sa_nheads, mha_nhead=xa_dec_mha_nheads,
												 dim_feedforward=xa_hiddn_size,
												 mha_kdim=feat_size, mha_vdim=3)

		self.sa_encoder = SATransformerEncoder(sa_enc_layer, num_layers=sa_enc_layers)
		self.xa_decoder = XATransformerDecoder(xa_dec_layer, num_layers=xa_dec_layers)

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

		features = features.permute(2, 0, 1)
		src = features
		coords = coords.view(d_bt, d_p, 4)[:, :, 1:]

		if self._pe_weight:
			src = F.normalize(src, dim=2)
			pe = self._positional_encoding(coords)
			if self._pe_weight != 1.0:
				pe = self._pe_weight * pe
			src = src + pe.permute(1, 0, 2)

		coords = coords.permute(1, 0, 2)
		coords1 = coords[:, :d_b, :]  # Coordinates of PC1
		coords2 = coords[:, d_b:2*d_b, :]  # Coordinates of PC2

		attn_feats = self.sa_encoder(src=src)

		src1 = attn_feats[:, :d_b, :]  # Attention Features of PC1
		src2 = attn_feats[:, d_b:2*d_b, :]  # Attention Features of PC2

		matches = self.xa_decoder(tgt_k=coords2, tgt_q=coords1, tgt_v=coords2,
								  src_k=src2, src_q=src1)

		attn_matrix = self.xa_decoder.attention_cross[-1]
		svd_weights = attn_matrix.sum(1)

		src_coords = coords1.permute(1, 0, 2)
		tgt_coords = matches.permute(1, 0, 2)

		batch_dict['transport'] = attn_matrix
		batch_dict['attention_features'] = attn_feats.permute(1, 0, 2)
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

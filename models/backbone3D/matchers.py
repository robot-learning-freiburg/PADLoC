import torch
import torch.nn as nn

from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer
from .attention_aggregator import AttentionAggregator


def normalize_matching(matching):
	matching_rows = matching.sum(2, keepdim=True)
	matching_rows[matching_rows < 1e-6] = 1  # Ignore rows that add up to 0 (Set to one to avoid division by 0)
	return matching / matching_rows


def sinkhorn_unbalanced(*, src_features, tgt_features, epsilon, gamma, max_iter):
	"""
	Sinkhorn algorithm
	Parameters
	----------
	src_features : torch.Tensor
		Feature for points cloud 1. Used to computed transport cost.
		Size B x N x C.
	tgt_features : torch.Tensor
		Feature for points cloud 2. Used to computed transport cost.
		Size B x M x C.
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

	# Transport cost matrix
	src_features = src_features / torch.sqrt(torch.sum(src_features ** 2, -1, keepdim=True) + 1e-8)
	tgt_features = tgt_features / torch.sqrt(torch.sum(tgt_features ** 2, -1, keepdim=True) + 1e-8)
	C = 1.0 - torch.bmm(src_features, tgt_features.transpose(1, 2))

	# Entropic regularisation
	k = torch.exp(-C / epsilon)

	# Early return if no iteration
	if max_iter == 0:
		return k

	# Init. of Sinkhorn algorithm
	power = gamma / (gamma + epsilon)
	a = (
			torch.ones(
				(k.shape[0], k.shape[1], 1), device=src_features.device, dtype=src_features.dtype
			)
			/ k.shape[1]
	)
	prob1 = (
			torch.ones(
				(k.shape[0], k.shape[1], 1), device=src_features.device, dtype=src_features.dtype
			)
			/ k.shape[1]
	)
	prob2 = (
			torch.ones(
				(k.shape[0], k.shape[2], 1), device=tgt_features.device, dtype=tgt_features.dtype
			)
			/ k.shape[2]
	)

	# Sinkhorn algorithm
	for _ in range(max_iter):
		# Update b
		kT_a = torch.bmm(k.transpose(1, 2), a)
		b = torch.pow(prob2 / (kT_a + 1e-8), power)
		# Update a
		k_b = torch.bmm(k, b)
		a = torch.pow(prob1 / (k_b + 1e-8), power)

	# Transportation map
	t = torch.mul(torch.mul(a, k), b.transpose(1, 2))

	return t


class TFEncMatcher(nn.Module):

	def __init__(self,  *, feature_size,
				 tf_xa_enc_nheads=4, tf_xa_enc_layers=1,
				 tf_xa_hiddn_size=None,
				 tf_skip_conn1=False,
				 tf_skip_conn2=True,
				 attn_agg_method=None,
				 dropout=0.1,
				 **_):

		super().__init__()

		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size

		enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
											  kdim=feature_size, vdim=3,
											  dim_feedforward=xa_hiddn_size, dropout=dropout,
											  skip_conn1=tf_skip_conn1, skip_conn2=tf_skip_conn2)
		self.tf = XATransformerEncoder(enc_layer, num_layers=tf_xa_enc_layers)

		self.linear = nn.Linear(feature_size, 3)

		self.attn_agg = AttentionAggregator(agg_method=attn_agg_method)

	def forward(self, *, src_features, tgt_features, tgt_coords):
		x = self.tf(q=src_features, k=tgt_features, v=tgt_coords)
		x = self.linear(x)

		matching = self.attn_agg(self.tf.attention)

		return x, matching


class TFMLEncMatcher(nn.Module):

	def __init__(self,  *, feature_size,
				 tf_xa_enc_nheads=4, tf_xa_enc_layers=1,
				 tf_xa_hiddn_size=None,
				 tf_skip_conn2=True,
				 attn_agg_method=None,
				 dropout=0.1,
				 **_):

		super().__init__()

		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size

		# Encoder Layer for Multi-Layer Feature Encoder
		feat_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
												   kdim=feature_size, vdim=feature_size,
												   dim_feedforward=xa_hiddn_size, dropout=dropout,
												   skip_conn1="q", skip_conn2=tf_skip_conn2)
		feat_norm = nn.LayerNorm(feature_size)
		# Encoder Layer for Single-Layer Coordinate Encoder
		enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
											  kdim=feature_size, vdim=3,
											  dim_feedforward=xa_hiddn_size, dropout=dropout,
											  skip_conn1=None, skip_conn2=tf_skip_conn2)

		# Multi-Layer Feature Encoder
		self.feat_tf = XATransformerEncoder(feat_enc_layer, num_layers=tf_xa_enc_layers, norm=feat_norm)
		# Single-Layer Coordinate Encoder
		self.tf = XATransformerEncoder(enc_layer, num_layers=1)

		self.linear = nn.Linear(feature_size, 3)

		self.attn_agg = AttentionAggregator(agg_method=attn_agg_method)

	def forward(self, *, src_features, tgt_features, tgt_coords):
		src_features_proj = self.feat_tf(q=src_features, k=tgt_features, v=tgt_features)
		x = self.tf(q=src_features_proj, k=tgt_features, v=tgt_coords)
		x = self.linear(x)

		attn = self.feat_tf.attention + self.tf.attention

		matching = self.attn_agg(attn)

		return x, matching


class UOTMatcher(nn.Module):

	def __init__(self, *, nb_iter=5, **_):
		super().__init__()

		# Mass regularisation
		self.gamma = nn.Parameter(torch.zeros(1))
		# Entropic regularisation
		self.epsilon = nn.Parameter(torch.zeros(1))

		self.nb_iter = nb_iter

	def forward(self, *, src_features, tgt_features, tgt_coords):
		matching = sinkhorn_unbalanced(src_features=src_features.permute(1, 0, 2),
									   tgt_features=tgt_features.permute(1, 0, 2),
									   epsilon=torch.exp(self.epsilon) + 0.03, gamma=torch.exp(self.gamma),
									   max_iter=self.nb_iter)

		matching = normalize_matching(matching)
		x = torch.bmm(matching, tgt_coords.permute(1, 0, 2))

		return x.permute(1, 0, 2), matching


class Matcher(nn.Module):

	_MATCHER_DICT = {
		"TFEncMatcher": TFEncMatcher,
		"TFMLEncMatcher": TFMLEncMatcher,
		"UOTMatcher": UOTMatcher,
	}

	def __init__(self, *args, matching_head=None, normalize_matches=True, **kwargs):

		super().__init__()

		if matching_head not in self._MATCHER_DICT:
			raise KeyError(f"Invalid Matching module ({matching_head}). Valid values: [{self._MATCHER_DICT.keys()}]")

		self.matcher = self._MATCHER_DICT[matching_head](*args, **kwargs)
		self.normalize = normalize_matches

	def forward(self, *args, **kwargs):

		x, matching = self.matcher(*args, **kwargs)

		if self.normalize:
			matching = normalize_matching(matching)

		return x, matching

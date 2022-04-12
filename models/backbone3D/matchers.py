import torch.nn as nn

from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer
from .attention_aggregator import AttentionAggregator


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
		# Encoder Layer for Single-Layer Coordinate Encoder
		enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
											  kdim=feature_size, vdim=3,
											  dim_feedforward=xa_hiddn_size, dropout=dropout,
											  skip_conn1=None, skip_conn2=tf_skip_conn2)

		# Multi-Layer Feature Encoder
		self.feat_tf = XATransformerEncoder(feat_enc_layer, num_layers=tf_xa_enc_layers)
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


class Matcher(nn.Module):

	_MATCHER_DICT = {
		"TFEncMatcher": TFEncMatcher,
		"TFMLEncMatcher": TFMLEncMatcher,
	}

	def __init__(self, *args, matching_head=None, **kwargs):

		super().__init__()

		if matching_head not in self._MATCHER_DICT:
			raise KeyError(f"Invalid Matching module ({matching_head}). Valid values: [{self._MATCHER_DICT.keys()}]")

		self.matcher = self._MATCHER_DICT[matching_head](*args, **kwargs)

	def forward(self, *args, **kwargs):
		return self.matcher(*args, **kwargs)

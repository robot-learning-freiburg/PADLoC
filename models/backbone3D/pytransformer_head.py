from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, _get_activation_fn

from .heads import compute_rigid_transform


class PositionalEncodingCart3D(nn.Module):
	"""
	Positional Encoding Module for 3D Cartesian Coordinates (x, y, z).
	Inspired by:
	https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
	"""
	def __init__(self, channels, **kwargs):
		"""

		:param channels: Number of channels
		:type channels: int
		"""
		super(PositionalEncodingCart3D, self).__init__()

		self._pe_weight = kwargs.get("position_encoder_weight") or 1

		if not self._pe_weight:
			return

		self.channels = channels
		subchannels = int(np.ceil(channels / 6) * 2)
		if subchannels % 2:
			subchannels += 1

		self._subchannels = subchannels

		# Used for integer numbers
		freq = 2 * np.pi / (10000 ** (torch.arange(0, self._subchannels, 2).float() / self._subchannels))
		# For positions in [0, 1]:
		# inv_freq = np.pi * (torch.arange(0, self._subchannels, 2).float() + 1)
		self.register_buffer("freq", freq)

		point_cloud_range = kwargs.get("PC_RANGE", [-70.4, -70.4, -3, 70.4, 70.4, 1])
		min_x, min_y, min_z, max_x, max_y, max_z = point_cloud_range
		range_min = torch.Tensor([min_x, min_y, min_z])
		range_max = torch.Tensor([max_x, max_y, max_z])
		range = range_max - range_min
		self.register_buffer("range_min", range_min)
		self.register_buffer("range", range)

	def forward(self, src: torch.Tensor) -> Union[None, torch.Tensor]:

		d_b, d_p, d_c = src.shape

		if d_c != 3:
			raise ValueError(f"Only Tensors whose last dimension is 3 are supported. Shape {src.shape[-1]}.")

		if self._pe_weight == 0:
			return None

		# Normalize cartesian coordinates to [0, 1] range based on the voxel grid dimensions
		norm_pos = (src - self.range_min) / self.range
		# Mutliply each coordinate times the frequency
		sin_inp = torch.einsum("...k,l->...kl", norm_pos, self.freq)

		# Compute the sin and cos, and stack them
		emb = torch.stack((torch.sin(sin_inp), torch.cos(sin_inp)), dim=-1)
		# Interleave, so that we get sin, cos, sin, cos...
		emb = emb.view(d_b, d_p, d_c, self._subchannels)
		# Reshape so that we get (sin, cos, sin, ...) for x, then (sin, cos, ...) for y and finally for z
		emb = emb.permute(0, 1, 3, 2).reshape(d_b, d_p, 3 * self._subchannels)
		emb = emb[:, :, :self.channels]

		# Normalize to be of the same magnitude as the tensor it will be added to
		emb = nn.functional.normalize(emb, dim=2)

		if self._pe_weight != 1.0:
			emb = self._pe_weight * emb

		return emb


class XATransformerDecoder(nn.Module):
	r"""TransformerDecoder is a stack of N decoder layers

	Args:
		decoder_layer: an instance of the TransformerDecoderLayer() class (required).
		num_layers: the number of sub-decoder-layers in the decoder (required).
		norm: the layer normalization component (optional).

	Examples::
		>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
		>>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
		>>> memory = torch.rand(10, 32, 512)
		>>> tgt = torch.rand(20, 32, 512)
		>>> out = transformer_decoder(tgt, memory)
	"""
	__constants__ = ['norm']

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(XATransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, tgt_q: Tensor, tgt_k: Tensor, tgt_v: Tensor,
				src_k: Tensor, src_v: Tensor,
				tgt_mask: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None,
				src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer in turn.

		Args:
			tgt_q: the target query sequence to the decoder layer (required).
			tgt_k: the target key sequence to the decoder (required).
			tgt_v: the target value sequence to the decoder (required).
			src_k: the sequence keys from the last layer of the encoder (required).
			src_v: the sequence values from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			src_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			src_key_padding_mask: the mask for the memory keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = tgt_q

		for mod in self.layers:
			output = mod(output, tgt_k, tgt_v,
						 src_k, src_v,
						 tgt_mask=tgt_mask,
						 src_mask=src_mask,
						 tgt_key_padding_mask=tgt_key_padding_mask,
						 src_key_padding_mask=src_key_padding_mask)

		if self.norm is not None:
			output = self.norm(output)

		return output


class XATransformerDecoderLayer(nn.Module):
	r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
	This standard decoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.

	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).

	Examples::
		>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
		>>> memory = torch.rand(10, 32, 512)
		>>> tgt = torch.rand(20, 32, 512)
		>>> out = decoder_layer(tgt, memory)
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(XATransformerDecoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(XATransformerDecoderLayer, self).__setstate__(state)

	def forward(self, tgt_q: Tensor, tgt_k: Tensor, tgt_v: Tensor,
				src_k: Tensor, src_v: Tensor,
				tgt_mask: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer.

		Args:
			tgt_q: the target query sequence to the decoder layer (required).
			tgt_k: the target key sequence to the decoder (required).
			tgt_v: the target value sequence to the decoder (required).
			src_k: the sequence keys from the last layer of the encoder (required).
			src_v: the sequence values from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			src_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			src_key_padding_mask: the mask for the memory keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""

		tgt2 = self.self_attn(tgt_q, tgt_k, tgt_v, attn_mask=tgt_mask,
							  key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt_v + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		tgt2 = self.multihead_attn(tgt, src_k, src_v, attn_mask=src_mask,
								   key_padding_mask=src_key_padding_mask)[0]
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt


class PyTransformerHead(nn.Module):

	def __init__(self, **kwargs):

		super(PyTransformerHead, self).__init__()

		use_positional_encoding = kwargs.get("positional_encoding") or False
		# desc_size = kwargs['feature_output_dim_3D']
		feat_size = kwargs['feature_size']
		num_points = kwargs['num_points']

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
		xa_dec_nheads = kwargs.get("tf_xa_dec_nheads") or 1
		xa_dec_layers = kwargs.get("tf_xa_dec_layers") or 1
		xa_hiddn_size = kwargs.get("tf_xa_hiddn_size") or 1024

		self.key_size = kwargs.get("tf_key_size") or 64

		self._positional_encoding = PositionalEncodingCart3D(feat_size, **kwargs)

		sa_enc_layer = nn.TransformerEncoderLayer(d_model=num_points, nhead=sa_enc_nheads, dim_feedforward=sa_hiddn_size)
		xa_dec_layer = XATransformerDecoderLayer(d_model=num_points, nhead=xa_dec_nheads, dim_feedforward=xa_hiddn_size)

		self.sa_encoder = nn.TransformerEncoder(sa_enc_layer, num_layers=sa_enc_layers)
		self.xa_decoder = XATransformerDecoder(xa_dec_layer, num_layers=xa_dec_layers)

		panoptic_loss_weight = kwargs['panoptic_weight']
		inverse_tf_loss_weight = kwargs['inv_tf_weight']
		self._compute_reverse_tf = panoptic_loss_weight > 0 or inverse_tf_loss_weight > 0

		# self._desc_decoder_layer = nn.TransformerDecoderLayer(d_model=num_points, nhead=1, dim_feedforward=256)
		# self.desc_decoder = nn.TransformerDecoder(self._desc_decoder_layer, 1)

		# self.FC1 = nn.Linear(feat_size, desc_size)
		# self.FC2 = nn.Linear(num_points, hidden_size)
		# self.FC3 = nn.Linear(hidden_size, desc_size)
		# self.relu = nn.ReLU()

		# self.wQ = nn.Parameter(torch.rand(n_dec_layers, feat_size, self.key_size))
		# self.wK = nn.Parameter(torch.rand(n_dec_layers, feat_size, self.key_size))

	def forward(self, batch_dict, **kwargs):

		features = batch_dict['point_features_NV']
		d_bt, d_f, d_p, _ = features.shape  # Dimensions: Batch*Tuple, Features, Points
		mode = kwargs.get("mode") or "pairs"
		d_t = 2 if mode == "pairs" else 3  # Dimension: Tuple size (2 <- pairs / 3 <- triplets)
		d_b = d_bt // d_t # Dimension: Batch size

		features = features.squeeze(-1).permute(1, 0, 2)

		features = torch.nn.functional.normalize(features, dim=0)
		points = batch_dict['point_coords']
		coords = points.view(d_bt, d_p, 4)[:, :, 1:]

		src = features

		# pe_coords = coords  # coords.view(d_bt, d_p, 1, 1, 1).permute(0, 2, 3, 4, 1)
		pe = self._positional_encoding(coords)

		if pe is not None:
			src = src + pe.permute(2, 0, 1)

		coords = coords.permute(2, 0, 1)
		coords1 = coords[:, :d_b, :]
		coords2 = coords[:, d_b:2*d_b, :]

		src = self.sa_encoder(src)

		src = src[:, :2*d_b, :]
		src1 = src[:, :d_b, :]
		src2 = src[:, d_b:2*d_b, :]

		sinkhorn_matches = self.xa_decoder.forward(coords2, coords1, coords2, src1, src2)

		batch_dict['sinkhorn_matches'] = sinkhorn_matches
		# batch_dict['transport'] = xa_decoder.attention_matrix()

		transformation1 = compute_rigid_transform(coords1.permute(1, 2, 0), sinkhorn_matches.permute(1, 2, 0),
												  torch.ones((d_b, d_p), device=coords1.device))

		batch_dict['transformation'] = transformation1
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		# Generate Global Descriptor
		if self.descriptor_head != "NetVLAD":
			#descriptor = self.desc_decoder(src)

			descriptor = self.relu(self.FC1(src.permute(1, 2, 0)))
			descriptor, _ = torch.max(descriptor, dim=1)

			#descriptor = self.relu(self.FC2(descriptor.reshape(B, P)))
			#descriptor = self.relu(self.FC3(descriptor))

			batch_dict['out_embedding'] = descriptor

		return batch_dict

import copy
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import ModuleList


def _get_clones(module, n):
	"""
	TODO
	:param module:
	:param n:
	:return:
	"""
	return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
	"""
	TODO
	:param activation:
	:return:
	"""
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SATransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers
	TODO
	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		>>> src = torch.rand(10, 32, 512)
		>>> out = transformer_encoder(src)
	"""
	__constants__ = ['norm']

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(SATransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm
		self.attention = []

	def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the input through the encoder layers in turn.

		Args:
			src: the sequence to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = src

		self.attention = []

		for mod in self.layers:
			output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
			self.attention.append(attn)

		if self.norm is not None:
			output = self.norm(output)

		return output


class SATransformerEncoderLayer(nn.Module):
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.
	TODO
	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(SATransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(XATransformerEncoderLayer, self).__setstate__(state)

	def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
				src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
		r"""Pass the input through the encoder layer.
		TODO
		Args:
			src: the sequence to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src, attn


class XATransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers
	TODO
	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		>>> src = torch.rand(10, 32, 512)
		>>> out = transformer_encoder(src)
	"""
	__constants__ = ['norm']

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(XATransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm
		self.attention = []

	def forward(self, *, k: Tensor, q: Tensor, v: Tensor,
				mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
				) -> Tensor:
		r"""Pass the input through the encoder layers in turn.

		Args:
			src: the sequence to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = q

		self.attention = []

		for mod in self.layers:
			output, attn = mod(q=output, k=k, v=v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
			self.attention.append(attn)

		if self.norm is not None:
			output = self.norm(output)

		return output


class XATransformerEncoderLayer(nn.Module):
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.
	TODO
	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)
	"""

	def __init__(self, d_model, nhead, kdim=None, vdim=None, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(XATransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(XATransformerEncoderLayer, self).__setstate__(state)

	def forward(self, *, q: Tensor, k: Tensor, v: Tensor, src_mask: Optional[Tensor] = None,
				src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
		r"""Pass the input through the encoder layer.
		TODO
		Args:
			k: the source key sequence to the decoder layer (required).
			q: the source query sequence to the decoder (required).
			v: the source value sequence to the decoder (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2, attn = self.self_attn(query=q, key=k, value=v, attn_mask=src_mask,
									key_padding_mask=src_key_padding_mask)
		# TODO: Is adding Q as residual/skip the way to go??? At least that's how it works in the standard decoder's XA
		src = q + self.dropout1(src2)
		src = self.norm1(src)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src, attn


class XATransformerDecoder(nn.Module):
	r"""TransformerDecoder is a stack of N decoder layers
	TODO
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
		self.attention_self = []
		self.attention_cross = []

	def forward(self, tgt_k: Tensor, tgt_q: Tensor, tgt_v: Tensor,
				src_k: Tensor, src_q: Tensor,
				tgt_mask: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None,
				src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer in turn.
		TODO
		Args:
			tgt_k: the target key sequence to the decoder layer (required).
			tgt_q: the target query sequence to the decoder (required).
			tgt_v: the target value sequence to the decoder (required).
			src_k: the sequence keys from the last layer of the encoder (required).
			src_q: the sequence queries from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			src_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			src_key_padding_mask: the mask for the memory keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = tgt_v
		self.attention_self = []
		self.attention_cross = []

		for mod in self.layers:
			output, attn_self, attn_cross = mod(tgt_k=output, tgt_q=tgt_q, tgt_v=output,
												src_k=src_k, src_q=src_q,
												tgt_mask=tgt_mask,
												src_mask=src_mask,
												tgt_key_padding_mask=tgt_key_padding_mask,
												src_key_padding_mask=src_key_padding_mask)

			self.attention_self.append(attn_self)
			self.attention_cross.append(attn_cross)

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
	TODO
	Args:
		sa_d_model: the number of expected features in the input (required).
		mha_d_model: the number of expected features in the input (required).
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

	def __init__(self, sa_d_model, mha_d_model, sa_nhead, mha_nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
				 sa_kdim=None, sa_vdim=None, mha_kdim=None, mha_vdim=None):
		super(XATransformerDecoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(sa_d_model, sa_nhead, dropout=dropout,
											   kdim=sa_kdim, vdim=sa_vdim)
		self.multihead_attn = nn.MultiheadAttention(mha_d_model, mha_nhead, dropout=dropout,
													kdim=mha_kdim, vdim=mha_vdim)
		# Implementation of Feedforward model
		if mha_vdim is None:
			mha_vdim = mha_d_model
		self.linear0 = None
		if mha_vdim != mha_d_model:
			self.linear0 = nn.Linear(mha_d_model, mha_vdim)
		self.linear1 = nn.Linear(mha_vdim, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, mha_vdim)

		self.norm1 = nn.LayerNorm(sa_d_model)
		self.norm2 = nn.LayerNorm(mha_vdim)
		self.norm3 = nn.LayerNorm(mha_vdim)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(XATransformerDecoderLayer, self).__setstate__(state)

	def forward(self, tgt_k: Tensor, tgt_q: Tensor, tgt_v: Tensor,
				src_k: Tensor, src_q: Tensor,
				tgt_mask: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None,
				src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
		r"""Pass the inputs (and mask) through the decoder layer.
		TODO
		Args:
			tgt_k: the target key sequence to the decoder layer (required).
			tgt_q: the target query sequence to the decoder (required).
			tgt_v: the target value sequence to the decoder (required).
			src_k: the sequence keys from the last layer of the encoder (required).
			src_q: the sequence queries from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			src_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			src_key_padding_mask: the mask for the memory keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""

		tgt2, attn_self = self.self_attn(query=tgt_q, key=tgt_k, value=tgt_v,
										 attn_mask=tgt_mask,
										 key_padding_mask=tgt_key_padding_mask)
		tgt = tgt_v + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		tgt2, attn_cross = self.multihead_attn(query=src_q, key=src_k, value=tgt,
											   attn_mask=src_mask,
											   key_padding_mask=src_key_padding_mask)

		if self.linear0 is not None:
			tgt2 = self.linear0(tgt2)

		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt, attn_self, attn_cross

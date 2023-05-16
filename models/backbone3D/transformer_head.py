import torch
from torch import nn
import numpy as np

from .heads import compute_rigid_transform


class PositionalEncoding3D(nn.Module):

	def __init__(self, channels):
		super(PositionalEncoding3D, self).__init__()

	def forward(self, x):
		return x


class TransformerHead(nn.Module):

	def __init__(self, **kwargs):

		super(TransformerHead, self).__init__()

		use_positional_encoding = kwargs.get("positional_encoding") or False
		self._positional_encoding = None
		if use_positional_encoding:
			self._positional_encoding = PositionalEncoding3D()

		desc_size = kwargs['feature_output_dim_3D']
		feat_size = kwargs['feature_size']
		num_points = kwargs['num_points']
		hidden_size = kwargs.get("desc_hidden_size") or 64
		n_head = kwargs.get('tf_enc_heads') or 1
		n_layers = kwargs.get('tf_enc_layers') or 1

		self.descriptor_head = kwargs['desc_head']

		n_dec_head = kwargs.get('tf_dec_heads') or 3
		n_dec_layers = kwargs.get('tf_dec_heads') or 1
		self.key_size = kwargs.get("tf_key_size") or 64

		self._encoder_layer = nn.TransformerEncoderLayer(d_model=num_points, nhead=n_head)
		self.encoder = nn.TransformerEncoder(self._encoder_layer, num_layers=n_layers)

		panoptic_loss_weight = kwargs['panoptic_weight']
		inverse_tf_loss_weight = kwargs['inv_tf_weight']
		self._compute_reverse_tf = panoptic_loss_weight > 0 or inverse_tf_loss_weight > 0

		self.wQ = nn.Parameter(torch.rand(n_dec_layers, feat_size, self.key_size))
		self.wK = nn.Parameter(torch.rand(n_dec_layers, feat_size, self.key_size))

	def forward(self, batch_dict, **kwargs):

		points = batch_dict['point_coords']
		if self._positional_encoding is not None:
			x = self._positional_encoding(points)

		features = batch_dict['point_features_NV']
		B, F, P, _ = features.shape
		features = features.reshape(B, F, P).permute(1, 0, 2)
		src = self.encoder(features)

		# Generate Global Descriptor
		if self.descriptor_head != "NetVLAD":

			descriptor = self.relu(self.FC1(src.permute(1, 2, 0)))
			descriptor, _ = torch.max(descriptor, dim=1)

			batch_dict['out_embedding'] = descriptor

		mode = kwargs.get("mode") or "pairs"

		T = 2 if mode == "pairs" else 3
		B = B // T
		src = src[:, :2*B, :].permute(1, 2, 0)  # Only apply Q and
		src = torch.unsqueeze(src, 1)

		queries = torch.matmul(src, self.wQ)
		keys_T = torch.matmul(src, self.wK).permute(0, 1, 3, 2)

		sqrt_key_size = np.sqrt(self.key_size)
		attn1 = torch.matmul(queries[:B, :, :], keys_T[B:2*B, :, :])
		attn1 = attn1 / sqrt_key_size

		# My Ideas: Sum up all the attention matrices and normalize them by rows to
		# get the matching matrices

		matching1 = torch.sum(attn1, 1)
		matching1 = torch.relu(matching1)  # ReLU activation to remove negative values

		row_sum1 = matching1.sum(-1, keepdim=True)
		row_sum1[row_sum1 == 0] = 1
		row_sum1 = 1 / row_sum1
		matching1 = torch.multiply(matching1, row_sum1)
		row_sum1 = matching1.sum(-1, keepdim=True)

		coords = points.view(T*B, P, 4)[:, :, 1:]
		coords1 = coords[:B, :, :]
		coords2 = coords[B:2*B, :, :]

		sinkhorn_matches1 = torch.matmul(matching1, coords2)

		batch_dict['transport'] = matching1

		transformation1 = compute_rigid_transform(coords1, sinkhorn_matches1, row_sum1.squeeze(-1))

		batch_dict['transformation'] = transformation1
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		if not self._compute_reverse_tf:
			return batch_dict

		# Do it in reverse (Queries from Anchor and Keys from Positive)
		# and the resulting transform should be the inverse of the previous one

		attn2 = torch.matmul(queries[B:2*B, :, :], keys_T[:B, :, :])
		attn2 = attn2 / sqrt_key_size
		attn2 = torch.softmax(attn2, dim=2)

		matching2 = torch.sum(attn2, 1)
		# matching2 = torch.log(matching2)
		matching2 = torch.softmax(matching2, dim=1)
		row_sum2 = matching2.sum(-1, keepdim=True)

		sinkhorn_matches2 = torch.matmul(matching2, coords1)

		transformation2 = compute_rigid_transform(coords2, sinkhorn_matches2, row_sum2.squeeze(-1))

		batch_dict['sinkhorn_matches2'] = sinkhorn_matches2
		batch_dict['transport2'] = matching2
		batch_dict['transformation2'] = transformation2

		return batch_dict

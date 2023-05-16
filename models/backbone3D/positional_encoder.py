from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodingCart3D(nn.Module):
	"""
	Positional Encoding Module for 3D Cartesian Coordinates (x, y, z).
	Inspired by:
	https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
	"""
	def __init__(self, channels, **kwargs):
		"""
		TODO
		:param channels: Number of channels
		:type channels: int
		"""
		super(PositionalEncodingCart3D, self).__init__()

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
		val_range = range_max - range_min
		self.register_buffer("range_min", range_min)
		self.register_buffer("range", val_range)

	def forward(self, src: torch.Tensor) -> Union[None, torch.Tensor]:

		d_b, d_p, d_c = src.shape

		if d_c != 3:
			raise ValueError(f"Only Tensors whose last dimension is 3 are supported. Shape {src.shape[-1]}.")

		# Normalize cartesian coordinates to [0, 1] range based on the voxel grid dimensions
		norm_pos = (src - self.range_min) / self.range
		# Mutliply each coordinate b the frequency
		sin_inp = torch.einsum("...k,l->...kl", norm_pos, self.freq)

		# Compute the sin and cos, and stack them
		emb = torch.stack((torch.sin(sin_inp), torch.cos(sin_inp)), dim=-1)
		# Interleave, so that we get sin, cos, sin, cos...
		emb = emb.view(d_b, d_p, d_c, self._subchannels)
		# Reshape so that we get (sin, cos, sin, ...) for x, then (sin, cos, ...) for y and finally for z
		emb = emb.permute(0, 1, 3, 2).reshape(d_b, d_p, 3 * self._subchannels)
		emb = emb[:, :, :self.channels]

		# Normalize to be of the same magnitude as the tensor it will be added to
		emb = F.normalize(emb, dim=2)

		return emb

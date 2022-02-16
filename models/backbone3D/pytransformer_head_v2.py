from torch import nn
import torch.nn.functional as F

from .heads import compute_rigid_transform
from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer
from utils.tools import SVDNonConvergenceError


def shannon_entropy(*, p, dim=-1, **_):
	"""

	:param p: Tensor of probabilities in [0, 1]
	:param dim: Dimension along which the probabilities add up to one

	:return:
	"""
	return - (p * p.log()).sum(dim=dim)


def hill_number(*, p, q, dim=-1, **_):
	"""

	:param p: Tensor of probabilities in [0, 1]
	:param q: Order of the diversity index.
	:param dim: Dimension along which the probabilities add up to one

	:return:
	"""
	if q == 1:
		return shannon_entropy(p=p).exp()

	return (p ** q).sum(dim=dim) ** (1 / (1 - q))


def norm_diversity(*, d, n, **_):
	return (1 / (n - 1)) * (n - d)


def norm_hill_number(*, p, q, dim=-1, **_):
	n = p.shape[dim]
	d = hill_number(p=p, q=q, dim=dim)
	return norm_diversity(d=d, n=n)


def berger_parker_index(*, p, dim=-1, **_):
	return p.max(dim=dim).values


def weight_sum(*, p, dim=-1, **_):
	return p.sum(dim=dim)


class PyTransformerHead2(nn.Module):
	"""
	TODO
	"""

	def __init__(self, *, feature_size,
				 tf_xa_enc_nheads=1, tf_xa_enc_layers=1, tf_xa_hiddn_size=None,
				 point_weighting_method="weight_sum", point_weighting_method_order=2,
				 **_):

		super(PyTransformerHead2, self).__init__()

		# Cross Attention Layer Parameters
		xa_enc_layers = tf_xa_enc_layers or 1
		# Default hidden size as done in the original paper "Attention is all you need": 4 times d_model
		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size

		sa_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
												 kdim=feature_size, vdim=3,
												 dim_feedforward=xa_hiddn_size)

		weighting_methods = {
			"hill": {"f": norm_hill_number, "kwargs": {"dim": -1}},
			"berger": {"f": berger_parker_index, "kwargs": {"dim": -1}},
			"weight_sum": {"f": weight_sum, "kwargs": {"dim": 1}}
		}

		if point_weighting_method not in weighting_methods:
			raise ValueError(f"Invalid point weighting method ({point_weighting_method}). Valid values: {weighting_methods.keys()}.")
		weighting_method = weighting_methods[point_weighting_method]
		weighting_method_kwargs = weighting_method["kwargs"].copy()
		weighting_method_kwargs["q"] = point_weighting_method_order
		self._weighting_method = weighting_method["f"]
		self._weighting_method_kwargs = weighting_method_kwargs

		self.xa_encoder = XATransformerEncoder(sa_enc_layer, num_layers=xa_enc_layers)
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

		tf_out = self.xa_encoder(k=features1, q=features2, v=coords2)
		attn_matrix = self.xa_encoder.attention[-1]

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

			raise SVDNonConvergenceError("SVD did not converge!")

		batch_dict['transformation'] = transformation
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		return batch_dict

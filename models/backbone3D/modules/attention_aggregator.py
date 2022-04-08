import torch.nn as nn


def _agg_sum(attn_matrices):
	return attn_matrices


def _agg_prod(attn_matrices):
	return attn_matrices


def _agg_hadamard(attn_matrices):
	return attn_matrices


def _agg_last(attn_matrices):
	return attn_matrices[-1]


class AttentionAggregator(nn.Module):

	_AGG_METHODS = {
		"sum": _agg_sum,
		"product": _agg_prod,
		"hadamard": _agg_hadamard,
		"last": _agg_last,
		None: _agg_last
	}

	def __init__(self, *, agg_method=None, **_):
		super(AttentionAggregator, self).__init__()

		if agg_method not in self._AGG_METHODS:
			raise KeyError(f"Invalid attention matrix aggregation method ({agg_method}). "
						   f"Valid values: [{self._AGG_METHODS.keys()}].")

		self._agg_function = self._AGG_METHODS[agg_method]

	def forward(self, attn_matrices):
		if not isinstance(attn_matrices, list):
			attn_matrices = [attn_matrices]

		return self._agg_function(attn_matrices)
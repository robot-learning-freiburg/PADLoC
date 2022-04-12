
def _agg_sum(attn_matrices):
	if len(attn_matrices) == 1:
		return attn_matrices[0]

	agg = attn_matrices[0]
	for m in attn_matrices[1:]:
		agg = agg + m

	return agg


def _agg_prod(attn_matrices):
	if len(attn_matrices) == 1:
		return attn_matrices[0]

	agg = attn_matrices[0]
	for m in attn_matrices[1:]:
		# (Am * (... (A2 * (A1 * (A0))))) ???
		agg = m @ agg

	return agg


def _agg_hadamard(attn_matrices):

	if len(attn_matrices) == 1:
		return attn_matrices[0]

	agg = attn_matrices[0]
	for m in attn_matrices[1:]:
		agg = agg * m

	return agg


def _agg_last(attn_matrices):
	return attn_matrices[-1]


class AttentionAggregator:

	_AGG_METHODS = {
		"sum": _agg_sum,
		"product": _agg_prod,
		"hadamard": _agg_hadamard,
		"last": _agg_last
	}

	def __init__(self, *, agg_method=None, **_):

		if agg_method not in self._AGG_METHODS:
			raise KeyError(f"Invalid attention matrix aggregation method ({agg_method}). "
						   f"Valid values: [{self._AGG_METHODS.keys()}].")

		self._agg_function = self._AGG_METHODS[agg_method]

	def __call__(self, attn_matrices):
		return self.forward(attn_matrices)

	def forward(self, attn_matrices):
		if not isinstance(attn_matrices, list):
			attn_matrices = [attn_matrices]

		return self._agg_function(attn_matrices)
import torch


def shannon_entropy(p, dim=-1, **_):
	"""
	TODO: DOC
	:param p: Tensor of probabilities in [0, 1]
	:param dim: Dimension along which the probabilities add up to one

	:return:
	"""
	return - (p * p.log()).sum(dim=dim)


def hill_number(p, order=1, dim=-1, **_):
	"""

	:param p: Tensor of probabilities in [0, 1]
	:param order: Order of the diversity index.
	:param dim: Dimension along which the probabilities add up to one

	:return:
	"""
	if order == 1:
		return shannon_entropy(p=p).exp()

	return (p ** order).sum(dim=dim) ** (1 / (1 - order))


def norm_diversity(*, diversity_index, n, **_):
	"""

	:param diversity_index:
	:param n:
	:param _:
	:return:
	"""
	return (1 / (n - 1)) * (n - diversity_index)


def norm_hill_number(p, order=1, dim=-1, **_):
	"""

	:param p:
	:param order:
	:param dim:
	:param _:
	:return:
	"""
	n = p.shape[dim]
	d = hill_number(p, order=order, dim=dim)
	return norm_diversity(diversity_index=d, n=n)


def berger_parker_index(p, dim=-1, normalize=True, **_):
	"""

	:param p:
	:param dim:
	:param normalize:
	:param _:
	:return:
	"""
	w = p.max(dim=dim).values

	if normalize:
		# Normalize to [0, 1] such that
		# if max(p) = 1 (Ultra-sharp distribution) -> w = 1,
		# if max(p) = 1/n (Ultra-flat distribution) -> w = 0
		n = p.shape[dim]
		w = (n / (n - 1)) * (w - (1 / n))

	return w


def weight_sum(p, dim=-1, **_):
	"""

	:param p:
	:param dim:
	:param _:
	:return:
	"""
	return p.sum(dim=dim)


def uniform_weights(p, dim=-1, **_):
	"""

	:param p:
	:param dim:
	:param _:
	:return:
	"""

	shape = p.shape

	if dim < 0:
		dim = len(shape) + dim

	shape = shape[:dim] + shape[dim + 1:]
	return torch.ones(shape, device=p.device)


class MatchWeighter:

	_WEIGHT_METHODS = {
		"hill": {"f": norm_hill_number, "kwargs": {"dim": -1, "order": 1}},
		"berger": {"f": berger_parker_index, "kwargs": {"dim": -1, "normalize": False}},
		"weight_sum": {"f": weight_sum, "kwargs": {"dim": 1}},
		"uniform": {"f": uniform_weights, "kwargs": {}},
		None: {"f": uniform_weights, "kwargs": {}},
	}

	def __init__(self, *,
				 weighting_method: None,
				 **kwargs):

		if weighting_method not in self._WEIGHT_METHODS:
			raise KeyError(f"Invalid match-weighting method. Valid values: ({self._WEIGHT_METHODS.keys()}, None).")

		weighting_config = self._WEIGHT_METHODS[weighting_method]
		self._weighting_f = weighting_config["f"]
		self._kwargs = weighting_config["kwargs"].copy()
		for k in self._kwargs:
			if k in kwargs:
				self._kwargs[k] = kwargs[k]

	def __call__(self, matching_matrices, **kwargs):
		return self.forward(matching_matrices, **kwargs)

	def forward(self, matching_matrices, **kwargs):
		tmp_kwargs = self._kwargs.copy()
		tmp_kwargs.update(kwargs)

		return self._weighting_f(matching_matrices, **tmp_kwargs)

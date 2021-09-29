import torch


def soft_kronecker(a, b=None, sigma=0.1):
	"""
	Soft Kronecker Delta

	"""

	a_dim = a.dim()
	if a_dim == 1:
		pass
	elif a_dim == 2:
		pass
	else:
		raise TypeError("Only for 1 and 2D tensors.")
	batch_a, channel_a = a.size()

	if b is None:
		b = a

	else:
		b_dim = b.dim()
		if a_dim != b_dim:
			raise TypeError("Dims dont match: {} and {}".format(a_dim, b_dim))

	Bb, Nb = b.size()

	if batch_a != Bb:
		raise TypeError("Batch Sizes dont match")

	amat = a.reshape(batch_a, channel_a, 1).repeat_interleave(Nb, dim=2).transpose(2, 1)
	bmat = b.reshape(batch_a, Nb, 1).repeat_interleave(channel_a, dim=2)

	diff = torch.exp(- torch.square((amat - bmat) / sigma))

	return diff

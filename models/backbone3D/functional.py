import torch


def vector_pairwise_diff(a, b=None):
	"""
	Kronecker Matrix Difference:
	Takes two vectors in and computes a matrix with the pair-wise difference between their elements.
	E.g. for vectors:
	        * a = (a1 a2 a3)
	        * b = (b1 b2 b3 b4)
	    A matrix comparing all entries between the two will be returned:
	                     b1    b2    b3    b4
	                 ⎛ a1-b1 a1-b2 a1-b3 a1-b4 ⎞ a1
	        * diff = ⎜ a2-b1 a2-b2 a2-b3 a2-b4 ⎟ a2
	                 ⎝ a3-b1 a3-b2 a3-b3 a3-b4 ⎠ a3

	Shapes:
		* B:  Batch dimension
		* Ca: Channel dimension of vector a
		* Cb: Channel dimension of vector b

	:param a: (Shape: (B, Ca)) First vector to do the difference.
	:param b: (Shape: (B, Cb)) (Optional) Second vector to do the difference with.
	                           If None, a will be compared to itself.

	:return: (Shape: (B, Ca, Cb)) Matrices of pairwise differences between the elements of a and b.
	"""

	a_dim = a.dim()
	if a_dim != 1 and a_dim != 2:
		raise TypeError("Only for 1 and 2D tensors.")
	batch_a, channel_a = a.size()

	if b is None:
		b = a

	else:
		b_dim = b.dim()
		if a_dim != b_dim:
			raise TypeError(f"Dims dont match: {a_dim} and {b_dim}")

	batch_b, channel_b = b.size()

	if batch_a != batch_b:
		raise TypeError("Batch Sizes dont match")

	a_mat = a.reshape(batch_a, channel_a, 1).repeat_interleave(channel_b, dim=2).transpose(2, 1)
	b_mat = b.reshape(batch_a, channel_b, 1).repeat_interleave(channel_a, dim=2)

	diff = a_mat - b_mat

	return diff


def soft_kronecker(a, b=None, sigma=0.1):
	"""
	Soft Kronecker Delta

	Performs the pair-wise comparison between the elements of vector a and vector b, returning a matrix where
	the rows represent the entries of a, and the columns represent the entries of b.
	The more similar they are, the closer the matrix entry corresponding to that pair of elements will be to 1.
	The more different they are, the closer the matrix entry will be to 0.
	It is soft, since it uses a gaussian to measure the difference, instead of just returning 1 for equal,
	0 for different, and is thus differentiable.

	Shapes:
		* B:  Batch dimension
		* Ca: Channel dimension of vector a
		* Cb: Channel dimension of vector b

	:param a: (Shape: (B, Ca)) Vector or Batch of vectors to do Soft Kronecker delta to.
	:param b: (Optional) (Shape: (B, Cb)) Vector or batch of vectors to do Soft Kronecker delta with a.
	          If None, then the Soft Kronecker will be performed between a and a.
	:param sigma: (float, Default=0.1) Standard deviation. Soft Kronecker performs a Normal distribution
	               over the difference between a and b.

	:return: (Shape: (B, Ca, Cb) Matrices of pair-wise comparisons between a and b.
	"""

	diff = vector_pairwise_diff(a, b)

	kronecker_mat = torch.exp(- torch.square(diff / sigma))

	return kronecker_mat


def hard_kronecker(a, b=None):
	"""
	Hard Kronecker Delta
	Performs the pair-wise comparison between the elements of vector a and vector b, returning a matrix where
	the rows represent the entries of a, and the columns represent the entries of b.
	If the entry of a is equal to the entry of be, then the matrix will have a 1 in the corresponding row and column.
	Otherwise the entry will be 0.
	It is NON-DIFFERENTIABLE!

	Shapes:
		* B:  Batch dimension
		* Ca: Channel dimension of vector a
		* Cb: Channel dimension of vector b

	:param a: (Shape: (B, Ca)) Vector or Batch of vectors to do Hard Kronecker delta to.
	:param b: (Shape: (B, Cb)) (Optional) Vector or batch of vectors to do Hard Kronecker delta with a.
	          If None, then the Hard Kronecker will be performed between a and a.

	:return: (Shape: (B, Ca, Cb) Matrices of pair-wise comparisons between a and b.
	"""

	diff = vector_pairwise_diff(a, b)

	kronecker_mat = torch.zeros_like(diff, device=diff.device)
	kronecker_mat[diff == 0] = 1

	return kronecker_mat

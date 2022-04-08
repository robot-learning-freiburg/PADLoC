
def split_apn_data(tensor, mode="pairs", slice_dim=0, get_negatives=False):
	"""
	Splits a tensor that has anchor, positive and potentially negative data concatenated along slice_dim.

	:param tensor:
	:param mode:
	:param slice_dim:
	:param get_negatives:
	:return:
	"""

	dims = len(tensor.shape)

	if (slice_dim >= dims) or (slice_dim < 0):
		raise ValueError(f"Invalid batch dimension ({slice_dim}). "
						 f"Value must be between 0 and the tensor dimensions ({dims}).")

	bt = tensor.shape[slice_dim]  # Batch-Tuple dimension

	if isinstance(mode, str):
		t = 2 if mode == "pairs" else 3  # Dimension: Tuple size (2 <- pairs / 3 <- triplets)
	elif isinstance(mode, int):
		t = mode
	else:
		raise TypeError(f"Unsupported type for mode ({type(mode)}. Supported types: (str, int)")

	b = bt // t  # Dimension: Batch size
	assert b * t == bt

	# Move batch dimension to be the first,
	if slice_dim != 0:
		tensor = tensor.transpose(slice_dim, 0)

	splits = []
	num_splits = 3 if t == 3 and get_negatives else 2
	for i in range(num_splits):
		# Then slice
		split = tensor[i * b: (i + 1) * b]

		# Finally permute batch dimension back to its original place
		if slice_dim != 0:
			split = split.transpose(0, slice_dim)

		splits.append(split)

	return tuple(splits)

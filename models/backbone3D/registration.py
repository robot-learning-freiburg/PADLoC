import torch

from .heads import compute_rigid_transform
from utils.tools import SVDNonConvergenceError


class SVDRegistration:

	def __init__(self, *, debug=True, **_):
		self.debug = debug

	def __call__(self, *, src_coords, tgt_coords, weights=None):
		return self.forward(src_coords=src_coords, tgt_coords=tgt_coords, weights=None)

	def forward(self, *, src_coords, tgt_coords, weights=None):

		if weights is None:
			b, m, _ = src_coords.shape
			weights = torch.ones((b, m), device=src_coords.device)

		try:
			return compute_rigid_transform(src_coords, tgt_coords, weights)

		except RuntimeError as e:
			if self.debug:
				print("SVD did not converge!!!!!")
				print("Debug Info:")
				print("\n\n\nsrc_coords: ", src_coords)
				print("\n\n\ntgt_coords:   ", tgt_coords)
				print("\n\n\n")
			print(e)
			raise SVDNonConvergenceError("SVD did not converge!")
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np

# Use Latex-style fonts
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.size'] = 16


class Monochrome:

	def __init__(self):

		self.pfx = "bw"

		self.dpi = 300
		self.point_cloud_dpi = 400

		self.linestyles = [
			"solid",
			(0, (5, 1)),  # densely dashed
			"dotted",
			(0, (3, 1, 1, 1)),  # densely dash-dotted,
			"dashed",
			(0, (3, 1, 1, 1, 1, 1)),  # densely dash-dot-dotted
		]

		self.cycler_markers = cycler("marker", ["^", "o", "s", "p", "P", "*"])
		self.cycler_linesstyles = cycler("linestyle", self.linestyles)
		self.cycler_colors = cycler("color", ["k"])
		self.cycler_lines = (self.cycler_colors * self.cycler_linesstyles)
		self.cycler_markerlines = (self.cycler_lines + self.cycler_markers)

		self.grid = dict(color="k", linestyle=":", linewidth=0.5, alpha=None)
		self.inset = dict(fc="white", ec="k", linewidth=0.5, linestyle="--")
		self.inset_label_bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
		self.curves = dict(linewidth=1.0)

		self.src_point_cloud = dict(s=0.1, c="k", marker="o", lw=0)
		self.src_sampled_point_cloud = dict(s=0.3, c="0.40", marker="*", lw=0)
		self.src_predicted_point_cloud = dict(s=0.3, c="0.40", marker="p", lw=0)

		self.tgt_point_cloud = dict(s=0.1, c="k", marker="o", lw=0)
		self.tgt_sampled_point_cloud = dict(s=0.3, c="k", marker="*", lw=0)
		self.tgt_predicted_point_cloud = dict(s=0.3, c="0.40", marker="p", lw=0)

		# Match confidence lines parameters
		self.match_lines_auto_range = True  # False
		self.match_lines_weight_function = None  # lambda x: torch.sigmoid(10 * (x-0.5))
		self.match_lines_min_lw = 0.05
		self.match_lines_max_lw = 0.3
		self.match_lines_lw_range = self.match_lines_max_lw - self.match_lines_min_lw
		self.match_lines_min_alpha = 0.1
		self.match_lines_max_alpha = 0.30
		self.match_lines_alpha_range = self.match_lines_max_alpha - self.match_lines_min_alpha

		# Registration parameters
		self.blend_method = "multiply"
		self.src_reg_point_cloud = dict(s=0.1, c="0.4", marker="o", lw=0)
		self.tgt_reg_point_cloud = dict(s=0.1, c="0.2", marker="o", lw=0)
		# self.ovr_reg_point_cloud = dict(s=0.1, c="k", marker="o", lw=0)
		# self.ovr_reg_point_cloud_color = np.array([0., 0., 0.])

		# Loop-Closure detection paths
		self.path_dpi = 300
		self.tp = dict(c="0.25", s=4, marker="o", lw=0)
		self.tn = dict(c="0.75", lw=0.5)
		self.fp = dict(c="0.0", s=1, marker="X", lw=0)
		self.fn = dict(c="0.5", s=2, marker="v", lw=0)

	def preprocess_confidence(self, confidence_weights):
		# Make confidence weights 1D if they are not
		if len(confidence_weights.shape) > 1:
			confidence_weights = confidence_weights.flatten()

		confidence_weights = self.normalize_confidence(confidence_weights)

		# Apply some non-linear transformation
		if self.match_lines_weight_function is not None:
			confidence_weights = self.match_lines_weight_function(confidence_weights)
			confidence_weights = self.normalize_confidence(confidence_weights)

		if self.match_lines_auto_range:
			min_w, max_w = confidence_weights.min(), confidence_weights.max()
			if max_w - min_w > 1e-6:
				confidence_weights = (confidence_weights - min_w) / (max_w - min_w)

		# Convert to numpy
		confidence_weights = confidence_weights.cpu().numpy()

		return confidence_weights

	@staticmethod
	def normalize_confidence(confidence_weights):
		# Normalize confidence weights to add up to one
		conf_sum = confidence_weights.sum()

		if abs(1 - conf_sum) > 1e-6:
			confidence_weights = confidence_weights / conf_sum

		return confidence_weights

	def match_lines_styles(self, confidence_weights):

		confidence_weights = self.preprocess_confidence(confidence_weights)

		linewidths = self.match_lines_lw_range * confidence_weights + self.match_lines_min_lw
		alphas = self.match_lines_alpha_range * confidence_weights + self.match_lines_min_alpha

		colors = np.zeros((linewidths.shape[0], 4))
		colors[:, 3] = alphas

		return dict(linewidths=linewidths, colors=colors)

	def blend_registration_buffers(self, img1, img2):
		# img1[img1[:, :, -1] == 0] = 0
		# img2[img2[:, :, -1] == 0] = 0
		#
		# img1 = img1.copy().astype(np.float) / 255
		# img2 = img2.copy().astype(np.float) / 255
		#
		# img = img1 + img2
		# overlap_mask = img[:, :, 3] > 1.
		#
		# img[overlap_mask, :3] = self.ovr_reg_point_cloud_color
		# img[overlap_mask, 3] /= 2
		#
		# np.clip(img, a_min=0, a_max=1) * 255
		# img = img.astype(np.uint8)
		#
		# return img

		if self.blend_method not in ["multiply", "min"]:
			# Make RGB zero if A==0
			img1[img1[:, :, -1] == 0] = 0
			img2[img2[:, :, -1] == 0] = 0

		if self.blend_method == "max":
			return np.maximum(img1, img2)

		if self.blend_method == "min":
			return np.minimum(img1, img2)

		if self.blend_method == "sum":
			# Increase resolution to avoid overflows
			img1 = img1.copy().astype(np.uint16)
			img2 = img2.copy().astype(np.uint16)

			img = np.minimum(img1 + img2, 255).astype(np.uint8)

			return img

		if self.blend_method == "sum_inv":
			# Increase resolution to avoid overflows
			img1 = img1.copy().astype(np.uint16)
			img2 = img2.copy().astype(np.uint16)

			img = img1 + img2
			img[img[:, :, 3] > 255, :3] = 255 - img[img[:, :, 3] > 255, :3]
			img = np.minimum(img, 255).astype(np.uint8)

			return img

		if self.blend_method == "multiply":
			img1 = img1[:, :, :-1].copy().astype(np.float) / 255
			img2 = img2[:, :, :-1].copy().astype(np.float) / 255

			img = np.clip(img1 * img2, a_min=0, a_max=1) * 255
			img = img.astype(np.uint8)

			return img

		if self.blend_method == "abs_diff":
			img1 = img1[:, :, :-1].copy().astype(np.float) / 255
			img2 = img2[:, :, :-1].copy().astype(np.float) / 255

			img = 1 - (img2 + (1 - img1))

			img = np.clip(img, a_min=0, a_max=1) * 255
			img = img.astype(np.uint8)

			return img

		raise NotImplementedError("Invalid blending method.")


class Color(Monochrome):

	def __init__(self):
		super().__init__()

		self.pfx = "cl"

		self.colors = [  # Taken from Dark2
			"#1b9e77",
			"#d95f02",
			"#7570b3",
			"#a6761d",
			"#666666",
			"#e6ab02",
			# "#e7298a",
		]

		self.cycler_colors = cycler('color', self.colors)

		self.cycler_lines = (self.cycler_colors + self.cycler_linesstyles)

		self.cycler_markerlines = (self.cycler_lines + self.cycler_markers)

		self.grid = dict(color="0.5", linestyle=":", linewidth=0.5, alpha=None)
		self.inset = dict(fc="white", ec="0.5", linewidth=0.5, linestyle="--")

		self.src_point_cloud = dict(s=0.1, c="#10401B", marker="o", lw=0)
		self.src_sampled_point_cloud = dict(s=0.3, c="#10401B", marker="*", lw=0)
		self.src_predicted_point_cloud = dict(s=0.3, c="#081A0D", marker="p", lw=0)

		self.tgt_point_cloud = dict(s=0.1, c="#3F0909", marker="o", lw=0)
		self.tgt_sampled_point_cloud = dict(s=0.3, c="#190809", marker="*", lw=0)
		self.tgt_predicted_point_cloud = dict(s=0.3, c="#190809", marker="p", lw=0)

		# Match confidence lines parameters
		self.match_lines_auto_range = True  # False
		self.match_lines_weight_function = None  # lambda x: torch.sigmoid(10 * (x-0.5))
		self.match_lines_min_lw = 0.1
		self.match_lines_max_lw = 0.6
		self.match_lines_lw_range = self.match_lines_max_lw - self.match_lines_min_lw
		self.match_lines_min_alpha = 0.15
		self.match_lines_max_alpha = 0.60
		self.match_lines_alpha_range = self.match_lines_max_alpha - self.match_lines_min_alpha

		# Registration parameters
		self.blend_method = "sum"  # "sum"
		self.src_reg_point_cloud = dict(s=0.1, c="#2d7c2c", marker="o", lw=0)
		self.tgt_reg_point_cloud = dict(s=0.1, c="#7a0a0c", marker="o", lw=0)
		# self.ovr_reg_point_cloud = dict(s=0.1, c="#002857", marker="o", lw=0)
		# self.ovr_reg_point_cloud_color = np.array([0., 0.1568627451, 0.3411764706])
		# self.src_reg_point_cloud = dict(s=0.1, c="#00FF00", marker="o", lw=0)
		# self.tgt_reg_point_cloud = dict(s=0.1, c="#FF0000", marker="o", lw=0)

		# Loop-Closure detection paths
		self.tp = dict(c="#1b5a1e", s=3, lw=0)  # Green
		self.tn = dict(c="0.25", lw=0.5)
		self.fp = dict(c="#e51009", s=1, lw=0)  # Red
		self.fn = dict(c="#1d0ad8", s=2, lw=0)  # Blue

	def match_lines_styles(self, confidence_weights):

		confidence_weights = self.preprocess_confidence(confidence_weights)

		linewidths = self.match_lines_lw_range * confidence_weights + self.match_lines_min_lw
		alphas = self.match_lines_alpha_range * confidence_weights + self.match_lines_min_alpha

		cm = plt.cm.get_cmap("YlOrRd")

		colors = cm(confidence_weights)
		colors[:, 3] = alphas

		return dict(linewidths=linewidths, colors=colors)


class Style:
	_STYLE_DICT = {
		"bw": Monochrome,
		"color": Color
	}

	def __new__(cls, style, use_latex=True):
		if style not in cls._STYLE_DICT:
			raise NotImplementedError(f"Invalid style ({style}). Supported values: {cls._STYLE_DICT.keys()}.")

		if use_latex:
			try:
				plt.rcParams['text.usetex'] = True
				plt.rcParams['font.family'] = "serif"
				plt.rcParams['font.size'] = 16
			except Exception:
				print("Unable to set matplotlib to use a Latex style.")

		return cls._STYLE_DICT[style]()


from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deep_closest_point import Transformer, SVDHead


class DeepClosestPointHead(nn.Module):
	"""
	Based on

	Deep Closest Point: Learning Representations for Point Cloud Registration
	https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf
	"""

	def __init__(self, *, feature_size,
				 tf_enc_layers=1,
				 tf_nheads=4,
				 tf_hiddn_size=1024,
				 dropout=0.0,
				 **_):

		super(DeepClosestPointHead, self).__init__()

		tf_hiddn_size = tf_hiddn_size or 4 * feature_size

		args = {
			"emb_dims": feature_size,
			"n_blocks": tf_enc_layers,
			"dropout": dropout,
			"ff_dims": tf_hiddn_size,
			"n_heads": tf_nheads,
		}

		args = Namespace(**args)

		self.pointer = Transformer(args=args)
		self.head = SVDHead(args=args)

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

		# features = features.permute(0, 2, 1)
		# coords = coords.permute(1, 0, 2)
		coords = coords.permute(0, 2, 1)

		# Normalize Features for some reason. Probably not required, since the KQV weights will mess them up anyways
		# features_norm = F.normalize(features, dim=1)

		# Split into anchor and positive features/coordinates
		features1 = features[:d_b, :, :]
		features2 = features[d_b:2*d_b, :, :]
		coords1 = coords[:d_b, :, :]  # Coordinates of PC1
		coords2 = coords[d_b:2*d_b, :, :]  # Coordinates of PC2


		src_embedding_p, tgt_embedding_p = self.pointer(features1, features2)

		src_embedding = features1 + src_embedding_p
		tgt_embedding = features2 + tgt_embedding_p

		transform_ab, tgt_coords, attn_matrix = self.head(src_embedding, tgt_embedding, coords1, coords2)
		transform_ba, _, _ = self.head(tgt_embedding, src_embedding, coords2, coords1)


		batch_dict['transport'] = attn_matrix
		batch_dict['sinkhorn_matches'] = tgt_coords

		batch_dict['transformation'] = transform_ab
		batch_dict['out_rotation'] = None
		batch_dict['out_translation'] = None

		return batch_dict

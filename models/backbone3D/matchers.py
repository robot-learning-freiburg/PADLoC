import torch.nn as nn

from .xatransformer import XATransformerEncoder, XATransformerEncoderLayer


class TFEncMatcher(nn.Module):

	def __init__(self,  *, feature_size,
				 tf_xa_enc_nheads=4, tf_xa_enc_layers=1,
				 tf_xa_hiddn_size=None,
				 # tf_skip_conn1=False,
				 # tf_skip_conn2=True,
				 dropout=0.1,
				 **_):

		super().__init__()

		xa_hiddn_size = (feature_size * 4) if tf_xa_hiddn_size is None else tf_xa_hiddn_size

		sa_enc_layer = XATransformerEncoderLayer(d_model=feature_size, nhead=tf_xa_enc_nheads,
												 kdim=feature_size, vdim=3,
												 dim_feedforward=xa_hiddn_size, dropout=dropout,
												 skip_conn1=True, skip_conn2=True)
		self.tf = XATransformerEncoder(sa_enc_layer, num_layers=tf_xa_enc_layers)
		# Remove the skip connection for the last encoder layer, so as to not get a mixture of Q + V in the final output.
		self.tf.layers[-1].skip_conn1 = False

		self.linear = nn.Linear(feature_size, 3)

	def forward(self, *, src_features, tgt_features, tgt_coords):
		x = self.tf(q=src_features, k=tgt_features, v=tgt_coords)
		x = self.linear(x)

		return x, self.tf.attention

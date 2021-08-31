import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class PointNet2Geometric(torch.nn.Module):
    def __init__(self):
        super(PointNet2Geometric, self).__init__()
        self.sa1_module = SAModule(0.5, 0.05, MLP([3, 32, 32, 64]))
        self.sa2_module = SAModule(0.25, 0.1, MLP([64 + 3, 64, 64, 128]))
        self.sa3_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256]))
        self.sa4_module = SAModule(0.25, 0.4, MLP([256 + 3, 256, 256, 512]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp4_module = FPModule(3, MLP([512 + 256, 256, 256]))
        self.fp3_module = FPModule(3, MLP([256 + 128, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 64, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128, 128, 128, 128]))

    def forward(self, x, *args):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        B, _, N, C = x.shape
        batch_dict = {}
        batch_idx = torch.arange(B, device=x.device).view(-1, 1).repeat(1, x.shape[2]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), x.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords.clone()
        batch_dict['batch_size'] = B
        x = x / 100.

        sa0_out = (None, x.view(-1, 3), batch_idx)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = x.view(B, -1, N, 1)

        batch_dict['point_features'] = x
        batch_dict['point_features_NV'] = x

        return batch_dict

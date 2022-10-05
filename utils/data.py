import time

import json
import torch
from torch.utils.data.dataloader import default_collate

from datasets.KITTI360Dataset import KITTI3603DDictPairs, KITTI3603DDictTriplets
from datasets.KITTI_DGR import KITTIDGR3DDictPairs
from datasets.KITTI_RPMNet import KITTIRPM3DDictPairs
from datasets.KITTI_data_loader import KITTILoader3DDictPairs, KITTILoader3DDictTriplets
import torch.utils.data


def datasets_concat_kitti(data_dir, sequences_list, transforms, data_type, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        poses_file = data_dir + "/sequences/" + sequence + "/poses.txt"
        if data_type == "RGB":
            raise NotImplementedError("Not Implemented")
        elif data_type == "3D":
            d = KITTILoader3DDictPairs(data_dir, sequence, poses_file, points_num, device, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti_triplets(data_dir, sequences_list, transforms, data_type, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        poses_file = data_dir + "/sequences/" + sequence + "/poses_SEMANTICKITTI.txt"
        if data_type == "RGB":
            raise NotImplementedError("Not Implemented")
        elif data_type == "3D":
            d = KITTILoader3DDictTriplets(data_dir, sequence, poses_file, points_num, device, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti360(data_dir, sequences_list, transforms, data_type, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        if data_type == "RGB":
            raise NotImplementedError("Not Implemented")
        elif data_type == "3D":
            d = KITTI3603DDictPairs(data_dir, sequence, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti360_triplets(data_dir, sequences_list, transforms, data_type, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        if data_type == "RGB":
            raise NotImplementedError("Not Implemented")
        elif data_type == "3D":
            d = KITTI3603DDictTriplets(data_dir, sequence, **kwargs)
        else:
            raise TypeError("Unknown data type to load")

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti_rpmnet(data_dir, sequences_list, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        poses_file = data_dir + "/sequences/" + sequence + "/poses_SEMANTICKITTI.txt"
        d = KITTIRPM3DDictPairs(data_dir, sequence, poses_file, points_num, device, **kwargs)

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list


def datasets_concat_kitti_DGR(data_dir, sequences_list, points_num, device, **kwargs):

    dataset_list = []

    for sequence in sequences_list:
        poses_file = data_dir + "/sequences/" + sequence + "/poses_SEMANTICKITTI.txt"
        d = KITTIDGR3DDictPairs(data_dir, sequence, poses_file, points_num, device, **kwargs)

        dataset_list.append(d)

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset, dataset_list



def get_dataset3d_mean_std(dataset):
    # xyz
    # mean: 0.5038, 0.5316, 0.2084
    # std: 0.0783, 0.0768, 0.0951
    # max: 3.7727, 2.6273, 6.5022
    # min: -3.8884, -3.0982, 0.3578

    # xs_max, ys_max, zs_max, xs_min, ys_min, zs_min = get_min_max(dataset)

    xs_max = 3.7727
    ys_max = 2.6273
    zs_max = 6.5022
    xs_min = -3.8884
    ys_min = -3.0982
    zs_min = 0.3578

    xs_mean = 0
    ys_mean = 0
    zs_mean = 0
    xs_std = 0
    ys_std = 0
    zs_std = 0
    cont = 0

    for sample in dataset:
        pcd = sample['anchor'].to('cuda:0')
        xs = pcd[:, 0]
        ys = pcd[:, 1]
        zs = pcd[:, 2]

        if cont % 100 == 0:
            print(cont, len(dataset))
            print(zs.min())

        xs = (xs - xs_min) / (xs_max - xs_min)
        ys = (ys - ys_min) / (ys_max - ys_min)
        zs = (zs - zs_min) / (zs_max - zs_min)

        xs_mean += xs.mean()
        ys_mean += ys.mean()
        zs_mean += zs.mean()

        xs_std += xs.std()
        ys_std += ys.std()
        zs_std += zs.std()

        cont += 1

    xs_mean /= cont
    ys_mean /= cont
    zs_mean /= cont

    xs_std /= cont
    ys_std /= cont
    zs_std /= cont

    print(xs_mean, ys_mean, zs_mean)
    print(xs_std, ys_std, zs_std)


def get_min_max(dataset):
    xs_min = 100000
    ys_min = 100000
    zs_min = 100000
    xs_max = 0
    ys_max = 0
    zs_max = 0
    cont = 0

    for sample in dataset:
        pcd = sample['anchor'].to('cuda:0')
        xs = pcd[:, 0]
        ys = pcd[:, 1]
        zs = pcd[:, 2]

        x_current_max = xs.max()
        y_current_max = ys.max()
        z_current_max = zs.max()

        x_current_min = xs.min()
        y_current_min = ys.min()
        z_current_min = zs.min()

        if xs_max < x_current_max:
            xs_max = x_current_max
        if ys_max < y_current_max:
            ys_max = y_current_max
        if zs_max < z_current_max:
            zs_max = z_current_max

        if xs_min > x_current_min:
            xs_min = x_current_min
        if ys_min > y_current_min:
            ys_min = y_current_min
        if zs_min > z_current_min:
            zs_min = z_current_min

        if cont % 100 == 0:
            print(cont, len(dataset))
            # print(xs.shape)
        cont += 1

    print(xs_max, ys_max, zs_max)
    print(xs_min, ys_min, zs_min)

    return xs_max, ys_max, zs_max, xs_min, ys_min, zs_min


def merge_inputs(queries):

    # Get the keys that are present in all queries
    keys = [set([k for k in q.keys()]) for q in queries]
    keys = keys[0].intersection(*keys)

    # Keys that will be collated into single tensors, built as all possible combinations between input_types and
    # data_type_suffixes.
    input_types = ["anchor", "positive", "negative"]
    data_type_suffixes = ["pose", "rot", "idx"]
    tensor_collate_keys = [i + "_" + t for i in input_types for t in data_type_suffixes]
    # Add individual keys to be collated into single tensors
    tensor_collate_keys.extend(["sequence"])

    # Collate inputs from queries into single tensors (in tensor_collate_keys)
    tensor_collates = {k: default_collate([q[k] for q in queries]) for k in keys if k in tensor_collate_keys}
    # Collate inputs from queries into lists of tensors (NOT in tensor_collate_keys)
    list_collates = {k: [q[k] for q in queries] for k in keys if k not in tensor_collate_keys}
    # Combine the two dictionaries
    tensor_collates.update(list_collates)

    return tensor_collates


class Timer(object):
    """A simple timer."""

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.times = []
        self.call_incs = []
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.times = []
        self.call_incs = []

    @property
    def avg(self):
        if self.calls > 0:
            return self.total_time / self.calls
        else:
            return 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True, call_inc=1):
        diff = time.time() - self.start_time
        self.diff = diff
        self.times.append(diff)
        self.call_incs.append(call_inc)
        self.total_time += diff
        self.calls += call_inc
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff

    def save_json(self, path):
        stats = {
            "count": int(self.calls),
            "sum": float(self.total_time),
            "mean": float(self.avg),
            "laps": [float(t) for t in self.times],
            "call_increments": [int(c) for c in self.call_incs]
        }
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=4)

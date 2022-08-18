from argparse import ArgumentParser
from pathlib import Path
import pickle

import faiss
import numpy as np
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from datasets.haomo import HaomoLoader
from evaluation_comparison.inference_yaw_general_boreas import BatchSamplePairs
from models.get_models import load_model
from utils.data import merge_inputs, Timer
from utils.geometry import mat2xyzrpy
import utils.rotation_conversion as rot


def eval_model(gpu, weights_path, *,
               dataset_path,
               seq1=None,
               seq2=None,
               stride=1,
               num_iters=1,
               batch_size=2):

    torch.cuda.set_device(gpu)
    device = torch.device("cuda")

    seq1 = seq1 if seq1 is not None else 2
    seq2 = seq2 if seq2 is not None else 3

    dataset_for_recall1 = HaomoLoader(dataset_path, seq1, stride)
    dataset_for_recall3 = HaomoLoader(dataset_path, seq2, stride)

    concat_dataset = ConcatDataset([dataset_for_recall1, dataset_for_recall3])

    test_pair_idxs = []
    test_pair_idxs_concat = []
    index = faiss.IndexFlatL2(3)
    poses1 = np.stack(dataset_for_recall1.poses).copy()
    poses3 = np.stack(dataset_for_recall3.poses).copy()
    index.add(poses3[:, :3, 3].astype(np.float32).copy())
    num_frames_with_loop = 0
    num_frames_with_reverse_loop = 0

    print("Generating Test pairs")
    for i in tqdm(range(len(dataset_for_recall1.poses))):
        current_pose = poses1[i:i + 1, :3, 3].astype(np.float32).copy()
        lims, D, I = index.range_search(current_pose, 4. ** 2)

        if lims[1] > 0:
            j = 0
            if j == 0:
                num_frames_with_loop += 1
                yaw_diff = rot.npto_XYZRPY(np.linalg.inv(poses3[I[j]]) @ poses1[i])[-1]
                yaw_diff = yaw_diff % (2 * np.pi)
                if 0.79 <= yaw_diff <= 5.5:
                    num_frames_with_reverse_loop += 1

            test_pair_idxs.append([I[j], i])
            test_pair_idxs_concat.append([len(dataset_for_recall1) + I[j], i])
    test_pair_idxs = np.array(test_pair_idxs)
    test_pair_idxs_concat = np.array(test_pair_idxs_concat)

    batch_sampler = BatchSamplePairs(concat_dataset, test_pair_idxs_concat, batch_size)

    recall_loader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                                num_workers=2,
                                                batch_sampler=batch_sampler,
                                                collate_fn=merge_inputs,
                                                pin_memory=True)

    override_cfg = {"batch_size": batch_size}
    model, exp_cfg = load_model(weights_path, override_cfg)

    model = model.to(device)

    yaw_error = []

    time_net = Timer()

    # Testing
    if not(exp_cfg["weight_rot"] > 0. or exp_cfg["weight_transl"] > 0.):
        print("Nothing to do!")
        return

    current_frame = 0
    yaw_preds = torch.zeros((len(dataset_for_recall3.poses), len(dataset_for_recall1.poses)))
    transl_errors = []

    time_net.reset()
    print("Evaluating Test Pairs")
    for batch_idx, sample in enumerate(tqdm(recall_loader)):

        model.eval()
        with torch.no_grad():

            anchor_list = []
            for i in range(len(sample["anchor"])):
                anchor = torch.from_numpy(sample["anchor"][i])
                anchor = anchor.float().to(device)
                anchor_list.append(model.backbone.prepare_input(anchor))

            model_in = KittiDataset.collate_batch(anchor_list)
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(device)

            torch.cuda.synchronize()
            time_net.tic()
            batch_dict = model(model_in, metric_head=True)
            torch.cuda.synchronize()
            time_net.toc()
            pred_transl = []

            if not exp_cfg["rot_representation"].startswith("6dof"):
                raise NotImplementedError("No other transformation representation supported")

            transformation = batch_dict["transformation"]
            homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(
                transformation.device)
            transformation = torch.cat((transformation, homogeneous), dim=1)
            transformation = transformation.inverse()
            for i in range(batch_dict["batch_size"] // 2):
                yaw_preds[test_pair_idxs[current_frame + i, 0], test_pair_idxs[current_frame + i, 1]] = \
                mat2xyzrpy(transformation[i])[-1].item()
                pred_transl.append(transformation[i][:3, 3].detach().cpu())

            for i in range(batch_dict["batch_size"] // 2):
                pose1 = dataset_for_recall3.poses[test_pair_idxs[current_frame + i, 0]]
                pose2 = dataset_for_recall1.poses[test_pair_idxs[current_frame + i, 1]]
                delta_pose = np.linalg.inv(pose1) @ pose2
                transl_error = torch.tensor(delta_pose[:3, 3]) - pred_transl[i]
                transl_errors.append(transl_error.norm())

                yaw_pred = yaw_preds[test_pair_idxs[current_frame + i, 0], test_pair_idxs[current_frame + i, 1]]
                yaw_pred = yaw_pred % (2 * np.pi)
                delta_yaw = rot.npto_XYZRPY(delta_pose)[-1]
                delta_yaw = delta_yaw % (2 * np.pi)
                diff_yaw = abs(delta_yaw - yaw_pred)
                diff_yaw = diff_yaw % (2 * np.pi)
                diff_yaw = (diff_yaw * 180) / np.pi
                if diff_yaw > 180.:
                    diff_yaw = 360 - diff_yaw
                yaw_error.append(diff_yaw)

            current_frame += batch_dict["batch_size"] // 2

    transl_errors = np.array(transl_errors)
    yaw_error = np.array(yaw_error)

    valid = yaw_error <= 5.
    valid = valid & (np.array(transl_errors) <= 2.)
    succ_rate = valid.sum() / valid.shape[0]
    rte_suc = transl_errors[valid].mean()
    rre_suc = yaw_error[valid].mean()

    stats_dict = {
        "rot": yaw_error,
        "transl": transl_errors,
        "Mean rotation error": yaw_error.mean(),
        "Median rotation error": np.median(yaw_error),
        "STD rotation error": yaw_error.std(),
        "Mean translation error": transl_errors.mean(),
        "Median translation error": np.median(transl_errors),
        "STD translation error": transl_errors.std(),
        "Success Rate": succ_rate,
        "RTE": rte_suc,
        "RRE": rre_suc,
    }

    return stats_dict, time_net.avg


def main(weights_path, dataset_path, seq1=None, seq2=None, stride=1, num_iters=1, gpu=0, batch_size=15, save_path=None):
    stats, time = eval_model(gpu, weights_path, dataset_path=dataset_path, seq1=seq1, seq2=seq2,
                             stride=stride, num_iters=num_iters, batch_size=batch_size)

    print(f"Evaluation of model: {weights_path}")
    print(f"On sequences {seq1} and {seq2}")
    print(f"Average duration: {time}")
    print(f"Success Rate: {stats['Success Rate']}, RTE: {stats['RTE']}, RRE: {stats['RRE']}")

    if save_path:
        print("Saving stats to ", save_path)
        with open(save_path, "wb") as f:
            pickle.dump(stats, f)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--weights_path", type=Path, default="/home/arceyd/MT/dat/haomo/")
    parser.add_argument("--dataset_path", type=Path, default="/home/arceyd/MT/dat/haomo/sequences/",
                        help="Dataset directory")
    parser.add_argument("--seq1", type=int, default=2)
    parser.add_argument("--seq2", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

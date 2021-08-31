import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pickle
import os
import scipy.io as sio
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch

from datasets.KITTI_data_loader import KITTILoader3DPoses


def compute_PR(pair_dist, poses, map_tree_poses, is_distance=True, ignore_last=False):
    real_loop = []
    detected_loop = []
    distances = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1

    for i in tqdm(range(100, last)):
        min_range = max(0, i-50)  # Scan Context
        current_pose = poses[i][:3, 3]
        indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), 4)
        valid_idxs = list(set(indices[0]) - set(range(min_range, last)))
        if len(valid_idxs) > 0:
            real_loop.append(1)
        else:
            real_loop.append(0)

        if is_distance:
            candidate = pair_dist[i, :i-50].argmin()
            detected_loop.append(-pair_dist[i, candidate])
        else:
            candidate = pair_dist[i, :i-50].argmax()
            detected_loop.append(pair_dist[i, candidate])
        candidate_pose = poses[candidate][:3, 3]
        distances.append(np.linalg.norm(candidate_pose-current_pose))

    distances = np.array(distances)
    detected_loop = -np.array(detected_loop)
    real_loop = np.array(real_loop)
    precision_fn = []
    recall_fn = []
    for thr in np.unique(detected_loop):
        asd = detected_loop<=thr
        asd = asd & real_loop
        asd = asd & (distances <= 4)
        tp = asd.sum()
        fn = (detected_loop<=thr) & (distances > 4) & real_loop
        fn2 = (detected_loop > thr) & real_loop
        fn = fn.sum() + fn2.sum()
        fp = (detected_loop<=thr) & (distances > 4) & (1 - real_loop)
        fp = fp.sum()
        # fp = (detected_loop<=thr).sum() - tp
        # fn = (real_loop.sum()) - tp
        if (tp+fp) > 0:
            precision_fn.append(tp/(tp+fp))
        else:
            precision_fn.append(1.)
        recall_fn.append(tp/(tp+fn))
    precision_fp = []
    recall_fp = []
    for thr in np.unique(detected_loop):
        asd = detected_loop<=thr
        asd = asd & real_loop
        asd = asd & (distances <= 4)
        tp = asd.sum()
        fp = (detected_loop<=thr) & (distances > 4)
        fp = fp.sum()
        fn = (detected_loop > thr) & (real_loop)
        fn = fn.sum()
        if (tp+fp) > 0:
            precision_fp.append(tp/(tp+fp))
        else:
            precision_fp.append(1.)
        recall_fp.append(tp/(tp+fn))

    return precision_fn, recall_fn, precision_fp, recall_fp


def compute_PR_pairs(pair_dist, poses, is_distance=True, ignore_last=False, positive_range=4, ignore_below=-1):
    real_loop = []
    detected_loop = []
    last = poses.shape[0]
    if ignore_last:
        last = last-1
    for i in tqdm(range(100, last)):
        current_pose = poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= positive_range:
                real_loop.append(1)
            elif dist_pose <= ignore_below:
                continue
            else:
                real_loop.append(0)
            if is_distance:
                detected_loop.append(-pair_dist[i, j])
            else:
                detected_loop.append(pair_dist[i, j])
    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)

    return precision, recall


def compute_AP(precision, recall):
    ap = 0.
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i-1])*precision[i]
    return ap


if __name__ == '__main__':

    sequence = "08"
    dataset_path = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY'
    device = torch.device('cuda:0')

    marker = 'x'
    markevery = 0.03

    pd_sc = sio.loadmat(f'pairs_dist_sc_{sequence}.mat')['pairs_dist']
    # pd_sc = np.load(f'pairs_dist_sc_{sequence}.npz')['arr_0']
    # pd_m2 = sio.loadmat(f'pairs_dist_m2dp_{sequence}.mat')['pairs_dist']
    pd_isc = np.load(f'pairs_dist_isc_{sequence}.npz')['arr_0']
    pd_m2 = np.load(f'pairs_dist_m2dp_{sequence}.npz')['arr_0']
    pd_iris = np.load(f'pairs_dist_iris2_{sequence}.npz')['arr_0']
    # pd_iris2 = np.load(f'pairs_dist_iris2_{sequence}.npz')['arr_0']
    pd_ours = np.load(f'pairs_dist_ours_{sequence}.npz')['arr_0']
    pd_ours2 = np.load(f'pairs_dist_ours_{sequence}_trained360-02.npz')['arr_0']
    pd_over = np.load(f'/home/cattaneod/CODES/overlapnet_custom/overlap_pairs_{sequence}.npz')['arr_0']
    pd_sg = np.load(f'pairs_dist_SGPR_{sequence}_0320.npz')['arr_0']

    dataset_for_recall = KITTILoader3DPoses(dataset_path, sequence,
                                            os.path.join(dataset_path, 'sequences', sequence,'poses_SEMANTICKITTI.txt'),
                                            4096, device, train=False,
                                            without_ground=False, loop_file='loop_GT_4m')
    map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])
    poses = np.stack(dataset_for_recall.poses)

    precision_sc_fn, recall_sc_fn, precision_sc_fp, recall_sc_fp = compute_PR(pd_sc, poses, map_tree_poses)
    precision_isc_fn, recall_isc_fn, precision_isc_fp, recall_isc_fp = compute_PR(pd_isc, poses, map_tree_poses, False)
    precision_m2_fn, recall_m2_fn, precision_m2_fp, recall_m2_fp = compute_PR(pd_m2, poses, map_tree_poses)
    precision_iris_fn, recall_iris_fn, precision_iris_fp, recall_iris_fp = compute_PR(pd_iris, poses, map_tree_poses)
    precision_over_fn, recall_over_fn, precision_over_fp, recall_over_fp = compute_PR(pd_over, poses, map_tree_poses, False)
    precision_ours_fn, recall_ours_fn, precision_ours_fp, recall_ours_fp = compute_PR(pd_ours, poses, map_tree_poses)
    precision_ours2_fn, recall_ours2_fn, precision_ours2_fp, recall_ours2_fp = compute_PR(pd_ours2, poses, map_tree_poses)
    precision_sg_fn, recall_sg_fn, precision_sg_fp, recall_sg_fp = compute_PR(pd_sg, poses, map_tree_poses, False)

    plt.clf()
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    plt.plot(recall_m2_fp, precision_m2_fp, label='M2DP', marker=marker, markevery=markevery)
    plt.plot(recall_sc_fp, precision_sc_fp, label='Scan Context', marker=marker, markevery=markevery)
    plt.plot(recall_isc_fp, precision_isc_fp, label='ISC', marker=marker, markevery=markevery)
    plt.plot(recall_iris_fp, precision_iris_fp, label='Lidar-IRIS', marker=marker, markevery=markevery)
    plt.plot(recall_over_fp, precision_over_fp, label='OverlapNet', marker=marker, markevery=markevery)
    plt.plot(recall_ours_fp, precision_ours_fp, label='Ours (KITTI)', marker=marker, markevery=markevery)
    plt.plot(recall_ours2_fp, precision_ours2_fp, label='Ours (KITTI-360)', marker=marker, markevery=markevery)
    plt.plot(recall_sg_fp, precision_sg_fp, label='SG-PR', marker=marker, markevery=markevery)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall [%]")
    plt.ylabel("Precision [%]")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    # plt.show()
    fig.savefig(f'./results_for_paper/new/{sequence}_fp_nolegend.pdf', bbox_inches='tight', pad_inches=0)
    plt.legend(loc="lower left")
    fig.savefig(f'./results_for_paper/new/{sequence}_fp.pdf', bbox_inches='tight', pad_inches=0)

    ap_m2_fp = compute_AP(precision_m2_fp, recall_m2_fp)
    ap_m2_fn = compute_AP(precision_m2_fn, recall_m2_fn)
    ap_sc_fp = compute_AP(precision_sc_fp, recall_sc_fp)
    ap_sc_fn = compute_AP(precision_sc_fn, recall_sc_fn)
    ap_isc_fp = compute_AP(precision_isc_fp, recall_isc_fp)
    ap_isc_fn = compute_AP(precision_isc_fn, recall_isc_fn)
    ap_iris_fp = compute_AP(precision_iris_fp, recall_iris_fp)
    ap_iris_fn = compute_AP(precision_iris_fn, recall_iris_fn)
    ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
    ap_ours_fn = compute_AP(precision_ours_fn, recall_ours_fn)
    ap_ours2_fp = compute_AP(precision_ours2_fp, recall_ours2_fp)
    ap_ours2_fn = compute_AP(precision_ours2_fn, recall_ours2_fn)
    ap_over_fp = compute_AP(precision_over_fp, recall_over_fp)
    ap_over_fn = compute_AP(precision_over_fn, recall_over_fn)
    ap_sg_fp = compute_AP(precision_sg_fp, recall_sg_fp)
    ap_sg_fn = compute_AP(precision_sg_fn, recall_sg_fn)

    precision_pair_m2, recall_pair_m2 = compute_PR_pairs(pd_m2, poses)
    precision_pair_sc, recall_pair_sc = compute_PR_pairs(pd_sc, poses)
    precision_pair_isc, recall_pair_isc = compute_PR_pairs(pd_isc, poses, False)
    precision_pair_iris, recall_pair_iris = compute_PR_pairs(pd_iris, poses)
    precision_pair_ours, recall_pair_ours = compute_PR_pairs(pd_ours, poses)
    precision_pair_ours2, recall_pair_ours2 = compute_PR_pairs(pd_ours2, poses)
    precision_pair_over, recall_pair_over = compute_PR_pairs(pd_over, poses, False)
    precision_pair_sg, recall_pair_sg = compute_PR_pairs(pd_sg, poses, False)

    plt.clf()
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    plt.plot(recall_pair_m2, precision_pair_m2, label='M2DP', marker=marker, markevery=markevery)
    plt.plot(recall_pair_sc, precision_pair_sc, label='Scan Context', marker=marker, markevery=markevery)
    plt.plot(recall_pair_isc, precision_pair_isc, label='ISC', marker=marker, markevery=markevery)
    plt.plot(recall_pair_iris, precision_pair_iris, label='Lidar-IRIS', marker=marker, markevery=markevery)
    plt.plot(recall_pair_over, precision_pair_over, label='OverlapNet', marker=marker, markevery=markevery)
    plt.plot(recall_pair_ours, precision_pair_ours, label='Ours (KITTI)', marker=marker, markevery=markevery)
    plt.plot(recall_pair_ours2, precision_pair_ours2, label='Ours (KITTI-360)', marker=marker, markevery=markevery)
    plt.plot(recall_pair_sg, precision_pair_sg, label='SG-PR', marker=marker, markevery=markevery)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall [%]")
    plt.ylabel("Precision [%]")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
    # plt.show()
    fig.savefig(f'./results_for_paper/new/{sequence}_pairs_nolegend.pdf', bbox_inches='tight', pad_inches=0)
    plt.legend(loc="lower left")
    fig.savefig(f'./results_for_paper/new/{sequence}_pairs.pdf', bbox_inches='tight', pad_inches=0)

    precision_pair_m2 = [x for _,x in sorted(zip(recall_pair_m2, precision_pair_m2))]
    recall_pair_m2 = sorted(recall_pair_m2)
    precision_pair_sc = [x for _,x in sorted(zip(recall_pair_sc, precision_pair_sc))]
    recall_pair_sc = sorted(recall_pair_sc)
    precision_pair_isc = [x for _,x in sorted(zip(recall_pair_isc, precision_pair_isc))]
    recall_pair_isc = sorted(recall_pair_isc)
    precision_pair_iris = [x for _,x in sorted(zip(recall_pair_iris, precision_pair_iris))]
    recall_pair_iris = sorted(recall_pair_iris)
    precision_pair_ours = [x for _,x in sorted(zip(recall_pair_ours, precision_pair_ours))]
    recall_pair_ours = sorted(recall_pair_ours)
    precision_pair_ours2 = [x for _,x in sorted(zip(recall_pair_ours2, precision_pair_ours2))]
    recall_pair_ours2 = sorted(recall_pair_ours2)
    precision_pair_over = [x for _,x in sorted(zip(recall_pair_over, precision_pair_over))]
    recall_pair_over = sorted(recall_pair_over)
    precision_pair_sg = [x for _,x in sorted(zip(recall_pair_sg, precision_pair_sg))]
    recall_pair_sg = sorted(recall_pair_sg)

    ap_m2_pair = compute_AP(precision_pair_m2, recall_pair_m2)
    ap_sc_pair = compute_AP(precision_pair_sc, recall_pair_sc)
    ap_isc_pair = compute_AP(precision_pair_isc, recall_pair_isc)
    ap_iris_pair = compute_AP(precision_pair_iris, recall_pair_iris)
    ap_ours_pair = compute_AP(precision_pair_ours, recall_pair_ours)
    ap_ours2_pair = compute_AP(precision_pair_ours2, recall_pair_ours2)
    ap_over_pair = compute_AP(precision_pair_over, recall_pair_over)
    ap_sg_pair = compute_AP(precision_pair_sg, recall_pair_sg)

    with open(f'./results_for_paper/new/{sequence}.csv', 'w') as file:
        file.write(f'Approach, ap_fp, ap_fn, ap_pair\n')
        file.write(f'M2DP, {ap_m2_fp}, {ap_m2_fn}, {ap_m2_pair}\n')
        file.write(f'SC, {ap_sc_fp}, {ap_sc_fn}, {ap_sc_pair}\n')
        file.write(f'ISC, {ap_isc_fp}, {ap_isc_fn}, {ap_isc_pair}\n')
        file.write(f'IRIS, {ap_iris_fp}, {ap_iris_fn}, {ap_iris_pair}\n')
        file.write(f'OverlapNet, {ap_over_fp}, {ap_over_fn}, {ap_over_pair}\n')
        file.write(f'SG-PR, {ap_sg_fp}, {ap_sg_fn}, {ap_sg_pair}\n')
        file.write(f'Ours, {ap_ours_fp}, {ap_ours_fn}, {ap_ours_pair}\n')
        file.write(f'Ours (KITTI-360), {ap_ours2_fp}, {ap_ours2_fn}, {ap_ours2_pair}\n')

    print("Done!")

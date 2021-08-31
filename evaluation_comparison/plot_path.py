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
    f1s = []
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
        f1 = 2 * (precision_fp[-1]*recall_fp[-1]) / (precision_fp[-1]+recall_fp[-1])
        f1s.append(f1)

    thr = np.unique(detected_loop)[np.argmax(np.array(f1s))]
    asd = detected_loop<=thr
    asd = asd & real_loop
    tp = asd & (distances <= 4)
    # tp = asd.sum()
    fp = (detected_loop<=thr) & (distances > 4)
    # fp = fp.sum()
    fn = (detected_loop > thr) & (real_loop)
    # fn = fn.sum()

    return tp, fp, fn


if __name__ == '__main__':

    sequence = "00"
    dataset_path = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY'
    device = torch.device('cuda:0')

    marker = 'x'
    markevery = 0.03

    pd_ours = np.load(f'pairs_dist_ours_{sequence}.npz')['arr_0']
    pd_ours2 = np.load(f'pairs_dist_ours_{sequence}_trained360-02.npz')['arr_0']

    dataset_for_recall = KITTILoader3DPoses(dataset_path, sequence,
                                            os.path.join(dataset_path, 'sequences', sequence,'poses_SEMANTICKITTI.txt'),
                                            4096, device, train=False,
                                            without_ground=False, loop_file='loop_GT_4m')
    poses = np.stack(dataset_for_recall.poses)

    map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])
    poses = np.stack(dataset_for_recall.poses)

    tp, fp, fn = compute_PR(pd_ours, poses, map_tree_poses)
    fig = plt.figure()
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.axis("off")
    plt.plot(poses[100:, 0, 3], poses[100:, 1, 3], 'lightgray')
    plt.scatter(poses[np.nonzero(fp)[0]+100, 0, 3], poses[np.nonzero(fp)[0]+100, 1, 3], c=[(228/255, 26/255, 28/255)], s=15)
    plt.scatter(poses[np.nonzero(fn)[0]+100, 0, 3], poses[np.nonzero(fn)[0]+100, 1, 3], c='b', s=15)
    plt.scatter(poses[np.nonzero(tp)[0]+100, 0, 3], poses[np.nonzero(tp)[0]+100, 1, 3], c=[(77./255, 175./255, 74./255)], s=15)
    # plt.show()
    fig.savefig(f'./results_for_paper/new/{sequence}_path.pdf', bbox_inches='tight', pad_inches=0)


    tp, fp, fn = compute_PR(pd_ours2, poses, map_tree_poses)
    plt.clf()
    fig = plt.figure()
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.axis("off")
    plt.plot(poses[100:, 0, 3], poses[100:, 1, 3], 'lightgray')
    plt.scatter(poses[np.nonzero(fp)[0]+100, 0, 3], poses[np.nonzero(fp)[0]+100, 1, 3], c=[(228./255, 26./255, 28./255)], s=15)
    plt.scatter(poses[np.nonzero(fn)[0]+100, 0, 3], poses[np.nonzero(fn)[0]+100, 1, 3], c='b', s=15)
    plt.scatter(poses[np.nonzero(tp)[0]+100, 0, 3], poses[np.nonzero(tp)[0]+100, 1, 3], c=[(77./255, 175./255, 74./255)], s=15)
    # plt.show()
    fig.savefig(f'./results_for_paper/new/{sequence}_path_trained360.pdf', bbox_inches='tight', pad_inches=0)


    print("Done!")

import os

import numpy as np
import torch
import pickle
import scipy.io as sio
from tqdm import tqdm

import utils.rotation_conversion as RT
from datasets.Freiburg import FreiburgRegistrationDataset
from datasets.KITTI360Dataset import KITTI3603DPoses
from datasets.KITTI_data_loader import KITTILoader3DPoses

drive = 'freiburg'
if '360' in drive:
    path = f'results_for_paper/baseline_0.3_2013_05_28_drive_00{drive[-2:]}_sync.pickle'
else:
    path = f'results_for_paper/baseline_0.3_{drive}_inv.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

for method in ['p2p', 'p2pl', 'fpfh', 'fgr']:
    rte = stats[method]['transl'].mean()
    rre = stats[method]['rot'].mean()
    valid = stats[method]['rot'] <= 5.
    valid = valid & (np.array(stats[method]['transl']) <= 2.)
    succ_rate = valid.sum() / valid.shape[0]
    rte_suc = stats[method]['transl'][valid].mean()
    rre_suc = stats[method]['rot'][valid].mean()
    print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')


if '360' in drive:
    path = f'results_for_paper/rpmnet_2013_05_28_drive_00{drive[-2:]}_sync.pickle'
else:
    path = f'./results_for_paper/rpmnet_{drive}.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')


path = f'/home/cattaneo/CODES/D3Feat-cu10/results_{drive}_loop.npz'
stats = np.load(path)['stats']

stats = stats[~np.isnan(stats[:,2])]
stats[:,2] = stats[:,2] * 180. / np.pi
rte = stats[:,1].mean()
rre = stats[:,2].mean()
valid = stats[:,2] <= 5.
valid = valid & (np.array(stats[:,1]) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats[valid,1].mean()
rre_suc = stats[valid,2].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

path = f'/home/cattaneo/CODES/FCGF/results_{drive}_loop.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

stats['transl'] = stats['transl'][~np.isnan(stats['rot'])]
stats['rot'] = stats['rot'][~np.isnan(stats['rot'])]
# stats['rot'][np.isnan(stats['rot'])] = 0.
stats['rot'] = stats['rot'] * 180. / np.pi
rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

path = f'/home/cattaneo/CODES/DGR/results_{drive}_loop.npz'
stats = np.load(path)['stats']

rte = stats[:,1].mean()
rre = stats[:,2].mean()
valid = stats[:,2] <= 5.
valid = valid & (np.array(stats[:,1]) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats[valid,1].mean()
rre_suc = stats[valid,2].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    # path = f'results_for_paper/lcdnet_2013_05_28_drive_00{drive[-2:]}_sync.pickle'
    path = f'results_for_paper/lcdnet00+08_2013_05_28_drive_00{drive[-2:]}_sync.pickle'
else:
    # path = f'./results_for_paper/lcdnet_{drive}.pickle'
    path = f'./results_for_paper/lcdnet00+08_{drive}.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    # path = f'results_for_paper/lcdnet_2013_05_28_drive_00{drive[-2:]}_sync_ransac.pickle'
    path = f'results_for_paper/lcdnet00+08_2013_05_28_drive_00{drive[-2:]}_sync_ransac.pickle'
else:
    # path = f'./results_for_paper/lcdnet_{drive}_ransac.pickle'
    path = f'./results_for_paper/lcdnet00+08_{drive}_ransac.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    # path = f'results_for_paper/lcdnet_2013_05_28_drive_00{drive[-2:]}_sync_icp.pickle'
    path = f'results_for_paper/lcdnet00+08_2013_05_28_drive_00{drive[-2:]}_sync_icp.pickle'
else:
    # path = f'./results_for_paper/lcdnet_{drive}_icp.pickle'
    path = f'./results_for_paper/lcdnet00+08_{drive}_icp.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    path = f'results_for_paper/lcdnet++_2013_05_28_drive_00{drive[-2:]}_sync.pickle'
else:
    path = f'./results_for_paper/lcdnet++_{drive}.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    path = f'results_for_paper/lcdnet++_2013_05_28_drive_00{drive[-2:]}_sync_ransac.pickle'
else:
    path = f'./results_for_paper/lcdnet++_{drive}_ransac.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')

if '360' in drive:
    path = f'results_for_paper/lcdnet++_2013_05_28_drive_00{drive[-2:]}_sync_icp.pickle'
else:
    path = f'./results_for_paper/lcdnet++_{drive}_icp.pickle'
with open(path, 'rb') as f:
    stats = pickle.load(f)

rte = stats['transl'].mean()
rre = stats['rot'].mean()
valid = stats['rot'] <= 5.
valid = valid & (np.array(stats['transl']) <= 2.)
succ_rate = valid.sum() / valid.shape[0]
rte_suc = stats['transl'][valid].mean()
rre_suc = stats['rot'][valid].mean()
print(f'{succ_rate},{rte_suc},{rre_suc},{rte},{rre}')
print('')

if '360' in drive:
    dataset_for_recall = KITTI3603DPoses('/home/cattaneo/Datasets/KITTI-360', f'2013_05_28_drive_00{drive[-2:]}_sync',
                                         train=False,
                                         without_ground=False, loop_file='loop_GT_4m_noneg')
elif 'freiburg' in drive:
    dataset_for_recall = FreiburgRegistrationDataset('/home/cattaneo/Datasets/Freiburg', False, get_pc=False)
else:
    dataset_for_recall = KITTILoader3DPoses('/home/cattaneo/Datasets/KITTI', drive,
                                            os.path.join('/home/cattaneo/Datasets/KITTI/', 'sequences', drive,
                                                         'poses_SEMANTICKITTI.txt'),
                                            4096, 'cpu', train=False,
                                            without_ground=False, loop_file='loop_GT_4m')

if 'freiburg' not in drive:
    yaw_over = np.load(f'/home/cattaneo/overlapnet_custom/overlap_pairs_yaw_{drive}.npz')['arr_0']
    yaw_over = 180 - yaw_over
    yaw_over = yaw_over % 360

    yaw_error = []
    for i in tqdm(range(100, yaw_over.shape[0])):
        current_pose = dataset_for_recall.poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = dataset_for_recall.poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= 4:
                delta_pose = np.linalg.inv(dataset_for_recall.poses[i]) @ dataset_for_recall.poses[j]
                # delta_pose = np.linalg.inv(dataset_for_recall.poses[j]) @ dataset_for_recall.poses[i]
                delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                delta_yaw = (delta_yaw * 180.) / np.pi
                delta_yaw = delta_yaw % 360.
                diff_yaw = abs(delta_yaw - yaw_over[i, j])
                diff_yaw = diff_yaw % 360
                if diff_yaw > 180.:
                    diff_yaw = 360 - diff_yaw
                yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

    if '360' not in drive:
        yaw_pred = sio.loadmat(f'./pairs_dist_sc_{drive}_yaw.mat')['pairs_rot']
    else:
        yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pairs_dist_sc_{drive}_yaw.npz')['arr_0']
    yaw_pred = yaw_pred * (360 / 60)
    yaw_error = []
    for i in tqdm(range(100, yaw_pred.shape[0])):
        current_pose = dataset_for_recall.poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = dataset_for_recall.poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= 4:
                if '360' in drive or 'freiburg' in drive:
                    delta_pose = np.linalg.inv(dataset_for_recall.poses[i]) @ dataset_for_recall.poses[j]
                else:
                    delta_pose = np.linalg.inv(dataset_for_recall.poses[j]) @ dataset_for_recall.poses[i]
                delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                delta_yaw = (delta_yaw * 180.) / np.pi
                delta_yaw = delta_yaw % 360.
                diff_yaw = abs(delta_yaw - yaw_pred[i, j])
                diff_yaw = diff_yaw % 360
                if diff_yaw > 180.:
                    diff_yaw = 360 - diff_yaw
                yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')


    yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pairs_dist_isc_{drive}_yaw.npz')['arr_0']
    yaw_pred = yaw_pred * (360 / 60)
    yaw_error = []
    for i in tqdm(range(100, yaw_pred.shape[0])):
        current_pose = dataset_for_recall.poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = dataset_for_recall.poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= 4:
                delta_pose = np.linalg.inv(dataset_for_recall.poses[j]) @ dataset_for_recall.poses[i]
                delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                delta_yaw = (delta_yaw * 180.) / np.pi
                delta_yaw = delta_yaw % 360.
                diff_yaw = abs(delta_yaw - yaw_pred[i, j])
                diff_yaw = diff_yaw % 360
                if diff_yaw > 180.:
                    diff_yaw = 360 - diff_yaw
                yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')


    yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pairs_dist_iris2_{drive}_yaw.npz')['arr_0']
    yaw_error = []
    for i in tqdm(range(100, yaw_pred.shape[0])):
        current_pose = dataset_for_recall.poses[i][:3, 3]
        for j in range(i-50):
            candidate_pose = dataset_for_recall.poses[j][:3, 3]
            dist_pose = np.linalg.norm(candidate_pose-current_pose)
            if dist_pose <= 4:
                delta_pose = np.linalg.inv(dataset_for_recall.poses[j]) @ dataset_for_recall.poses[i]
                delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
                delta_yaw = (delta_yaw * 180.) / np.pi
                delta_yaw = delta_yaw % 360.
                diff_yaw = abs(delta_yaw - yaw_pred[i, j])
                diff_yaw = diff_yaw % 360
                if diff_yaw > 180.:
                    diff_yaw = 360 - diff_yaw
                yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

else:

    yaw_over = np.load(f'/home/cattaneo/overlapnet_custom/overlap_pairs_yaw_{drive}.npz')['arr_0']
    yaw_over = 180 - yaw_over
    yaw_over = yaw_over % 360

    yaw_error = []
    for sample in dataset_for_recall:
        delta_pose = sample['transformation']
        delta_pose = np.linalg.inv(delta_pose)
        delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
        delta_yaw = (delta_yaw * 180.) / np.pi
        delta_yaw = delta_yaw % 360.
        diff_yaw = abs(delta_yaw - yaw_over[sample['positive_id'], sample['anchor_id']])
        diff_yaw = diff_yaw % 360
        if diff_yaw > 180.:
            diff_yaw = 360 - diff_yaw
        yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

    yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pairs_dist_sc_{drive}_yaw.npz')['arr_0']
    yaw_pred = yaw_pred * (360 / 60)
    yaw_error = []
    for sample in dataset_for_recall:
        delta_pose = sample['transformation']
        delta_pose = np.linalg.inv(delta_pose)
        delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
        delta_yaw = (delta_yaw * 180.) / np.pi
        delta_yaw = delta_yaw % 360.
        diff_yaw = abs(delta_yaw - yaw_pred[sample['positive_id'], sample['anchor_id']])
        diff_yaw = diff_yaw % 360
        if diff_yaw > 180.:
            diff_yaw = 360 - diff_yaw
        yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

    yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pairs_dist_isc_{drive}_yaw.npz')['arr_0']
    yaw_pred = yaw_pred * (360 / 60)
    yaw_error = []
    for sample in dataset_for_recall:
        delta_pose = sample['transformation']
        # delta_pose = np.linalg.inv(delta_pose)
        delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
        delta_yaw = (delta_yaw * 180.) / np.pi
        delta_yaw = delta_yaw % 360.
        diff_yaw = abs(delta_yaw - yaw_pred[sample['positive_id'], sample['anchor_id']])
        diff_yaw = diff_yaw % 360
        if diff_yaw > 180.:
            diff_yaw = 360 - diff_yaw
        yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

    yaw_pred = np.load(f'/home/cattaneo/lcd-comparison/build/pair_dist_iris2_{drive}_bias.npy')
    yaw_pred = yaw_pred.reshape((25612,25612))
    yaw_error = []
    for sample in dataset_for_recall:
        delta_pose = sample['transformation']
        # delta_pose = np.linalg.inv(delta_pose)
        delta_yaw = RT.npto_XYZRPY(delta_pose)[-1]
        delta_yaw = (delta_yaw * 180.) / np.pi
        delta_yaw = delta_yaw % 360.
        diff_yaw = abs(delta_yaw - yaw_pred[sample['positive_id'], sample['anchor_id']])
        diff_yaw = diff_yaw % 360
        if diff_yaw > 180.:
            diff_yaw = 360 - diff_yaw
        yaw_error.append(diff_yaw)
    yaw_error = np.array(yaw_error)
    valid = yaw_error < 5.
    succ_rate = valid.sum() / valid.shape[0]
    rre_suc = yaw_error[valid].mean()
    print(f'{succ_rate},{rre_suc},{yaw_error.mean()}')

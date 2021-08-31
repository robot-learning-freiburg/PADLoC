import os
import numpy as np
import pykitti
import pickle
from sklearn.neighbors import KDTree

base_folder = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY'
sequence = 0

kitti = pykitti.odometry(base_folder, f'{sequence:02d}')
poses2_file = os.path.join(base_folder, 'sequences', f'{sequence:02d}', 'poses_SEMANTICKITTI.txt')
poses2 = []
cam0_to_velo = np.array(kitti.calib.T_cam0_velo)
with open(poses2_file, 'r') as f:
    for x in f:
        x = x.strip().split()
        x = [float(v) for v in x]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = np.array(x[0:4])
        pose[1, 0:4] = np.array(x[4:8])
        pose[2, 0:4] = np.array(x[8:12])
        pose[3, 3] = 1.0
        pose = np.linalg.inv(cam0_to_velo) @ (pose @ cam0_to_velo)
        poses2.append(pose)
poses2 = np.stack(poses2)
poses2[:, 2, 3] = 0.

subsample_idxs = [0]
prev_T = poses2[0, :3, 3]
split_idx = -1
for i in range(1, poses2.shape[0]):
    current_T = poses2[i, :3, 3]
    if np.linalg.norm(current_T - prev_T) > 3.:
        subsample_idxs.append(i)
        prev_T = current_T
        if i > 1700 and split_idx == -1:
            split_idx = len(subsample_idxs)-1
subsample_idxs = np.array(subsample_idxs)
kdtree = KDTree(poses2[subsample_idxs[:split_idx], :3, 3])

pairs = []
for i in range(split_idx, len(subsample_idxs)):
    current_T = poses2[subsample_idxs[i], :3, 3]
    matching_idx = kdtree.query_radius(np.expand_dims(current_T, 0), 1.5)[0]
    if len(matching_idx) > 0:
        pairs.append(np.array([subsample_idxs[i], subsample_idxs[matching_idx[0]]]))

pairs = np.stack(pairs)
oreos_dict = {}
oreos_dict['pairs'] = pairs
oreos_dict['subsample_idxs'] = subsample_idxs
oreos_dict['split_idx'] = split_idx
with open('oreos_pairs2.pickle', 'wb') as f:
    pickle.dump(oreos_dict, f)
print(pairs)
import time

import faiss
import torch
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset
import os
import numpy as np
import random
import pickle
import bisect
import open3d as o3d

from utils.rotation_conversion import euler2mat, npquat2mat


class M2DGR(Dataset):
	"""KITTI ODOMETRY DATASET"""

	def __init__(self, dir, sequence, without_ground=False, train=False):
		"""

		:param dataset: directory where dataset is located
		:param sequence: KITTI sequence
		:param poses: csv with data poses
		"""

		self.dir = dir
		self.sequence = sequence

		base2lidar = np.eye(4)
		base2lidar[:3, 3] = [-0.09825, 0.00582, 0.72673]
		poses = os.path.join(dir, f'street_{sequence}_aligned.txt')
		poses2 = []
		self.stamps = []
		with open(poses, 'r') as f:
			for x in f:
				x = x.strip().split(' ')
				self.stamps.append(int(x[0]))
				x = [float(v) for v in x[1:]]
				pose = torch.zeros((4, 4), dtype=torch.float64)
				rot_mat = npquat2mat([x[-1], x[-4], x[-3], x[-2]])  # Conversion from xyzw to wxyz, and then to rotation matrix
				pose = rot_mat
				pose[:3, 3] = x[:3]
				pose[0, 3] += 2853536
				pose[1, 3] -= 4667028
				pose[2, 3] -= 3268793
				# pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
				pose = pose @ base2lidar
				poses2.append(pose)
		lidar_files = os.listdir(os.path.join(dir, f'pcd_street{sequence}'))
		lidar_files = sorted(lidar_files)
		self.lidar_stamps = self.stamps

		self.poses = np.stack(poses2)
		self.train = train
		self.without_ground = without_ground

		# index = faiss.IndexFlatL2(3)
		# index.add(self.poses[:500, :3, 3].copy())
		# test_pair_idxs = []
		# for i in range(1000, self.poses.shape[0]):
		#     current_pose = self.poses[i:i+1, :3, 3].copy()
		#     index.add(self.poses[i-500:i-599, :3, 3].copy())
		#     lims, D, I = index.range_search(current_pose, 4.**2)
		#     for j in range(lims[0], lims[1]):
		#         test_pair_idxs.append([I[j], i])
		# self.test_pair_idxs = np.array(test_pair_idxs)


	def get_velo(self, idx, dir, sequence, without_ground, jitter=False):
		if without_ground:
			raise NotImplementedError()
		else:
			velo_path = os.path.join(dir, f'pcd_street{sequence}', f'{self.lidar_stamps[idx]}.pcd')
			scan = o3d.io.read_point_cloud(velo_path)
		scan = np.asarray(scan.points)
		homogeneous = np.zeros((scan.shape[0], 4), dtype=scan.dtype)
		homogeneous[:, :3] = scan
		homogeneous[:, -1] = 1.
		scan = homogeneous

		if jitter:
			noise = 0.01 * np.random.randn(scan.shape[0], scan.shape[1]).astype(np.float32)
			noise = np.clip(noise, -0.05, 0.05)
			scan = scan + noise

		return scan

	def __len__(self):
		return len(self.poses)

	def __getitem__(self, idx):

		anchor_pcd = torch.from_numpy(self.get_velo(idx, self.dir, self.sequence, self.without_ground, False))
		anchor_pose = self.poses[idx]

		# if self.train:
		#     x = self.poses[idx][0, 3]
		#     y = self.poses[idx][1, 3]
		#     z = self.poses[idx][2, 3]
		#
		#     anchor_pose = torch.tensor([x, y, z])
		#     possible_match_pose = torch.tensor([0., 0., 0.])
		#     negative_pose = torch.tensor([0., 0., 0.])
		#
		#     indices = list(range(len(self.poses)))
		#     cont = 0
		#     positive_idx = idx
		#     negative_idx = idx
		#     while cont < 2:
		#         i = random.choice(indices)
		#         possible_match_pose[0] = self.poses[idx][0, 3]
		#         possible_match_pose[1] = self.poses[idx][1, 3]
		#         possible_match_pose[2] = self.poses[idx][2, 3]
		#         distance = torch.norm(anchor_pose - possible_match_pose)
		#         if distance <= 5 and idx == positive_idx:
		#             positive_idx = i
		#             cont += 1
		#         elif distance > 25 and idx == negative_idx:  # 1.5 < dist < 2.5 -> unknown
		#             negative_idx = i
		#             cont += 1
		#     if self.use_panoptic or self.use_semantic:
		#         positive_pcd, positive_logits = get_velo_with_panoptic(positive_idx, self.dir, self.sequence,
		#                                                            self.use_semantic, self.use_panoptic, self.jitter)
		#         positive_pcd = torch.from_numpy(positive_pcd)
		#         positive_logits = torch.from_numpy(positive_logits)
		#
		#         negative_pcd, negative_logits = get_velo_with_panoptic(negative_idx, self.dir, self.sequence,
		#                                                                self.use_semantic, self.use_panoptic, self.jitter)
		#         negative_pcd = torch.from_numpy(negative_pcd)
		#         negative_logits = torch.from_numpy(negative_logits)
		#     else:
		#         positive_pcd = torch.from_numpy(get_velo(positive_idx, self.dir, self.sequence, self.without_ground, self.jitter))
		#         negative_pcd = torch.from_numpy(get_velo(negative_idx, self.dir, self.sequence, self.without_ground, self.jitter))
		#
		#     sample = {'anchor': anchor_pcd,
		#               'positive': positive_pcd,
		#               'negative': negative_pcd}
		#     if self.use_panoptic or self.use_semantic:
		#         sample['anchor_logits'] = anchor_logits
		#         sample['positive_logits'] = positive_logits
		#         sample['negative_logits'] = negative_logits
		# else:
		if True:
			sample = {'anchor': anchor_pcd, 'anchor_pose': anchor_pose}

		return sample


# if __name__ == '__main__':
#     dataset = MulRan('/data/MulRan', 'Riverside03')
#
#     pcd1 = o3d.geometry.PointCloud()
#     pcd2 = o3d.geometry.PointCloud()
#     sample1 = dataset.__getitem__(dataset.test_pair_idxs[0][0])
#     sample2 = dataset.__getitem__(dataset.test_pair_idxs[0][1])
#     pcd1.points = o3d.utility.Vector3dVector(sample1['anchor'][:, :3].numpy().copy())
#     pcd2.points = o3d.utility.Vector3dVector(sample2['anchor'][:, :3].numpy().copy())
#
#     global_pcd = o3d.geometry.PointCloud()
#     for i in range(100):
#         sample = dataset.__getitem__(i)
#         scan = sample['anchor'].numpy()
#         pose = sample['anchor_pose']
#         scan[:, -1] = 1.
#         scan = pose @ scan.T
#         local_pcd = o3d.geometry.PointCloud()
#         local_pcd.points = o3d.utility.Vector3dVector(scan.T[:, :3].copy())
#         global_pcd.points.extend(local_pcd.points)
#     # o3d.io.write_point_cloud('mulran.pcd', global_pcd)
#     o3d.visualization.draw_geometries([global_pcd])

if __name__ == '__main__':
	dataset1 = M2DGR('/home/cattaneo/Datasets/M2DGR', '02')
	dataset3 = M2DGR('/home/cattaneo/Datasets/M2DGR', '06')

	index = faiss.IndexFlatL2(3)
	index.add(dataset1.poses[:, :3, 3].astype(np.float32).copy())
	test_pair_idxs = []
	for i in range(dataset3.poses.shape[0]):
		current_pose = dataset3.poses[i:i + 1, :3, 3].astype(np.float32).copy()
		D, I = index.search(current_pose, 1)
		if D < 4**2:
			test_pair_idxs.append([I[0, 0], i])
		# lims, D, I = index.range_search(current_pose, 4. ** 2)
		# for j in range(lims[0], lims[1]):
		#     test_pair_idxs.append([I[j], i])
	test_pair_idxs = np.array(test_pair_idxs)

	pcd1 = o3d.geometry.PointCloud()
	pcd2 = o3d.geometry.PointCloud()
	# vis = o3d.visualization.Visualizer()
	# vis.create_window()
	# vis.add_geometry(pcd1)
	# vis.add_geometry(pcd2)

	# for i in range(test_pair_idxs.shape[0]):
	for i in range(1):
		# vis.remove_geometry(pcd1)
		# vis.remove_geometry(pcd2)
		sample1 = dataset1.__getitem__(test_pair_idxs[i*10][0])
		sample2 = dataset3.__getitem__(test_pair_idxs[i*10][1])

		scan = sample1['anchor'].double().numpy()
		pose = sample1['anchor_pose']
		scan[:, -1] = 1.
		scan = pose @ scan.T
		scan2 = sample2['anchor'].double().numpy()
		pose2 = sample2['anchor_pose']
		scan2[:, -1] = 1.
		scan2 = pose2 @ scan2.T

		pcd1.points = o3d.utility.Vector3dVector(scan.T[:, :3].copy())
		pcd2.points = o3d.utility.Vector3dVector(scan2.T[:, :3].copy())
		pcd1.paint_uniform_color(np.array([1., 0., 0.]))
		pcd2.paint_uniform_color(np.array([0., 1., 0.]))
		o3d.io.write_point_cloud("source.pcd", pcd1)
		o3d.io.write_point_cloud("target.pcd", pcd2)
		# vis.add_geometry(pcd1)
		# vis.add_geometry(pcd2)
		# vis.poll_events()
		# vis.update_renderer()
		# time.sleep(0.5)

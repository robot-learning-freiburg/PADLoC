import argparse
import pickle

import faiss
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from sklearn.neighbors import KDTree
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import tqdm

from datasets.FordCampus import FordCampusDataset
from models.get_models import load_model
from utils.data import merge_inputs
from utils.tools import set_seed
from models.backbone3D.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from evaluation_comparison.plot_PR_curve import compute_PR, compute_AP, compute_PR_pairs

torch.backends.cudnn.benchmark = True

EPOCH = 1


def prepare_input(model, samples, exp_cfg, device):
    anchor_list = []
    for point_cloud in samples:
        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor_set = furthest_point_sample(point_cloud[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
            a = anchor_set[0, :].long()
            anchor_i = point_cloud[a]
        else:
            anchor_i = point_cloud

        if exp_cfg['3D_net'] != 'PVRCNN':
            anchor_list.append(anchor_i[:, :3].unsqueeze(0))
        else:
            anchor_list.append(model.backbone.prepare_input(anchor_i))
            del anchor_i

    if exp_cfg['3D_net'] != 'PVRCNN':
        point_cloud = torch.cat(tuple(anchor_list), 0)
        model_in = point_cloud
        model_in = model_in / 100.
    else:
        model_in = KittiDataset.collate_batch(anchor_list)
        for key, val in model_in.items():
            if not isinstance(val, np.ndarray):
                continue
            model_in[key] = torch.from_numpy(val).float().to(device)
    return model_in


def geometric_verification(model, dataset, id_query, id_candidate, device, exp_cfg):
    model.eval()
    with torch.no_grad():
        anchor_list = []

        sample_query = dataset.__getitem__(id_query)
        sample_candidate = dataset.__getitem__(id_candidate)
        query_pc = sample_query['anchor'].to(device)
        candidate_pc = sample_candidate['anchor'].to(device)

        model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

        transformation = batch_dict['transformation']
        homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
        transformation = torch.cat((transformation, homogeneous), dim=1)
        # TEST
        # transformation[0, 0, 3] = 0
        # transformation[0, 1, 3] = 0
        # transformation[0, 2, 3] = 0

        query_intensity = query_pc[:, -1].clone()
        query_pc = query_pc.clone()
        query_pc[:, -1] = 1.
        transformed_query_pc = (transformation[0] @ query_pc.T).T
        transformed_query_pc[:, -1] = query_intensity

        model_in = prepare_input(model, [transformed_query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=False, compute_embeddings=True)

        emb = batch_dict['out_embedding']
        if exp_cfg['norm_embeddings']:
            emb = emb / emb.norm(dim=1, keepdim=True)

    return (emb[0] - emb[1]).norm().detach().cpu()


def geometric_verification2(model, dataset, id_query, id_candidate, device, exp_cfg):
    model.eval()
    with torch.no_grad():

        sample_query = dataset.__getitem__(id_query)
        sample_candidate = dataset.__getitem__(id_candidate)
        query_pc = sample_query['anchor'].to(device)
        candidate_pc = sample_candidate['anchor'].to(device)

        model_in = prepare_input(model, [query_pc, candidate_pc], exp_cfg, device)

        batch_dict = model(model_in, metric_head=True, compute_embeddings=False)

    return batch_dict['transport'].sum(-1).sum().detach().cpu()


class SamplePairs(Sampler):

    def __init__(self, data_source, pairs):
        super(SamplePairs, self).__init__(data_source)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return [self.pairs[i, 0] for i in range(len(self.pairs))]


class BatchSamplePairs(BatchSampler):

    def __init__(self, data_source, pairs, batch_size):
        # super(BatchSamplePairs, self).__init__(batch_size, True)
        self.data_source = data_source
        self.pairs = pairs
        self.batch_size = batch_size
        self.count = 0

    def __len__(self):
        return 2*len(self.pairs)

    def __iter__(self):
        self.count = 0
        while 2*self.count + self.batch_size < 2*len(self.pairs):
            current_batch = []
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(self.batch_size//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch
            self.count += self.batch_size//2
        if 2*self.count < 2*len(self.pairs):
            diff = 2*len(self.pairs)-2*self.count
            current_batch = []
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 0])
            for i in range(diff//2):
                current_batch.append(self.pairs[self.count+i, 1])
            yield current_batch


def main_process(gpu, weights_path, args):
    global EPOCH
    rank = gpu

    set_seed(args.seed)

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # saved_params = torch.load(weights_path, map_location='cpu')
    override_cfg = dict(
        batch_size=args.batch_size,
        test_sequence=args.seq,
    )

    model, exp_cfg = load_model(weights_path, override_cfg_dict=override_cfg, is_training=args.non_deterministic)

    # asd = torch.load('/home/cattaneod/rpmnet_08_4m_shared.tar', map_location='cpu')

    # exp_cfg = saved_params['config']
    # exp_cfg = saved_params['config']
    # exp_cfg['batch_size'] = args.batch_size

    # current_date = datetime.now()

    # exp_cfg['test_sequence'] = args.seq
    # sequences_training = ["00", "03", "04", "05", "06", "07", "08", "09"]  # compulsory data in sequence 10 missing
    # sequences_validation = [exp_cfg['test_sequence']]
    # sequences_training = set(sequences_training) - set(sequences_validation)
    # sequences_training = list(sequences_training)

    # if 'loop_file' not in exp_cfg:
    #     exp_cfg['loop_file'] = 'loop_GT'
    # if 'sinkhorn_type' not in exp_cfg:
    #     exp_cfg['sinkhorn_type'] = 'flot'
    # if 'shared_embeddings' not in exp_cfg:
    #     exp_cfg['shared_embeddings'] = False
    # if 'use_semantic' not in exp_cfg:
    #     exp_cfg['use_semantic'] = False
    # if 'use_panoptic' not in exp_cfg:
    #     exp_cfg['use_panoptic'] = False
    # if 'noneg' in exp_cfg['loop_file']:
    #     exp_cfg['loop_file'] = 'loop_GT_4m'
    # if 'head' not in exp_cfg:
    #     exp_cfg['head'] = 'SuperGlue'

    # dataset_for_recall = KITTILoader3DPoses(args.data, sequences_validation[0],
    #                                         os.path.join(args.data, 'sequences', sequences_validation[0],'poses_SEMANTICKITTI.txt'),
    #                                         exp_cfg['num_points'], device, train=False,
    #                                         without_ground=exp_cfg['without_ground'], loop_file=exp_cfg['loop_file'])
    dataset_for_recall = FordCampusDataset(args.data, seq=args.seq,
                                           without_ground=exp_cfg['without_ground'])

    # dataset_list_valid = [dataset_for_recall]

    # get_dataset3d_mean_std(training_dataset)

    # final_dest = ''

    # with open(f'/home/cattaneod/CODES/overlapnet_custom/GT/{exp_cfg["test_sequence"]}/GT2.pickle', 'rb') as f:
    #     gts = pickle.load(f)
    # pairwise_overlap = np.zeros((len(gts), len(gts)))
    # for i in range(len(gts)):
    #     pairwise_overlap[i, i:] = gts[i][i:, 4]
    # pairwise_yaw = np.zeros((len(gts), len(gts)))
    # for i in range(len(gts)):
    #     pairwise_yaw[i, i:] = gts[i][i:, 7]
    # # pairwise_yaw = (180 - pairwise_yaw) % 360
    # # pairwise_yaw = torch.tensor(pairwise_yaw)
    # pairwise_yaw = torch.tensor(pairwise_yaw * 180 / np.pi) % 360

    # with open('/home/cattaneod/CODES/deep_lcd/oreos_pairs2.pickle', 'rb') as f:
    #     oreos_dict = pickle.load(f)
    # test_pair_idxs = oreos_dict['pairs']
    # test_pair_idxs = test_pair_idxs[:, ::-1]
    # subsample_idxs = oreos_dict['subsample_idxs']
    # split_idx = oreos_dict['split_idx']

    MapLoader = torch.utils.data.DataLoader(dataset=dataset_for_recall,
                                            batch_size=args.batch_size,
                                            num_workers=2,
                                            shuffle=False,
                                            collate_fn=merge_inputs,
                                            pin_memory=True)

    # model = get_model(exp_cfg, is_training=False)
    # renamed_dict = OrderedDict()
    # for key in saved_params['state_dict']:
    #     if not key.startswith('module'):
    #         renamed_dict = saved_params['state_dict']
    #         break
    #     else:
    #         renamed_dict[key[7:]] = saved_params['state_dict'][key]
    #
    # res = model.load_state_dict(renamed_dict, strict=False)
    # if len(res[0]) > 0:
    #     print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")

    model.train()
    model = model.to(device)

    map_tree_poses = KDTree(np.stack(dataset_for_recall.poses)[:, :3, 3])

    local_iter = 0.
    transl_error_sum = 0
    yaw_error_sum = 0
    emb_list_map = []
    rot_errors = []
    transl_errors = []
    for i in range(args.num_iters):
        rot_errors.append([])
        transl_errors.append([])

    for batch_idx, sample in enumerate(tqdm(MapLoader)):

        model.eval()
        with torch.no_grad():

            anchor_list = []
            for i in range(len(sample['anchor'])):
                anchor = sample['anchor'][i].to(device)

                if exp_cfg['3D_net'] != 'PVRCNN':
                    anchor_set = furthest_point_sample(anchor[:, 0:3].unsqueeze(0).contiguous(), exp_cfg['num_points'])
                    a = anchor_set[0, :].long()
                    anchor_i = anchor[a]
                else:
                    anchor_i = anchor

                if exp_cfg['3D_net'] != 'PVRCNN':
                    anchor_list.append(anchor_i[:, :3].unsqueeze(0))
                else:
                    anchor_list.append(model.backbone.prepare_input(anchor_i))
                    del anchor_i

            if exp_cfg['3D_net'] != 'PVRCNN':
                anchor = torch.cat(anchor_list)
                model_in = anchor
                # Normalize between [-1, 1], more or less
                model_in = model_in / 100.
            else:
                model_in = KittiDataset.collate_batch(anchor_list)
                for key, val in model_in.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    model_in[key] = torch.from_numpy(val).float().to(device)

            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)

            emb = batch_dict['out_embedding']
            # if exp_cfg['norm_embeddings']:
            #     emb = emb / emb.norm(dim=1, keepdim=True)
            emb_list_map.append(emb)

    emb_list_map = torch.cat(emb_list_map).cpu().numpy()
    # map_tree = KDTree(emb_list_map)

    # Recall@k
    recall = np.zeros(10)
    total_frame = 0

    emb_list_map_norm = emb_list_map / np.linalg.norm(emb_list_map, axis=1, keepdims=True)
    pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)
    if args.pr_filename:
        print(f"Saving Pairwise Distances to {args.pr_filename}")
        np.savez(args.pr_filename, pair_dist)
    poses = np.stack(dataset_for_recall.poses)
    precision_ours_fn, recall_ours_fn, precision_ours_fp, recall_ours_fp = compute_PR(pair_dist, poses, map_tree_poses)
    ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
    ap_ours_fn = compute_AP(precision_ours_fn, recall_ours_fn)
    print(weights_path)
    print(exp_cfg['test_sequence'])
    print("AP FP: ", ap_ours_fp)
    print("AP FN: ", ap_ours_fn)
    precision_pair_ours, recall_pair_ours = compute_PR_pairs(pair_dist, poses)
    precision_pair_ours = [x for _, x in sorted(zip(recall_pair_ours, precision_pair_ours))]
    recall_pair_ours = sorted(recall_pair_ours)
    ap_ours_pair = compute_AP(precision_pair_ours, recall_pair_ours)
    print("AP Pairs: ", ap_ours_pair)
    if args.stats_filename:
        save_dict = {
            "AP_FP": ap_ours_fp,
            "AP_FN": ap_ours_fn,
            "AP_Pairs": ap_ours_pair
        }

        print(f"Saving Stats file to {args.stats_filename}.")
        with open(args.stats_filename, "wb") as f:
            pickle.dump(save_dict, f)
    # # FAISS
    # real_loop = []
    # detected_loop = []
    # distances = []
    #
    # index = faiss.IndexFlatL2(emb_list_map.shape[1])
    # # index = faiss.IndexIVFFlat(quantizer, emb_list_map.shape[1], 10, faiss.METRIC_L2)
    # # index.train(emb_list_map[:50])
    # index.add(emb_list_map[:1])
    #
    # for i in tqdm(range(1001, emb_list_map.shape[0])):
    #     min_range = max(0, i-1000)  # Scan Context
    #     current_pose = dataset_for_recall.poses[i][:3, 3]
    #
    #     indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), 4)
    #     valid_idxs = list(set(indices[0]) - set(range(min_range, emb_list_map.shape[0])))
    #     if len(valid_idxs) > 0:
    #         real_loop.append(1)
    #     else:
    #         real_loop.append(0)
    #
    #     index.add(emb_list_map[i-1000:i-999])
    #     nearest = index.search(emb_list_map[i:i+1], 1)
    #
    #     total_frame += 1
    #     detected_loop.append(-nearest[0][0][0])
    #     candidate_pose = dataset_for_recall.poses[nearest[1][0][0]][:3, 3]
    #     distances.append(np.linalg.norm(candidate_pose-current_pose))
    #
    # distances = np.array(distances)
    # detected_loop = -np.array(detected_loop)
    # real_loop = np.array(real_loop)
    # precision_real = []
    # recall_real = []
    # for thr in np.arange(detected_loop.min(), detected_loop.max()+0.02, 0.001):
    #     asd = detected_loop<thr
    #     asd = asd & real_loop
    #     asd = asd & (distances <= 4)
    #     tp = asd.sum()
    #     fp = (detected_loop<thr).sum() - tp
    #     fn = (real_loop.sum()) - tp
    #     if (tp+fp) > 0:
    #         precision_real.append(tp/(tp+fp))
    #     else:
    #         precision_real.append(1.)
    #     recall_real.append(tp/(tp+fn))
    #
    # plt.clf()
    # plt.plot(recall_real, precision_real, label='Ours Real')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
    #
    # detected_loop = []
    # distances = []
    #
    # index = faiss.IndexFlatL2(emb_list_map.shape[1])
    # index.add(emb_list_map_norm[:1])
    #
    # for i in tqdm(range(1001, emb_list_map.shape[0])):
    #     min_range = max(0, i-1000)  # Scan Context
    #     current_pose = dataset_for_recall.poses[i][:3, 3]
    #
    #     index.add(emb_list_map_norm[i-1000:i-999])
    #     nearest = index.search(emb_list_map_norm[i:i+1], 1)
    #
    #     detected_loop.append(-nearest[0][0][0])
    #     candidate_pose = dataset_for_recall.poses[nearest[1][0][0]][:3, 3]
    #     distances.append(np.linalg.norm(candidate_pose-current_pose))
    #
    # distances = np.array(distances)
    # detected_loop = -np.array(detected_loop)
    # precision_norm_real = []
    # recall_norm_real = []
    # for thr in np.arange(detected_loop.min(), detected_loop.max()+0.02, 0.001):
    #     asd = detected_loop<thr
    #     asd = asd & real_loop
    #     asd = asd & (distances <= 4)
    #     tp = asd.sum()
    #     fp = (detected_loop<thr).sum() - tp
    #     fn = (real_loop.sum()) - tp
    #     if (tp+fp) > 0:
    #         precision_norm_real.append(tp/(tp+fp))
    #     else:
    #         precision_norm_real.append(1.)
    #     recall_norm_real.append(tp/(tp+fn))
    #
    # plt.clf()
    # plt.plot(recall_real, precision_real, label='Ours Real')
    # plt.plot(recall_norm_real, precision_norm_real, label='Ours Norm Real')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
    #
    # # ALL PAIRS PR CURVE
    # real_loop = []
    # detected_loop = []
    # pair_dist = faiss.pairwise_distances(emb_list_map_norm, emb_list_map_norm)
    # for i in tqdm(range(1001, emb_list_map.shape[0])):
    #     current_pose = dataset_for_recall.poses[i][:3, 3]
    #     for j in range(i-1000):
    #         candidate_pose = dataset_for_recall.poses[j][:3, 3]
    #         dist_pose = np.linalg.norm(candidate_pose-current_pose)
    #         if dist_pose <= 4:
    #             real_loop.append(1)
    #         else:
    #             real_loop.append(0)
    #
    #         detected_loop.append(-pair_dist[i, j])
    # precision_pairs, recall_pairs, _ = precision_recall_curve(real_loop, detected_loop)
    #
    # plt.clf()
    # plt.plot(recall_real, precision_real, label='Ours Real')
    # plt.plot(recall_norm_real, precision_norm_real, label='Ours Norm Real')
    # plt.plot(recall_pairs, precision_pairs, label='Ours Pairs Norm')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
    #
    # real_loop = []
    # detected_loop = []
    # detected_loop_geometric = []
    # distances = []
    # index = faiss.IndexFlatL2(emb_list_map.shape[1])
    # index.add(emb_list_map_norm[:50])
    # for i in tqdm(range(100, emb_list_map.shape[0])):
    #     min_range = max(0, i-50)  # Scan Context
    #     current_pose = dataset_for_recall.poses[i][:3, 3]
    #
    #     indices = map_tree_poses.query_radius(np.expand_dims(current_pose, 0), 4)
    #     valid_idxs = list(set(indices[0]) - set(range(min_range, emb_list_map.shape[0])))
    #     if len(valid_idxs) > 0:
    #         real_loop.append(1)
    #     else:
    #         real_loop.append(0)
    #
    #     index.add(emb_list_map_norm[i-50:i-49])
    #     nearest = index.search(emb_list_map_norm[i:i+1], 1)
    #
    #     distance_after_verification = geometric_verification(model, dataset_for_recall, i, nearest[1][0][0], device, exp_cfg)
    #
    #     detected_loop.append(-nearest[0][0][0])
    #     candidate_pose = dataset_for_recall.poses[nearest[1][0][0]][:3, 3]
    #     distances.append(np.linalg.norm(candidate_pose-current_pose))
    #     detected_loop_geometric.append(-distance_after_verification)
    #
    # distances = np.array(distances)
    # detected_loop = -np.array(detected_loop_geometric)
    # real_loop = np.array(real_loop)
    # precision_real_geom = []
    # recall_real_geom = []
    # for thr in np.arange(detected_loop.min(), detected_loop.max()+0.02, 0.001):
    #     asd = detected_loop<thr
    #     asd = asd & real_loop
    #     asd = asd & (distances <= 4)
    #     tp = asd.sum()
    #     fp = (detected_loop<thr).sum() - tp
    #     fn = (real_loop.sum()) - tp
    #     if (tp+fp) > 0:
    #         precision_real_geom.append(tp/(tp+fp))
    #     else:
    #         precision_real_geom.append(1.)
    #     recall_real_geom.append(tp/(tp+fn))
    #
    # plt.clf()
    # plt.plot(recall_real, precision_real, label='Ours Real')
    # plt.plot(recall_norm_real, precision_norm_real, label='Ours Norm Real')
    # plt.plot(recall_pairs, precision_pairs, label='Ours Pairs Norm')
    # plt.plot(recall_real_geom, precision_real_geom, label='Ours Norm + Geom')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/arceyd/MT/dat/Ford Campus/',
                        help='dataset directory')
    parser.add_argument('--seq', type=str, default='1')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--weights_path', default='/home/arceyd/MT/cp/3D/')
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument("--pr_filename", type=str, default=None)
    parser.add_argument("--stats_filename", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--non_deterministic", action="store_true")

    args = parser.parse_args()

    main_process(0, args.weights_path, args)

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn

from evaluation_comparison.plot.plot_pan_pc import plot_pc_on_ax, iso_transform, transform_vertices
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from datasets.KITTI_data_loader import KITTILoader3DDictSingle, KITTILoader3DDictPairs
from models.get_models import get_model
from utils.geometry import get_rt_matrix, mat2xyzrpy
from utils.tools import set_seed


def load_model(cp_file, device):
    saved_params = torch.load(cp_file, map_location='cpu')
    config = saved_params["config"]

    model = get_model(config,
                      is_training=False
                      )

    model.load_state_dict(saved_params['state_dict'], strict=True)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.eval()
    model.to(device)

    return model


def load_dataset(kitti_path, seq,
                 loop_file="loop_GT_4m", **_):
    poses = kitti_path / "sequences" / seq / "poses.txt"
    dataset = KITTILoader3DDictSingle(kitti_path, sequence=seq,
                                      npoints=None, device=None,  # Mandatory arguments that are not used!
                                      poses=poses, loop_file=loop_file,
                                      use_semantid=True, use_panoptic=True, use_logits=False, filter_dynamic=False)

    return dataset


def move_to_sample(frame, sample_pfx, frame_pfx, device, sample=None, suffixes=None):
    if sample is None:
        sample = {}

    if suffixes is None:
        suffixes = ["idx", "", "pose", "rot", "semantic", "supersem", "panoptic", "instance"]

    for suffix in suffixes:
        frame_k = frame_pfx + ("_" if frame_pfx and suffix else "") + suffix
        sample_k = sample_pfx + ("_" if sample_pfx and suffix else "") + suffix
        frame_val = frame[frame_k]
        if isinstance(frame_val, torch.Tensor):
            frame_val = frame_val.to(device)

        sample[sample_k] = frame_val

    return sample


def get_augmented_transform(device,
                            tx=0., ty=0., tz=0.,
                            rr=0., rp=0., ry=0.,
                            angle_units="rad",
                            rot_params="xyz"
                            ):
    trans = torch.Tensor([tx, ty, tz]).to(device)
    rot = torch.Tensor([rr, rp, ry]).to(device)

    if angle_units != "rad":
        rot = rot * np.pi / 180

    return get_rt_matrix(trans, rot, rot_parmas=rot_params)


def augment_sample(device, sample, tf, key="anchor"):
    inv_tf = torch.linalg.inv(tf)

    pose = get_rt_matrix(sample[f"{key}_pose"].to(device), sample[f"{key}_rot"].to(device), rot_parmas="xyz")
    augmented_pose = pose @ tf

    augmented_tra_rot = mat2xyzrpy(augmented_pose)

    sample[f"{key}_pose"] = augmented_tra_rot[:3]
    sample[f"{key}_rot"] = augmented_tra_rot[3:]

    points = torch.ones_like(sample[key]).to(device)
    points[:, :3] = sample[key][:, :3]
    augmented_points = (inv_tf @ points.T).T
    augmented_points[:, 3] = sample[key][:, 3]
    sample[key] = augmented_points

    return sample


def get_sample(dataset, frames, device, augment_positive=None):
    # Hack for dealing with the indices of the KittiDataset class,
    # since they are not the ones from the dataset, but rather depend on the GT Loop file...
    # dataset_idx = {dataset.loop_gt[k]['idx']: k for k in range(len(dataset))}
    # frames_idx = [dataset_idx[frame] for frame in frames]

    # Get the data for the anchor and positive samples.
    # Not using the auto-chosen ones by the GT Loop file to allow an arbitrary selection of any two frames
    anc_frame = dataset[frames[0]]

    pos_frame = dataset[frames[1]]

    # We can generate the second sample by augmenting the second
    if augment_positive is not None:
        augmented_tf = get_augmented_transform(device, **augment_positive)
        pos_frame = augment_sample(device, pos_frame, augmented_tf, key="anchor")

    # Now recombine the two frames into a single sample dictionary
    sample = move_to_sample(frame=anc_frame, sample_pfx="", frame_pfx="",
                            suffixes=["sequence", "class_one_hot_map", "superclass_one_hot_map"], device=device)
    sample = move_to_sample(frame=anc_frame, sample_pfx="anchor", frame_pfx="anchor", sample=sample, device=device)
    sample = move_to_sample(frame=pos_frame, sample_pfx="positive", frame_pfx="anchor", sample=sample, device=device)

    # Delete the loaded frames to release memory just in case
    del anc_frame, pos_frame
    return sample


def compute_samples_transforms(samples):
    keys = ["anchor", "positive", "negative"]

    for k in keys:
        if k + "_pose" in samples and k + "_transform" not in samples:
            poses = samples[k + "_pose"]
            rotat = samples[k + "_rot"]
            samples[k + "_transform"] = [get_rt_matrix(p, r, rot_parmas="xyz") for p, r in zip(poses, rotat)]

    if "p2a_transform" not in samples:
        samples["p2a_transform"] = [torch.linalg.inv(anc) @ pos
                                    for anc, pos in zip(samples["anchor_transform"], samples["positive_transform"])]
        samples["a2p_transform"] = [torch.linalg.inv(p2a) for p2a in samples["p2a_transform"]]

    return samples


def samples_np_to_torch(samples, device):
    for k, v in samples.items():
        if isinstance(v, np.ndarray):
            samples[k] = torch.from_numpy(v).float()

        if isinstance(samples[k], torch.Tensor):
            samples[k] = samples[k].to(device)


def collate_samples(samples, model, device):
    if not isinstance(samples, list):
        samples = [samples]

    sample_keys = samples[0].keys()
    collated_sample = {k: [s[k] for s in samples] for k in sample_keys}

    preproc_keys = ["anchor", "positive", "negative"]

    model_in = []
    for k in preproc_keys:
        if k in collated_sample:
            preproc_sample = [model.backbone.prepare_input(subsample) for subsample in collated_sample[k]]
            model_in += preproc_sample
        # del collated_sample[k]

    model_in = KittiDataset.collate_batch(model_in)
    samples_np_to_torch(model_in, device)
    model_in.update(collated_sample)
    compute_samples_transforms(model_in)

    return model_in


def get_samples(dataset, frame_list, model, device, augment_positive=None):
    samples = [get_sample(dataset, frames, device, augment_positive=augment_positive) for frames in frame_list]
    model_in = collate_samples(samples, model, device)

    return model_in


def plot_matches(*, anc_pc, anc_samp_ids,
                 pos_pc, pos_samp_ids, p2a_transform,
                 pred_pos_pc, match_weights,
                 device, style,
                 anc_sem=None,
                 pos_sem=None,
                 image_path=None,
                 z_distance=100.):
    # Anchor Point Cloud's Transformation (only the isometric projection)
    iso_tf = iso_transform(device)

    # Positive Point Cloud's Transformation
    # Move to anchor reference frame
    pos_rel_matrix = p2a_transform
    # Translate in z
    pos_traz_matrix = get_rt_matrix(torch.Tensor([0., 0., -z_distance]).to(device), torch.zeros(3).to(device),
                                    rot_parmas="xyz")
    pos_fin_matrix = pos_traz_matrix @ pos_rel_matrix
    # Isometric Projection
    pos_fin_matrix = iso_tf @ pos_fin_matrix

    # Transform Point Clouds
    proj_anc_pc = transform_vertices(vertices=anc_pc[:, :3], transform_matrix=iso_tf,
                                     device=device).cpu().numpy()
    proj_pos_pc = transform_vertices(vertices=pos_pc[:, :3], transform_matrix=pos_fin_matrix,
                                     device=device).cpu().numpy()
    proj_pred_pos_pc = transform_vertices(vertices=pred_pos_pc[:, :3], transform_matrix=pos_fin_matrix,
                                          device=device).cpu().numpy()

    anc_samp_ids = anc_samp_ids.cpu().numpy()
    pos_samp_ids = pos_samp_ids.cpu().numpy()

    # Split point clouds into sampled and unsampled points
    proj_samp_anc_pc = proj_anc_pc[anc_samp_ids]
    proj_samp_pos_pc = proj_pos_pc[pos_samp_ids]
    proj_uns_anc_pc = np.delete(proj_anc_pc, anc_samp_ids, axis=0)
    proj_uns_pos_pc = np.delete(proj_pos_pc, pos_samp_ids, axis=0)

    fig = plt.figure(dpi=style.point_cloud_dpi, figsize=(4.1, 5.8))
    ax = fig.add_subplot()
    ax.axis("off")
    ax.set_aspect("equal")

    # Plot Anchor Point Cloud
    if anc_sem is not None:
        anc_sem = anc_sem.cpu().numpy().T.flatten().astype(int)
        plot_pc_on_ax(ax, proj_anc_pc[:, :2], semantic_labels=anc_sem, zorder=1)
    else:
        ax.scatter(proj_uns_anc_pc[:, 0], proj_uns_anc_pc[:, 1], zorder=1, **style.src_point_cloud)
        ax.scatter(proj_samp_anc_pc[:, 0], proj_samp_anc_pc[:, 1], zorder=2, **style.src_sampled_point_cloud)

    # Plot Matching Lines
    geometric_accuracy = anc_pc[anc_samp_ids][:, :3]
    pred_pos_pc_anc = transform_vertices(vertices=pred_pos_pc[:, :3], transform_matrix=p2a_transform, device=device)
    geometric_accuracy = geometric_accuracy - pred_pos_pc_anc
    geometric_accuracy = geometric_accuracy.square()
    geometric_accuracy = geometric_accuracy.sum(dim=1)
    geometric_accuracy = geometric_accuracy.sqrt().cpu().numpy()
    good_matches = geometric_accuracy < 5.

    match_segments = np.concatenate(
        [proj_samp_anc_pc[:, :2].reshape(-1, 1, 2), proj_pred_pos_pc[:, :2].reshape(-1, 1, 2)], axis=1)
    match_lines = LineCollection(match_segments[good_matches], **style.match_lines_styles(match_weights[good_matches],
                                                                                          np.ones(good_matches.sum(),
                                                                                                  dtype=bool)),
                                 zorder=11)
    ax.add_collection(match_lines)

    # Plot Positive Point Cloud
    if pos_sem is not None:
        pos_sem = pos_sem.cpu().numpy().T.flatten().astype(int)
        plot_pc_on_ax(ax, proj_pos_pc[:, :2], semantic_labels=pos_sem, zorder=21)
    else:
        ax.scatter(proj_uns_pos_pc[:, 0], proj_uns_pos_pc[:, 1], zorder=21, **style.tgt_point_cloud)
        ax.scatter(proj_samp_pos_pc[:, 0], proj_samp_pos_pc[:, 1], zorder=22, **style.tgt_point_cloud)
        ax.scatter(proj_pred_pos_pc[:, 0], proj_pred_pos_pc[:, 1], zorder=23, **style.tgt_predicted_point_cloud)

    fig.tight_layout(pad=0.05)
    if image_path is None:
        fig.show()
        return

    print(f"Saving figure {image_path}.")
    fig.savefig(image_path)
    plt.close()


def lerp(x0, y0, x1, y1, x):
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1

    return m * x + b


def plot_attn_pcs(matches, anc, anc_samp_ids, selected_anc_point_idx, pos, pos_samp_ids, img_path, ext="png"):
    fig1 = plt.figure(dpi=600, figsize=(10, 10))
    # fig, axs = plt.subplots(nrows=1, ncols=2, sharey="all")
    # ax1, ax2 = axs[0], axs[1]
    ax1 = fig1.add_subplot()
    # ax1.axis("off")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$\mathbf{P}^p$")
    ax1.set_ylabel(r"$\mathbf{P}^a$")
    ax1.set_title("Attention Matrix $\mathbf{M}$")
    # # Exagerate values with sigmoid
    # matches = (0.5 - matches) / 0.1
    # matches = 1. / (1. + torch.exp(matches))
    # # Exagerate values with supercircle
    # exp = 2.
    # matches = torch.abs(matches - 1.)
    # matches = 1. - torch.pow(matches, exp)
    # matches = torch.pow(matches, 1. / exp)
    np_matches = matches.detach().cpu().numpy()
    attn_img = ax1.imshow(np_matches, vmin=0., vmax=1., cmap='Blues', interpolation="none")

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)

    fig1.colorbar(attn_img, cax=cax1)

    fig1.savefig(f"{img_path}_attn.pdf", dpi=600)
    plt.close(fig1)
    # ax2.axis("off")
    # ax2.set_aspect("auto")
    # np_conf = conf_weights.view((-1, 1)).detach().cpu().numpy()
    # ax2.imshow(np_conf)
    #
    # fig.savefig("test_heatmap.pdf")

    anc_samp_mask = torch.zeros(anc.shape[0], dtype=torch.bool)
    anc_samp_mask[anc_samp_ids] = 1
    anc_unsamp_mask = torch.ones(anc.shape[0], dtype=torch.bool)
    anc_unsamp_mask[anc_samp_ids] = 0
    anc_unsamp = anc[anc_unsamp_mask]
    anc_samp = anc[anc_samp_mask]
    anc_main = anc_samp[selected_anc_point_idx].reshape((1, -1))

    pos_samp_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    pos_samp_mask[pos_samp_ids] = 1
    pos_unsamp_mask = torch.ones(pos.shape[0], dtype=torch.bool)
    pos_unsamp_mask[pos_samp_ids] = 0
    pos_samp = pos[pos_samp_mask]
    pos_unsamp = pos[pos_unsamp_mask]
    pos_weights = matches[selected_anc_point_idx]

    range_margin = 1.
    anc_max_range = torch.max(anc_samp, dim=0).values
    anc_min_range = torch.min(anc_samp, dim=0).values
    pos_max_range = torch.max(pos_samp, dim=0).values
    pos_min_range = torch.min(pos_samp, dim=0).values
    max_range = torch.maximum(anc_max_range, pos_max_range) * range_margin
    min_range = torch.minimum(anc_min_range, pos_min_range) * range_margin
    max_range = max_range.detach().cpu().numpy()
    min_range = min_range.detach().cpu().numpy()
    plot_range = np.stack([min_range, max_range])

    fig2 = plt.figure(dpi=600)
    ax2 = fig2.add_subplot()

    anc_unsamp = anc_unsamp.detach().cpu().numpy()
    anc_samp = anc_samp.detach().cpu().numpy()
    anc_main = anc_main.detach().cpu().numpy()

    default_style = dict(
        marker="o",
        lw=0,
    )

    unsamp_style = dict(
        c="#C0C0C0",
        s=0.25,
    )
    anc_samp_style = dict(
        c="#696960",
        s=0.5,
    )
    anc_main_style = dict(
        # c="#B22202",
        c="#F00000",
        s=4,
    )

    ax2.scatter(anc_unsamp[:, 0], anc_unsamp[:, 1], **default_style, **unsamp_style)
    ax2.scatter(anc_samp[:, 0], anc_samp[:, 1], **default_style, **anc_samp_style)
    ax2.scatter(anc_main[:, 0], anc_main[:, 1], **default_style, **anc_main_style)

    ax2.set_xlim(plot_range[:, 0])
    ax2.set_ylim(plot_range[:, 1])
    ax2.axis("off")
    ax2.set_aspect("equal")

    fig2.tight_layout(pad=0.05)
    fig2.savefig(f"{img_path}_anc.{ext}")
    plt.close(fig2)

    fig3 = plt.figure(dpi=600)
    ax3 = fig3.add_subplot()

    pos_unsamp = pos_unsamp.detach().cpu().numpy()
    pos_samp = pos_samp.detach().cpu().numpy()
    pos_weights = pos_weights.detach().cpu().numpy()

    min_size = anc_samp_style["s"]
    max_size = anc_main_style["s"]
    min_weight = pos_weights.min()
    max_weight = pos_weights.max()
    pos_sizes = lerp(min_weight, min_size, max_weight, max_size, pos_weights)

    min_color = np.array(to_rgb(anc_samp_style["c"]))
    max_color = np.array(to_rgb(anc_main_style["c"]))
    pos_colors = lerp(min_weight, min_color, max_weight, max_color, pos_weights.reshape(-1, 1))

    # Reorder so that the points with the highest weights are drawn last (on top)
    pos_order = np.argsort(pos_weights)
    pos_samp = pos_samp[pos_order]
    pos_sizes = pos_sizes[pos_order]
    pos_colors = pos_colors[pos_order]

    ax3.scatter(pos_unsamp[:, 0], pos_unsamp[:, 1], **default_style, **unsamp_style)
    ax3.scatter(pos_samp[:, 0], pos_samp[:, 1], c=pos_colors, s=pos_sizes, vmin=0, vmax=1, **default_style)

    ax3.set_xlim(plot_range[:, 0])
    ax3.set_ylim(plot_range[:, 1])
    ax3.axis("off")
    ax3.set_aspect("equal")

    fig3.tight_layout(pad=0.05)
    fig3.savefig(f"{img_path}_pos.{ext}")
    plt.close(fig3)


def infer_and_plot(*, dataset, device, model, frames, image_path, **_):
    samples = get_samples(dataset, frames, model, device)

    with torch.no_grad():
        model(samples, metric_head=True, compute_embeddings=True,
              compute_transl=False, compute_rotation=True, mode="pairs")

    batch_size = len(samples["anchor_idx"])

    for i in range(batch_size):
        anc_samp_ids = samples["keypoint_idxs"][i]
        pos_samp_ids = samples["keypoint_idxs"][batch_size + i]

        if "conf_weights" in samples:
            conf_weights = samples["conf_weights"][i]
        else:
            conf_weights = torch.ones(samples["transport"].shape[1], device=samples["transport"].device)

        matches = samples["transport"][i]

        ordered_weights = torch.argsort(conf_weights)
        match_quality = {
            "med": ordered_weights[ordered_weights.shape[0] // 2],
            "bst": ordered_weights[-1],
            "wst": ordered_weights[0],
            "fix": 36
        }

        anc = samples["anchor"][i][:, :2]  # Only take x,y
        p2a_tf = samples["p2a_transform"][i]
        pos = samples["positive"][i][:, :3]
        pos = transform_vertices(vertices=pos, transform_matrix=p2a_tf, device=pos.device)
        pos = pos[:, :2]  # Only take x,y

        for k, v in match_quality.items():
            img_path = f"{image_path}/attnplt_s{dataset.sequence}_f{frames[i][0]}_p{v}_f{frames[i][1]}_{k}"

            selected_anc_point_idx = v
            # selected_point_weight = conf_weights[selected_anc_point_idx]

            plot_attn_pcs(matches, anc, anc_samp_ids, selected_anc_point_idx,
                          pos, pos_samp_ids, img_path, ext="png")

        pass


def path_arg(value):
    return Path(value).expanduser()


def path_list_arg(value):
    return [path_arg(p) for p in value.split(",")]


def seq_arg(value):
    return f"{int(value):02d}"


def frames_arg(value):
    return [[int(frame) for frame in frames.split(",")] for frames in value.split(";")]


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--kitti_path", "-k", type=path_arg)
    parser.add_argument("--seq", "-s", type=seq_arg)
    parser.add_argument("--frames", "-i", type=frames_arg)

    parser.add_argument("--cp_path", "-c", type=path_arg)

    parser.add_argument("--gpu", "-g", type=int, default=0)

    parser.add_argument("--image_path", "-f", type=str)

    args = parser.parse_args()

    return vars(args)


def main(kitti_path, seq, cp_path, frames, image_path, gpu=0,
         seed=0):
    set_seed(seed)

    kitti_dataset = load_dataset(kitti_path=kitti_path, seq=seq)

    torch.cuda.set_device(gpu)
    cuda_device = torch.device(gpu)

    model = load_model(cp_file=cp_path, device=cuda_device)
    infer_and_plot(dataset=kitti_dataset, device=cuda_device, frames=frames, model=model,
                   image_path=image_path)


def main_all_pairs(kitti_path, seq, cp_path, image_path, loop_file="loop_GT_4m", gpu=0, seed=0, frames=None,
                   batch_size=8):

    set_seed(seed)

    poses_path = kitti_path / "sequences" / seq / "poses.txt"
    pairs_dataset = KITTILoader3DDictPairs(kitti_path, sequence=seq,
                                           npoints=None, device=None,  # Mandatory arguments that are not used!
                                           poses=poses_path, loop_file=loop_file,
                                           use_semantid=True, use_panoptic=True, use_logits=False, filter_dynamic=False)
    kitti_dataset = load_dataset(kitti_path=kitti_path, seq=seq)

    torch.cuda.set_device(gpu)
    cuda_device = torch.device(gpu)

    model = load_model(cp_file=cp_path, device=cuda_device)

    frames = []
    for f in pairs_dataset.loop_gt:
        f_anc = f["idx"]
        frames.extend([[f_anc, f_pos] for f_pos in f["positive_idxs"]])

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    # Re-run where we left off... TODO: remove
    batches = batches[263:]

    for batch in batches:
        infer_and_plot(dataset=kitti_dataset, device=cuda_device, model=model, frames=batch, image_path=image_path)


if __name__ == "__main__":
    # main_all_pairs(**parse_args())
    main(**parse_args())

from pathlib import Path
from models.get_models import load_model
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from utils.data import merge_inputs
from evaluation_comparison.inference_yaw_sensitivity_general import preprocess_sample, get_dataset,\
    generate_pairs, BatchSamplePairs

if __name__ == "__main__":

    root_path = Path("/home/arceyd/MT/cp/3D/")

    cp_paths = {
        "LCDNet": root_path / "16-09-2021_00-02-34/checkpoint_last_iter.tar",
        "DCP": root_path / "04-04-2022_18-34-14/checkpoint_last_iter.tar",
        "PADLOC": root_path / "27-05-2022_19-10-54/checkpoint_last_iter.tar"
    }

    dataset = "kitti"
    data = "/home/arceyd/MT/dat/kitti/dataset"
    sequence = "08"

    batch_size = 2

    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    override_cfg = dict(
        batch_size=batch_size,
        test_sequence=sequence
    )

    def _profile_table(label, func, *args, **kwargs):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True,
                     # use_cuda=True,  # deprecated. Add to activities instead
                     with_flops=True) as prof:
            with record_function(label):
                res = func(*args, **kwargs)
        print(prof.key_averages().table(top_level_events_only=True))

        return res


    for k, cp in cp_paths.items():

        def _load():
            tmp_model, tmp_exp_cfg = load_model(cp, override_cfg_dict=override_cfg, is_training=False)
            tmp_model = tmp_model.to(device)
            tmp_model.eval()
            return tmp_model, tmp_exp_cfg

        def _load_sample(tmp_model, tmp_exp_cfg):
            ds = get_dataset(dataset, data, sequence, device, tmp_exp_cfg)
            poses, test_pair_idxs = generate_pairs(ds, positive_distance=4)
            batch_sampler = BatchSamplePairs(ds, test_pair_idxs, batch_size)

            recall_loader = torch.utils.data.DataLoader(dataset=ds,
                                                        # batch_size=exp_cfg['batch_size'],
                                                        num_workers=2,
                                                        # sampler=sampler,
                                                        batch_sampler=batch_sampler,
                                                        # worker_init_fn=init_fn,
                                                        collate_fn=merge_inputs,
                                                        pin_memory=True)

            tmp_sample = next(iter(recall_loader))
            tmp_sample = preprocess_sample(tmp_sample, tmp_model, tmp_exp_cfg, device)
            return tmp_sample

        print(f"Model {k}")

        # model, exp_cfg = _profile_table("load_model", load)
        model, exp_cfg = _load()

        # This one just runs for ever... best to not profile
        # sample = _profile_table("load_data", load_sample, model, exp_cfg)
        sample = _load_sample(model, exp_cfg)

        output = _profile_table("model_inference", model, sample)

        print("\n"*3)
    pass

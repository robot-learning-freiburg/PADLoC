from pathlib import Path
from models.get_models import load_model
import torch
from ptflops import get_model_complexity_info

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

    for k, cp in cp_paths.items():
        model, exp_cfg = load_model(cp, override_cfg_dict=override_cfg, is_training=False)
        ds = get_dataset(dataset, data, sequence, device, exp_cfg)
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

        model = model.to(device)
        model.eval()

        sample = next(iter(recall_loader))
        sample = preprocess_sample(sample, model, exp_cfg, device)

        def _input_constructor(_):
            return dict(batch_dict=sample)

        # flops = FlopCountAnalysis(model, sample)
        macs, params = get_model_complexity_info(model, (1, 2, 3), input_constructor=_input_constructor,
                                                 as_strings=False, print_per_layer_stat=True, verbose=True)

        print(f"Model {k} uses {macs} MACs and has {params} parameters.")

    pass

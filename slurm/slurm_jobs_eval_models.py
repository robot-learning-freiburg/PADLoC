from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from pathlib import Path


Model = namedtuple("Model", ["dir", "label", "batch_size", "checkpoints"])
CheckPoint = namedtuple("Checkpoint", ["label", "file"])
Dataset = namedtuple("Dataset", ["label", "dir", "func", "sequences", "z_offset"])
EvalFunction = namedtuple("EvalFunction", ["label", "function"])


def write_slurm_job(jobs, partition="aisdlc_gpu-rtx2080", gpus=4, memory=50000, cores=8, cwd="~", job_name="",
                    job_path="", create_dirs=None):

    mkdir_job = ""
    if create_dirs is not None:
        mkdir_job = "\n".join([f"mkdir -p {create_dir}" for create_dir in create_dirs])

    for i, job in enumerate(jobs):
        if not job:
            continue

        txt = f"""#!/bin/bash
#SBATCH -p {partition}
#SBATCH --mem {memory}
#SBATCH -t 1-00:00
#SBATCH -c {cores}
#SBATCH --gres=gpu:{gpus}
#SBATCH -D {cwd}
#SBATCH -o log/%x.%N.%j.out
#SBATCH -e log/%x.%N.%j.err
#SBATCH -J {job_name}

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME"
echo "Using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /home/arceyd/miniconda3/etc/profile.d/conda.sh
conda activate padloc

export PYTHONPATH=/home/arceyd/padloc/src/LCDNet:$PYTHONPATH

{mkdir_job}

{job}

echo "DONE";
echo "Finished at $(date)";                 
"""

        job_file = f"{job_path}/{job_name}.txt"
        print(f"Saving job file {job_file}")
        with open(job_file, "w") as f:
            f.write(txt)


def to_cli_arg(key, value):
    arg = ""

    if key is not None:
        arg += f"--{key}"

    if isinstance(value, bool):
        return arg

    return f"{arg} {value}"


def job_cmd(cmd, **kwargs):

    cmd_cli_args = " ".join([to_cli_arg(k, v) for k, v in kwargs.items()])
    return f"python {cmd} {cmd_cli_args}"


def job_lcd_kitt(dataset_path, cp_path, seq, batch_size, stats_save_path, pairs_save_path, **_):
    cmd = "evaluation_comparison/inference_placerecognition_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="kitti",
        sequence=seq,
        loop_file="loop_GT_4m",
        batch_size=batch_size,
        save_path=pairs_save_path,
        stats_save_path=stats_save_path,
    )
    return job_cmd(cmd, **kwargs)


def job_reg_kitt(dataset_path, cp_path, seq, batch_size, stats_save_path, ransac, **_):
    cmd = "evaluation_comparison/inference_yaw_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="kitti",
        sequence=seq,
        loop_file="loop_GT_4m",
        batch_size=batch_size,
        save_path=stats_save_path,
        ransac=ransac,
    )
    return job_cmd(cmd, **kwargs)


def job_lcd_k360(dataset_path, cp_path, seq, batch_size, stats_save_path, pairs_save_path, **_):
    cmd = "evaluation_comparison/inference_placerecognition_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="kitti360",
        sequence=seq,
        loop_file="loop_GT_4m_noneg",
        batch_size=batch_size,
        save_path=pairs_save_path,
        stats_save_path=stats_save_path,
    )
    return job_cmd(cmd, **kwargs)


def job_reg_k360(dataset_path, cp_path, seq, batch_size, stats_save_path, **_):
    cmd = "evaluation_comparison/inference_yaw_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="kitti360",
        sequence=seq,
        loop_file="loop_GT_4m_noneg",
        batch_size=batch_size,
        save_path=stats_save_path,
    )
    return job_cmd(cmd, **kwargs)


def job_lcd_ford(dataset_path, cp_path, seq, batch_size, stats_save_path, pairs_save_path, **_):
    cmd = "evaluation_comparison/inference_placerecognition_ford.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        seq=seq,
        batch_size=batch_size,
        stats_filename=stats_save_path,
        pr_filename=pairs_save_path
    )
    return job_cmd(cmd, **kwargs)


def job_reg_ford(dataset_path, cp_path, seq, batch_size, stats_save_path, ransac, **_):
    cmd = "evaluation_comparison/inference_yaw_general_ford.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        seq=seq,
        batch_size=batch_size,
        save_path=stats_save_path,
        positive_distance=10.,
        ransac=ransac,
    )
    return job_cmd(cmd, **kwargs)


def job_lcd_frbg(dataset_path, cp_path, batch_size, stats_save_path, pairs_save_path, z_offset, **_):
    cmd = "evaluation_comparison/inference_placerecognition_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="freiburg",
        batch_size=batch_size,
        save_path=pairs_save_path,
        stats_save_path=stats_save_path,
        z_offset=z_offset,
    )
    return job_cmd(cmd, **kwargs)


def job_reg_frbg(dataset_path, cp_path, batch_size, stats_save_path, z_offset, ransac, **_):
    cmd = "evaluation_comparison/inference_yaw_general_freiburg.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="freiburg",
        batch_size=batch_size,
        save_path=stats_save_path,
        z_offset=z_offset,
        ransac=ransac,
    )
    return job_cmd(cmd, **kwargs)


def main(jobs_path, ransac=False):
    ws_path = Path("/work/dlclarge2/arceyd-padloc/")

    cp_path = ws_path / "cp"

    models = [
        # Trained on KITTI360
        # ==============================================================================================================
        # LCDNet
        # Model("10-08-2022_16-28-08", "lcdnet360", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar"),
        #     # CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     # CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        #     ]
        # ),

        # PADLoC
        # Model("16-08-2022_13-20-28", "padloc360", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar"),
        #     # CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     # CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        #     ]
        # ),

        # Trained on Semantic KITTI
        # ==============================================================================================================
        # LCDNet
        Model("16-09-2021_00-02-34", "lcdnet", 15, [
                CheckPoint("lastiter", "checkpoint_last_iter.tar"),
                # CheckPoint("e95", "checkpoint_95_recall_91.437.tar"),
                # CheckPoint("bestrot", "best_model_so_far_rot.tar"),
                # CheckPoint("bestauc", "best_model_so_far_auc.tar"),
            ]
        ),

        # PADLoC
        Model("27-05-2022_19-10-54", "padloc", 15, [
                CheckPoint("lastiter", "checkpoint_last_iter.tar"),
                # CheckPoint("bestrot", "best_model_so_far_rot.tar"),
                # CheckPoint("bestauc", "best_model_so_far_auc.tar"),
            ]
        ),

        # DCP
        Model("04-04-2022_18-34-14", "dcp", 8, [
                CheckPoint("lastiter", "checkpoint_last_iter.tar"),
                # CheckPoint("e95", "checkpoint_95_recall_91.437.tar"),
                # CheckPoint("bestrot", "best_model_so_far_rot.tar"),
                # CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        ]),

        # Ablation Weights
        # ------------------------------------------------------------
        # Model("03-08-2022_10-48-12", "uniform", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar")
        # ]),
        # Model("05-08-2022_15-54-52", "colsum", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar")
        # ]),
        # Model("07-08-2022_05-39-08", "shannon", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar")
        # ]),
        # Model("07-08-2022_07-37-46", "hill2", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar")
        # ]),
        # Model("09-08-2022_00-21-24", "hill4", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar")
        # ]),

        # Ablation Losses (Overleaf)
        # ------------------------------------------------------------
        # Model("28-07-2022_10-18-11", "L____L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar")
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("31-07-2022_22-27-38", "LS___L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar")
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("27-07-2022_10-01-40", "LSM__L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar")
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("29-07-2022_17-06-54", "LSMP_L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar")
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("27-05-2022_19-10-54", "LSMPRL", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar")
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),

        # Ablation Losses (Reordered)
        # ------------------------------------------------------------
        # Model("02-09-2022_11-06-53", "LR___L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar"),
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("02-09-2022_15-44-48", "LRS__L", 15, [
        #     CheckPoint("lastiter", "checkpoint_last_iter.tar"),
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
        # Model("02-09-2022_17-31-44", "LRSM_L", 15, [
        #     # CheckPoint("lastiter", "checkpoint_last_iter.tar"),
        #     CheckPoint("bestrot", "best_model_so_far_rot.tar"),
        #     CheckPoint("bestauc", "best_model_so_far_auc.tar"),
        # ]),
    ]

    dataset_path = ws_path / "dat"
    ds_kitt_path = str(dataset_path / "KITTI")
    # ds_k360_path = str(dataset_path / "KITTI360")
    ds_ford_path = str(dataset_path / "FordCampus")
    ds_frbg_path = str(dataset_path / "Freiburg")

    func_kitt = [
        EvalFunction("reg", job_reg_kitt),
        EvalFunction("lcd", job_lcd_kitt)
    ]
    # func_k360 = [
    #     EvalFunction("reg", job_reg_k360),
    #     EvalFunction("lcd", job_lcd_k360)
    # ]
    func_ford = [
        EvalFunction("reg", job_reg_ford),
        EvalFunction("lcd", job_lcd_ford)
    ]
    func_frbg = [
        EvalFunction("reg", job_reg_frbg),
        EvalFunction("lcd", job_lcd_frbg)
    ]

    datasets = [
        # Dataset("kitt", ds_kitt_path, func_kitt, ["00", "02", "05", "06", "07", "08", "09"], None),
        Dataset("kitt", ds_kitt_path, func_kitt, ["08"], 0),
        # Dataset("k360", ds_k360_path, func_k360, ["2013_05_28_drive_0002_sync", "2013_05_28_drive_0009_sync"], None),
        # Dataset("ford", ds_ford_path, func_ford, ["1", "2"], None),
        Dataset("ford", ds_ford_path, func_ford, ["1"], 0),
        Dataset("frbg", ds_frbg_path, func_frbg, [None], 0),
        # Dataset("frbg_zshift0", ds_frbg_path, func_frbg, [None], 0)
    ]

    partition = "aisdlc_gpu-rtx2080"

    src_path = str(Path("/home/arceyd/padloc/src/LCDNet/"))

    out_path = ws_path / "res" / "final_ransac"

    Path(jobs_path).mkdir(parents=True, exist_ok=True)
    ransac_str = "_ransac" if ransac else ""

    for model in models:
        model_time = datetime.strptime(model.dir, "%d-%m-%Y_%H-%M-%S").strftime("%y%m%d%H%M%S")
        model_label = f"{model.label}_{model_time}"

        for cp in model.checkpoints:
            for dataset in datasets:
                for func in dataset.func:
                    for seq in dataset.sequences:
                        seq_str = f"_seq_{seq}" if seq is not None else ""
                        job_name = f"eval_{func.label}_{model_label}{ransac_str}_{cp.label}_{dataset.label}{seq_str}"
                        model_path = str(cp_path / model.dir / cp.file)
                        save_path = out_path / model_label / cp.label / dataset.label / func.label
                        stats_save_path = str(save_path / f"{job_name}.pickle")
                        pairs_save_path = str(save_path / f"{job_name}.npz")

                        job_cmd_list = [
                            func.function(
                                dataset_path=dataset.dir, cp_path=model_path, seq=seq,
                                batch_size=model.batch_size,
                                stats_save_path=stats_save_path, pairs_save_path=pairs_save_path,
                                z_offset=dataset.z_offset,
                                ransac=ransac,
                            )
                        ]

                        write_slurm_job(job_cmd_list, partition=partition, cwd=src_path, job_name=job_name,
                                        job_path=jobs_path, gpus=1, create_dirs=[str(save_path)])


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--jobs_path", default="../tmp_slurm_jobs/eval_abl_loss_best/")

    parser.add_argument("--ransac", action="store_true")

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())

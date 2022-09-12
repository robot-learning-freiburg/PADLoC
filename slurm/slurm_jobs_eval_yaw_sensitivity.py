from datetime import datetime
from pathlib import Path

from slurm_jobs_eval_models import write_slurm_job, job_cmd, Model, CheckPoint, Dataset, EvalFunction


def job_yaw_sensitivity_kitt(dataset_path, cp_path, seq, batch_size, stats_save_path, pairs_save_path, **_):
    cmd = "evaluation_comparison/inference_yaw_sensitivity_general.py"
    kwargs = dict(
        data=dataset_path,
        weights_path=cp_path,
        dataset="kitti",
        sequence=seq,
        loop_file="loop_GT_4m",
        batch_size=batch_size,
        save_path=stats_save_path,
    )
    return job_cmd(cmd, **kwargs)


def main():

    ws_path = Path("/work/dlclarge2/arceyd-padloc/")

    cp_path = ws_path / "cp"

    models = [

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
            ]
        )


    ]

    dataset_path = ws_path / "dat"
    ds_kitt_path = str(dataset_path / "KITTI")
    ds_k360_path = str(dataset_path / "KITTI360")
    ds_ford_path = str(dataset_path / "FordCampus")
    ds_frbg_path = str(dataset_path / "Freiburg")

    func_kitt = [
        EvalFunction("reg", job_yaw_sensitivity_kitt),
    ]
    # func_k360 = [
    #     EvalFunction("reg", job_reg_k360),
    #     EvalFunction("lcd", job_lcd_k360)
    # ]
    # func_ford = [
    #     EvalFunction("reg", job_reg_ford),
    #     EvalFunction("lcd", job_lcd_ford)
    # ]
    # func_frbg = [
    #     EvalFunction("reg", job_reg_frbg),
    #     EvalFunction("lcd", job_lcd_frbg)
    # ]

    datasets = [
        # Dataset("kitt", ds_kitt_path, func_kitt, ["00", "02", "05", "06", "07", "08", "09"], None),
        Dataset("kitt", ds_kitt_path, func_kitt, ["08"], 0),
        # Dataset("k360", ds_k360_path, func_k360, ["2013_05_28_drive_0002_sync", "2013_05_28_drive_0009_sync"], None),
        # # Dataset("ford", ds_ford_path, func_ford, ["1", "2"], None),
        # Dataset("ford", ds_ford_path, func_ford, ["1"], 0),
        # Dataset("frbg", ds_frbg_path, func_frbg, [None], 0),
        # Dataset("frbg_zshift0", ds_frbg_path, func_frbg, [None], 0)
    ]

    partition = "aisdlc_gpu-rtx2080"

    src_path = str(Path("/home/arceyd/padloc/src/LCDNet/"))
    jobs_path = "../tmp_slurm_jobs/eval_yaw_sensitivity/"
    out_path = ws_path / "res" / "yaw_sensitivity"

    Path(jobs_path).mkdir(parents=True, exist_ok=True)

    for model in models:
        model_time = datetime.strptime(model.dir, "%d-%m-%Y_%H-%M-%S").strftime("%y%m%d%H%M%S")
        model_label = f"{model.label}_{model_time}"

        for cp in model.checkpoints:
            for dataset in datasets:
                for func in dataset.func:
                    for seq in dataset.sequences:
                        seq_str = f"_seq_{seq}" if seq is not None else ""
                        job_name = f"eval_{func.label}_{model_label}_{cp.label}_{dataset.label}{seq_str}"
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
                            )
                        ]

                        write_slurm_job(job_cmd_list, partition=partition, cwd=src_path, job_name=job_name,
                                        job_path=jobs_path, gpus=1, create_dirs=[str(save_path)])


if __name__ == "__main__":
    main()

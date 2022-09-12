# import os
# import sys
from argparse import ArgumentParser
from collections import namedtuple
from itertools import combinations
from pathlib import Path
import torch
# import torch.multiprocessing as mp

# from evaluation_comparison.inference_yaw_general_boreas import main_process as inference_yaw
# from evaluation_comparison.inference_placerecognition_mulran import main_process as inference_lcd

from evaluation_comparison.boreas_utils import pr_filename, yaw_stats_filename, lcd_stats_filename

SEQUENCES = [
    "2020-11-26-13-58",
    "2020-12-01-13-26",
    "2020-12-18-13-44",
    "2021-01-15-12-17",
    "2021-01-19-15-08",
    "2021-01-26-11-22",
    "2021-02-02-14-07",
    "2021-03-02-13-38",
    "2021-03-23-12-43",
    "2021-03-30-14-23",
    "2021-04-08-12-44",
    "2021-04-13-14-49",
    "2021-04-15-18-55",
    "2021-04-20-14-11",
    "2021-04-29-15-55",
    "2021-05-06-13-19",
    "2021-05-13-16-11",
    "2021-06-03-16-00",
    "2021-06-17-17-52",
    "2021-08-05-13-34",
    "2021-09-02-11-42",
    "2021-09-07-09-35",
    "2021-10-15-12-35",
    # "2021-10-22-11-36",  # Missing /applanix/lidar_poses.csv
    "2021-11-02-11-16",
    "2021-11-14-09-47",
    "2021-11-16-14-10",
    "2021-11-23-14-27",
]
SEQUENCE_PAIRS = combinations(SEQUENCES, 2)

Model = namedtuple("model", ["dir", "cp_reg", "cp_lcd", "label", "batch_size"])

MODELS = [
    Model(dir="16-09-2021_00-02-34", label="lcdnet", batch_size=20,
          cp_reg="best_model_so_far_rot.tar", cp_lcd="best_model_so_far_auc.tar"),
    Model(dir="27-05-2022_19-10-54", label="padloc", batch_size=20,
          cp_reg="best_model_so_far_rot.tar", cp_lcd="best_model_so_far_auc.tar")
]


# def multi_eval(func, output_naming_func, sequence_pairs,
# dataset_path, models, cp_path, save_path, gpus, batch_size=1):
# 	for seq1, seq2 in sequence_pairs:
# 		seq1_dir = f"boreas-{seq1}"
# 		seq2_dir = f"boreas-{seq2}"
#
# 		if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
# 			print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
# 			continue
#
# 		for model in models:
# 			output_file = output_naming_func(seq1, seq2, model.label, save_path)
#
# 			if output_file.exists():
# 				print(f"{func} already executed on pair {seq1} - {seq2}. Skipping.")
# 				continue
#
# 			inference_yaw(gpus[0], cp_path / model.dir / model.cp,
# 						  dataset_path=dataset_path, seq1=seq1_dir, seq2=seq2_dir,
# 						  save_path=yaw_stats_file, batch_size=batch_size)
#
# 			pr_file = pr_filename(seq1, seq2, model.label, save_path)
# 			if not pr_file.exists():
# 				os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[1:])
# 				mp.spawn(inference_lcd, nprocs=len(gpus) - 1, )


# def eval_reg(dataset_path, models, cp_path, save_path):
# 	for seq1, seq2 in SEQUENCE_PAIRS:
# 		seq1_dir = f"boreas-{seq1}"
# 		seq2_dir = f"boreas-{seq2}"
#
# 		if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
# 			print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
# 			continue
#
# 		for model in models:
# 			yaw_stats_file = yaw_stats_filename(seq1, seq2, model.label, save_path)
#
# 			if yaw_stats_file.exists():
# 				print(f"Registration Eval already executed on pair {seq1} - {seq2} (see {yaw_stats_file.stem}). Skipping.")
# 				continue
#
# 			stdout = sys.stdout
# 			null = open(os.devnull, "w")
# 			sys.stdout = null
# 			try:
# 				inference_yaw(0, cp_path / model.dir / model.cp,
# 							  dataset_path=dataset_path, seq1=seq1_dir, seq2=seq2_dir,
# 							  save_path=yaw_stats_file, batch_size=model.batch_size)
# 			except Exception as e:
# 				print(f"Exception when evaluating pair {seq1} - {seq2}. Skipping. \n{e}")
# 			sys.stdout = stdout


def write_slurmjob(jobs, partition="aisdlc_gpu-rtx2080", gpus=4, memory=50000, cores=8, cwd="~", jobname="",
                   jobpath=""):
    for i, (seqs, job) in enumerate(jobs):
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
#SBATCH -J {jobname}{i:03}_{seqs}

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source /etc/cuda_env
cuda11.7

source /home/cattaneo/anaconda3/etc/profile.d/conda.sh
conda activate lcdnet10

export PYTHONPATH=.:$PYTHONPATH

{job}

echo "DONE";
echo "Finished at $(date)";                 
"""

        jobfile = f"{jobpath}/{jobname}{i:03}_{seqs}.txt"
        print(f"Saving job file {jobfile}")
        with open(jobfile, "w") as f:
            f.write(txt)


def eval_cmd(dataset_path, models, cp_path, save_path, job_path, find_path=None):
    gpus = torch.cuda.device_count()

    if find_path is None:
        find_path = save_path

    jobs = []
    for seq1, seq2 in SEQUENCE_PAIRS:
        seq1_dir = f"boreas-{seq1}"
        seq2_dir = f"boreas-{seq2}"

        # if not seq_pair_downloaded(seq1_dir, seq2_dir, dataset_path):
        # 	print(f"Either seq {seq1} or {seq2} not found. Skipping pair.")
        # 	continue

        job = ""

        for model in models:
            check_yaw_stats_file = yaw_stats_filename(seq1, seq2, model.label, find_path / "reg_stats")

            weights_path_reg = Path(cp_path) / model.dir / model.cp_reg

            if not check_yaw_stats_file.exists():
                if job:
                    job += "\n"
                yaw_stats_file = yaw_stats_filename(seq1, seq2, model.label, save_path)
                job += f"python evaluation_comparison/inference_yaw_general_boreas.py" \
                       f" --dataset_path {dataset_path}" \
                       f" --weights_path {weights_path_reg}" \
                       f" --batch_size 10" \
                       f" --dataset boreas" \
                       f" --seq1 {seq1_dir} --seq2 {seq2_dir}" \
                       f" --save_path {yaw_stats_file}\n"

            check_pr_file = pr_filename(seq1, seq2, model.label, find_path / "pr_pairs")
            check_lcd_stats_file = lcd_stats_filename(seq1, seq2, model.label, find_path / "pr_stats")

            weights_path_lcd = Path(cp_path) / model.dir / model.cp_lcd

            if not (check_pr_file.exists() and check_lcd_stats_file.exists()):
                if job:
                    job += "\n"
                pr_file = pr_filename(seq1, seq2, model.label, save_path)
                lcd_stats_file = lcd_stats_filename(seq1, seq2, model.label, save_path)
                job += f"python evaluation_comparison/inference_placerecognition_mulran.py" \
                       f" --data {dataset_path}" \
                       f" --weights_path {weights_path_lcd}" \
                       f" --batch_size 10" \
                       f" --dataset boreas" \
                       f" --seq1 {seq1_dir} --seq2 {seq2_dir}" \
                       f" --gpu_count {gpus} --pr_filename {pr_file}" \
                       f" --stats_filename {lcd_stats_file}\n"

        jobs.append((f"{seq1}_{seq2}", job))

    write_slurmjob(jobs, gpus=1, memory=50000, cwd="/home/cattaneo/lcdnet_jab/src", jobname="eval_boreas",
                   jobpath=job_path)


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", type=Path,
                        default="/work/dlclarge2/cattaneo-Datasets/boreas2/")
    parser.add_argument("--find_path", type=Path,
                        default="/home/arceyd/Documents/Projects/PADLoC/res/boreas_comparison/")
    parser.add_argument("--save_path", type=Path,
                        default="/home/cattaneo/lcdnet_jab/res/boreas_comparison/")
    parser.add_argument("--cp_path", type=Path,
                        default="/home/cattaneo/lcdnet_jab/cp/")
    parser.add_argument("--job_path", type=Path,
                        default="/home/arceyd/Documents/Projects/PADLoC/res/boreas_comparison/slurm/jobfiles6/")

    args = parser.parse_args()

    return vars(args)


def main_process(dataset_path, save_path, cp_path, job_path, find_path):
    return eval_cmd(dataset_path, MODELS, cp_path, save_path, job_path, find_path)


if __name__ == "__main__":
    main_process(**cli_args())

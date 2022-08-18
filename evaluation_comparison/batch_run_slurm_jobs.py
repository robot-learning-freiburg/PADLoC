import os
from argparse import ArgumentParser


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("directory", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--ext", type=str, default="txt")
    parser.add_argument("--nice", type=int, default=0)

    args = parser.parse_args()

    return vars(args)


def main(directory, batch_size=100, batch=0, ext="txt", nice=0):
    jobs = sorted([f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and f[-3:] == ext)])

    job_batch = jobs
    if batch >= 0:
        batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

        job_batch = batches[batch]

    if nice:
        nice = f"--nice={nice}"

    for job in job_batch:
        print(f"sbatch {job}")
        os.system(f"sbatch {directory}/{job} {nice}")


if __name__ == "__main__":
    main(**cli_args())

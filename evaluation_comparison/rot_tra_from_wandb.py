from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import wandb


def csv_arg(value, delim=","):
    if value is None:
        return []
    value = value.split(delim)
    value = [v.strip() for v in value]
    return value


def path_arg(value):
    return Path(value).expanduser()


def cli_args():

    parser = ArgumentParser()

    parser.add_argument("-p", "--project", type=str, default="joseab10/deep_lcd")
    parser.add_argument("-r", "--runs", type=csv_arg)
    parser.add_argument("-l", "--labels", type=csv_arg, default=None)
    parser.add_argument("-c", "--curves", type=csv_arg, default="Rotation Mean Error, Translation Error")
    parser.add_argument("-k", "--curve_labels", type=csv_arg, default="rme,tme")

    parser.add_argument("-f", "--path", type=path_arg, default="./")
    parser.add_argument("-g", "--file_prefix", type=str, default="")
    parser.add_argument("-x", "--file_extension", type=str, default="csv")

    args = parser.parse_args()

    return vars(args)


def main(*, project, runs, curves, path=None, labels=None, curve_labels=None,
         sep=",", file_prefix="", file_extension="csv", **_):

    api = wandb.Api()
    # filters = [{"display_name": r} for r in runs]
    # wandb_runs = api.runs(path=project, filters={"$or": filters},
    #                 order=None)
    wandb_runs = [api.runs(path=project, filters={"display_name": run}, order=None)[0] for run in runs]

    if labels is None:
        labels = runs

    if curve_labels is None:
        curve_labels = curves

    curves_data = {}
    for curve, curve_label in zip(curves, curve_labels):
        curve_data = None
        for run, label in zip(wandb_runs, labels):
            data = run.history(keys=[curve])
            data = data.rename(columns={curve: label})

            if data["_step"][0] == 0:
                data["_step"] += 1

            if curve_data is None:
                curve_data = data
            else:
                curve_data = pd.merge(curve_data, data, left_on="_step", right_on="_step", how="outer")

        curve_data = curve_data.rename(columns={"_step": "Step"})
        curves_data[curve_label] = curve_data

        if path is not None:
            filename = path / f"{file_prefix}{curve_label}.{file_extension}"
            print(f"Saving data to file {filename}.")
            curve_data.to_csv(filename, sep=sep, index=False)

    return curves_data


if __name__ == "__main__":
    args = cli_args()
    main(**args)

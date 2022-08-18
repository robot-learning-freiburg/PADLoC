from argparse import ArgumentParser
from collections import namedtuple


Dataset = namedtuple("Dataset", ["file_pfx", "label", "sequences"])
Model = namedtuple("Model", ["file_pfx", "label"])

DATASETS = [
    Dataset(file_pfx="freiburg", label="Freiburg", sequences=[""]),
    Dataset(file_pfx="kitti", label="KITTI", sequences=["00", "08"]),
    Dataset(file_pfx="kitti360", label="KITTI-360",
            sequences=["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                       "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                       "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"])
]

MODELS = [
    Model(file_pfx="lcdnet", label="LCDNet"),
    Model(file_pfx="padloc", label="PADLoC"),
    Model(file_pfx="dcp", label="DCP")
]


def filename_yaw(model, dataset, seq, ext="pickle"):
    if seq:
        seq = f"_{seq}"
    return f"{model}_{dataset}{seq}.{ext}"


def cli_args():
    parser = ArgumentParser()

    parser.add_argument()

    args = parser.parse_args()

    return vars(args)


def main():
    pass


if __name__ == "__main__":
    main(**cli_args())

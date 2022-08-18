from datasets.KITTI360Dataset import KITTI3603DDictPairs
from tqdm import tqdm


def iterate_samples(dataset):
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]


if __name__ == "__main__":
    data_path = "/home/arceyd/MT/dat/kitti360"
    sequences = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                 "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync",
                 "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"]
    loop_file = "loop_GT_4m_noneg"

    for seq in sequences:
        print(f"Sequence {seq}")
        ds = KITTI3603DDictPairs(data_path, seq, loop_file=loop_file, train=False, use_panoptic=True)
        iterate_samples(ds)

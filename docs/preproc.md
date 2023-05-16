# Datasets

## Semantic KITTI
Download the Semantic KITTI dataset from the [official website](http://semantic-kitti.org/dataset.html#download).

### Pre-processing

Generate the ground truth loop closure files using the `generate_loop_GT_KITTI.py`

```shell
python data_process/generate_loop_GT_KITTI.py \
    --dataset_dir DATA_DIR \
    [--output_dir OUT_DIR] \
    [--sequences SEQ]
```

#### Arguments
| Argument                          | Description                                                                                                                                                               |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--dataset_dir__ ___DATA_PATH___ | Path to the Semantic KITTI dataset.                                                                                                                                       |
| __--output_dir__ ___OUT_DIR___    | Path where the ground truth files will be saved. If `None`, the files will be stored within the corresponding sequence directories of the KITTI dataset. Default: `None`. |
| __--sequences__ ___SEQ___         | Comma-separated list of the KITTI sequences to pre-process. Default: `00,03,04,05,06,07,08,09`.                                                                           |

## KITTI 360

From KITTI360`s [official website](https://www.cvlibs.net/datasets/kitti-360/download.php), download:
* Raw Velodyne Scans (119 GB)
* Calibrations (3 kB)
* Vehicle Poses (8.9 MB)

and extract into to the official directory structure.

### Pre-processing

Generate the ground truth loop closure files using the `generate_loop_GT_KITTI.py`

```shell
python data_process/generate_loop_GT_KITTI.py \
    --dataset_dir DATA_DIR \
    [--output_dir OUT_DIR] \
    [--sequences SEQ]
```

#### Arguments
| Argument                          | Description                                                                                                                                                                                                                                                                                                                                        |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--dataset_dir__ ___DATA_PATH___ | Path to the Semantic KITTI dataset.                                                                                                                                                                                                                                                                                                                |
| __--output_dir__ ___OUT_DIR___    | Path where the ground truth files will be saved. If `None`, the files will be stored within the corresponding sequence directories of the KITTI dataset. Default: `None`.                                                                                                                                                                          |
| __--sequences__ ___SEQ___         | Comma-separated list of the KITTI sequences to pre-process. Default: `"2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync","2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0006_sync","2013_05_28_drive_0007_sync", "2013_05_28_drive_0009_sync", "2013_05_28_drive_0010_sync"`. |

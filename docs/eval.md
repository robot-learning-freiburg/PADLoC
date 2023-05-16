# Evaluation

This document shows the details on how to evaluate a trained model on the tasks of Loop Closure Detection and Registration.

## Loop Closure Detection (Place Recognition)

The script in this section is used to evaluate the accuracy of the loop closure detection performance of a trained model on a given set of data.

This script can be executed in two ways:
* Run inference of a pre-trained model on a given dataset to generate the global descriptor vectors for every scan. Then a distance metric is computed for every pair of frame descriptors, generating a square matrix. This matrix can be saved to a file.
* Read the pairwise distances between all pairs of descriptors from a file, such as the one generated by running the inference, specifying whether the entries in the matrix represent a distance, or a similarity.

Then, it compares the closest neighbors in descriptor space with those in euclidean space and computes classification metrics.

### Usage

```shell
python evaluation_comparison/inference_placerecognition_general.py \
  --weights_path CP_PATH \
  [--data DATASET_PATH] \
  [--dataset DATASET] \
  [--loop_file LOOP_GT] \
  [--sequence SEQ] \
  [--batch_size BATCH] \
  [--pos_distance POS_DIST] \
  [--neg_frames NEG_FRAMES] \
  [--start_frame START_FRAME] \
  [--ignore_last] \
  [--save_path SAVE_PATH] \
  [--force_inference] \
  [--not_distance] \
  [--stats_save_path STATS_PATH] \
  [--save_times_path TIMES_PATH]
```

#### Arguments

##### Main Arguments
| Argument                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--weights_path__ ___CP_PATH___       | Path to the pre-trained model weights.                                                                                                                                                                                                                                                                                                                                                                                                    |
| __--data__ ___DATA_DIR___              | Path to the dataset directory. Default: `/data`                                                                                                                                                                                                                                                                                                                                                                                           |
| __--dataset__ ___DATASET___            | String representing the dataset on which to evaluate the model. Valid values: `kitti`, `kitti360`. Default: `kitti`.                                                                                                                                                                                                                                                                                                                      |
| __--loop_file__ ___LOOP_GT___          | Name of the GT Loop file for the selected dataset. Required if the model was trained in one dataset and is being evaluated on a different one.                                                                                                                                                                                                                                                                                            |
| __--sequence__ ___SEQ___               | String representing the sequence of the dataset to evaluate on. Default: `08` if dataset is `kitti`or `2013_05_28_drive_0002_sync` for `kitti360`.                                                                                                                                                                                                                                                                                        |
| __--batch_size__ ___BATCH___           | Number of loop pairs to perform inference on every time. Default: `6`.                                                                                                                                                                                                                                                                                                                                                                    |

##### Loop Closure GT Conditions
| Argument                            | Description                                                                                                                                             |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--pos_distance__ ___POS_DIST___   | Distance in meters below which two scans are considered to form a loop closure. Default: `4.`.                                                          |
| __--neg_frames__ ___NEG_FRAMES___   | Minimum number of frames between to scans for them to be considered a loop closure. Used for filtering out consecutive scans (odometry). Default: `50`. |
| __--start_frame__ ___START_FRAME___ | Start considering loop closures after this number of frames. Default: `100`.                                                                            |
| __--ignore_last__                   | The last frame will not be considered as a loop closure candidate. Default: `False`                                                                     |

##### Pairwise distances
| Argument                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--save_path__ ___SAVE_PATH___ | Path to the numpy pairwise distance file. If the file doesn't exist, or the `--force_inference` flag is set, the script will run inference, compute the pairwise distance between the descriptors of all pairs of scans and save them to this path. If the file exists and `--force_inference` is not set, no inference will be run and the distances will be read from this file. If None, inference will be performed, but the pairwise distances will not be saved. Useful for avoiding to run the costly inference every time. |
| __--force_inference__           | If set, the script will perform inference and compute the pairwise distances between frames. If not set, and the `--save_path` argument was set to an existing pairwise distance file, the distances from said file will be used and no inference will be run.                                                                                                                                                                                                                                                                     |
| __--not_distance__              | If reading the pairwise metrics from a file using the `--save_path` argument, and the values in that matrix are not a distance but rather a similarity measure, this flag must be used. Only applicable for some of the baselines.                                                                                                                                                                                                                                                                                                 |

##### Additional Output
| Argument                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--stats_save_path__ ___STATS_PATH___ | Path to the JSON file where the evaluation statistics and metrics will be saved. If not provided, the results will only be printed to the standard output.                                                                                                                                                                                                                                                                                |
| __--save_times_path__ ___TIMES_PATH___ | Path to the JSON file where the timing statistics will be saved. If not provided, no benchmarking will be saved.                                                                                                                                                                                                                                                                                                                          |

## Registration

This script is used for evaluating a pre-trained model on the registration task, that is, estimating the relative 6DOF transformation between two scans.

The final transformation can be obtained via:
* Prediction of the registration head of the model.
* Matching points via RANSAC over the features extracted by the model's backbone.
* Transformation computed via ICP, initialized with either of the previous registrations.

### Usage

```shell
python evaluation_comparison/inference_yaw_general.py \
  --weights_path CP_PATH \
  [--data DATASET_PATH] \
  [--dataset DATASET] \
  [--loop_file LOOP_GT] \
  [--sequence SEQ] \
  [--batch_size BATCH] \
  [--ransac] \
  [--icp] \
  [--save_path SAVE_PATH] \
  [--save_times_path TIMES_PATH]
```

#### Arguments

##### Main Arguments
| Argument                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--weights_path__ ___CP_PATH___       | Path to the pre-trained model weights.                                                                                                                                                                                                                                                                                                                                                                                                    |
| __--data__ ___DATA_DIR___              | Path to the dataset directory. Default: `/data`                                                                                                                                                                                                                                                                                                                                                                                           |
| __--dataset__ ___DATASET___            | String representing the dataset on which to evaluate the model. Valid values: `kitti`, `kitti360`. Default: `kitti`.                                                                                                                                                                                                                                                                                                                      |
| __--loop_file__ ___LOOP_GT___          | Name of the GT Loop file for the selected dataset. Required if the model was trained in one dataset and is being evaluated on a different one.                                                                                                                                                                                                                                                                                            |
| __--sequence__ ___SEQ___               | String representing the sequence of the dataset to evaluate on. Default: `08` if dataset is `kitti`or `2013_05_28_drive_0002_sync` for `kitti360`.                                                                                                                                                                                                                                                                                        |
| __--batch_size__ ___BATCH___           | Number of loop pairs to perform inference on every time. Default: `6`.                                                                                                                                                                                                                                                                                                                                                                    |

##### Registration Method
| Argument                      | Description                                                                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| __--ransac__                  | If set, RANSAC will be performed using the extracted features to register the scans, instead of the transformation predicted by the model. |
| __--icp__                     | If set, the transformation predicted either by the model or RANSAC will be refined using ICP.                                              |

##### Output
| Argument                               | Description                                                                                                      |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------|
| __--save_path__ ___SAVE_PATH___        | Path to the JSON file where the evaluation statistics will be saved. If not provided, no file will be saved.     |
| __--save_times_path__ ___TIMES_PATH___ | Path to the JSON file where the timing statistics will be saved. If not provided, no benchmarking will be saved. |
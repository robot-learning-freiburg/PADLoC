# Training

This document shows the details on how to train a model for loop-closure detection and registration.

## Example usage

In order to reproduce the results from our paper, that is, to train a model with the PADLoC architecture on the KITTI dataset located at `/data` for 100 epochs, storing the model checkpoints in `/cp`, simply run the following command. 

```shell
python training2D3D_hard_KITTI_DDP.py 
```

## Usage

```shell
python training2D3D_hard_KITTI_DDP.py \
    [--data DATA_PATH] \
    [--dataset DATASET] \
    [--config CONFIG_PATH] \
    [--epochs EPOCHS] \
    [--checkpoints_dest CP_PATH] \
    [--gpu GPU] \
    [--gpu_count GPU_COUNT] \
    [--port PORT] \
    [--weights WEIGHTS_PATH] \
    [--resume] \
    [--strict_weight_load] \
    [--freeze_loaded_weights] \
    [--freeze_weights_containing STR] \
    [--unfreeze_weights_containing STR] \
    [--wandb] \
    [--print_iteration ITER] \ 
    [--print_other_losses] \
    [ [CFG_KEY_1:VAL_1] ... [CFG_KEY_N:VAL_N] ]
```

### Arguments

#### Main arguments
| Argument                                               | Description                                                                                                                                                                                                                                 |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __--data__ ___DATA_PATH___                             | Path to the dataset. Default: `/data`                                                                                                                                                                                                       |
| __--dataset__ ___DATASET___                            | String representing the name of the dataset used for training. Default: `"kitti"`. Valid values: `"kitti"`, `"kitti360"`.                                                                                                                   |
| __--config__ ___CONFIG_PATH___                         | Path to the `.yaml` configuration file used for the model and training. Default: `wandb_config_padloc.yaml`. See [Model Configuration](#model-configuration) and [Overriding the model configuration](#overriding-the-model-configuration). |
| __--epochs__ ___EPOCHS___                              | Number of epochs to run the training. Default: `100`.                                                                                                                                                                                       |
| __--checkpoints_dest__ ___CP_PATH___                   | Path to the directory where the model checkpoints will be saved after every epoch. Default: `/cp`.                                                                                                                                          |

#### Distributed Training

| Argument                             | Description                                                                                                |
|--------------------------------------|------------------------------------------------------------------------------------------------------------|
| __--gpu__ ___GPU___                  | When running on a single GPU, use this argument to specify which.                                          |
| __--gpu_count__ ___GPU_COUNT___      | Number of GPUs to use during training. If set to `-1`, all the available GPUs will be used. Default: `-1`. |
| __--port__ ___PORT___                | TCP port used by the Torch Distributed Master. It must be a free port. Default: `8888`.                    |

#### Initializing and Resuming

| Argument                                    | Description                                                                                                                                   |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| __--weights__ ___WEIGHTS_PATH___            | Path to a pre-trained model checkpoint. Default: `None`. See [Initializing with pre-trained weights](#initializing-with-pre-trained-weights). |
| __--resume__                                | If set, a previously unfinished training will be resumed where it left off. See [Resuming a training](#resuming-a-training).                  |
| __--strict_weight_load__                    | If set, only checkpoint files entirely matching the model configuration will be loaded.                                                       |
| __--freeze_loaded_weights__                 | Freezes all the loaded weights so that they are not modified during training.                                                                 |
| __--freeze_weights_containing__ ___STR___   | Freezes the weights containing the string ___STR___ in their path. Default: `''`.                                                             |
| __--unfreeze_weights_containing__ ___STR___ | Un-freezes the weights containing the string ___STR___ in their path. Default: `''`.                                                          |

#### Logging
| Argument                         | Description                                                                                                                |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| __--wandb__                      | If set, training data will be logged using WandB. See [Tracking experiments with WandB](#tracking-experiments-with-wandb). |
| __--print_iteration__ ___ITER___ | Logs training information to stdout, such as losses, after every ___ITER___ iterations. Default: `20`.                     |
| __--print_other_losses__         | If set, not only will the total loss value will be printed, but also the sub-losses.                                       |

### Model configuration

The model architecture, configuration and hyper-parameters are defined in `.yaml` files at the root directory.

It is recommended to create a configuration file to keep track of the architecture, settings and hyper-parameters for every experiment.
The default configuration file used for the paper is `wandb_config_padloc.yaml`. Please check the contents of this file for the available settings.

### Overriding the model configuration

While it is best to create a separate configuration file in order to keep track of the experiments,
the model configuration can also be overriden when calling the training script by passing additional arguments in the
form `key:value`.

For example, to use a batch size during training, different from that in the configuration, the command can be used as follows.
```shell
python training2D3D_hard_KITTI_DDP.py [...] batch_size:2
```

All of the settings and parameters in the configuration file can be overriden in this way. 
Please check the configuration files for more information on the available settings.

### Initializing with pre-trained weights

The model can be initialized with pre-trained weights, of either of the full model, or only some of its parts, like the backbone.
In order to do so, the path to the model checkpoint must be passed with the `--weights /PATH/TO/WEIGHTS` argument.

By default, no strict loading is performed, meaning that weights belonging to partial or mismatching architectures can
be loaded. The lists of loaded, missing and unmatched parameters will be logged to standard output.
If this is not the desired behavior, the `--strict_weight_load` flag can be used, in which case a mismatch between the
model configuration and the checkpoint weights will raise and exception.

Finally, all the loaded weights are trainable by default, meaning that their values will most likely change during training.
If some parts of the model should be frozen during training, the following options can be used.
* `--freeze_loaded_weights`: All of the learnable parameters that were loaded will be frozen during training.
* `--freeze_weights_containing STR`: Parameters containing the string `STR` will be frozen during training. For example, by passing `backbone`, we can freeze the entire backbone and leave the matching and registration heads free to learn.
* `--unfreeze_weights_containing STR`: Parameters containing the string `STR` will be unfrozen. Useful for allowing some weights to learn, after having used either of the previous options.

### Resuming a training

If a training gets interrupted due to a crash, power failure or any other reason, it can be resumed by passing the path to the last checkpoint with the `--weights /PATH/TO/LAST/CHECKPOINT` option, as well as using the `--resume` flag.

By using the `--resume` flag, the optimizer and epoch count will be set to the state saved in the checkpoint. Also, the model checkpoints will be saved in the same directory and the logging using WandB will be resumed, without creating a new experiment.

### Tracking experiments with WandB
Weights and Biases is an online service that can be used to log experimental data recorded
during training, such as environment, git commit, model configuration, log files, training losses, evaluation metrics, model checkpoints and more.

In order to make use of it:
1. Create an account at [WandB](https://wandb.ai/).
2. If neither the supplied `Dockerfile` nor `environment.yaml` were used during the installation procedure,
manually install the `wandb` python package using `pip`.
    ```shell
    pip install wandb
    ```
3. Generate an API Key at the user settings page of [WandB](https://wandb.ai/).
4. Save the login information to the local machine by running the following command and pasting the API key when prompted.
    ```shell
    wandb login 
    ```
   > &#x26a0;&#xfe0f; When running the training inside the Docker Container, it is best to not use the `--rm` flag when starting the image so as to not loose the wandb login information.
   > Otherwise, this step must be executed every time the container is run.
5. Use the `--wandb` flag when running the training.
# Installation

## Docker Container
While it is possible to directly create a virtual or anaconda environment and install all the required dependencies,
it is recommended to run the code inside a Docker Container.


### Prerequisites

In order to correctly run, a PC running Ubuntu with an NVIDIA GPU that supports CUDA 11.3 is required.

When running the code inside a Docker container, it is required to install Docker and
NVIDIA-Docker according to their official instructions.

1. Install Docker following the official [instructions](https://docs.docker.com/get-docker/).
2. Run the official post-installation [steps](https://docs.docker.com/engine/install/linux-postinstall/) for Docker in Linux.
3. Install NVIDIA-Docker following the official [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Installation Procedure
Follow these steps to create the Docker image used for running the PADLoC source code.

1. Clone the github repository.
    ```shell
    git clone https://https://github.com/robot-learning-freiburg/PADLoC
   ```

2. Build the Docker image.
    ```shell
    cd PADLoC
    docker build -f docker/Dockerfile --tag padloc .
    ```

### Running the Docker Container
Use the `docker run` command to start the container. Some useful flags and arguments are shown here. For more arguments and additional information on the `docker run` command, please refer to the
[official documentation](https://docs.docker.com/engine/reference/run/).

```shell
docker run \
    [-it] \
    [--rm] \
    [--gpus GPUS] \
    [-m MEM] \
    [--shm-size=SHM] \
    [-v /HOST/PATH:/GUEST/PATH[:ro] ] \
    [--name NAME] \
    IMG
```

| Argument                                               | Description                                                                                                                                                                                                         |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| __-it__                                                | Opens an interactive terminal.                                                                                                                                                                                      |
| __--rm__                                               | Automatically remove the container when it exits.                                                                                                                                                                   |
| __--gpus__ ___GPUS___                                  | Provide access to the host's GPUs. Use ```--gpus all``` to allow access to all available GPUs, or only allow some specific GPUs with ```--gpus '"device=0,2"'``` (notice the use of both single and double quotes). |
| __-m__ ___MEM___                                       | The maximum amount of memory that this container is allowed to use. Defined as an integer value with a suffix of either ```b``` (bytes), ```k``` (kilobytes), ```m``` (megabytes) or ```g``` (gigabytes).           |
| __--shm_size=__***SHM***                               | The maximum amount of shared memory that this container is allowed to use. Defined as an integer value with a suffix of either ```b``` (bytes), ```k``` (kilobytes), ```m``` (megabytes) or ```g``` (gigabytes).    |
| __-v__ ___/HOST/PATH___**:**___/GUEST/PATH___**[:ro]** | Mount a local path `/host/path` from the host as a volume on the container, with mount point in `/guest/path`. Optionally, `:ro` can be added at the end of the path mapping to specify a Read-Only volume.         |
| __--name__ ___NAME___                                  | Assigns a name to the running container, instead of using the auto-generated one. Useful for referring to the container.                                                                                            |
| ___IMG___                                              | Docker image name                                                                                                                                                                                                   |

#### Example Usage

Run the `padloc` docker container with all the GPUs, 64GB of memory, 16GB of shared memory and mounting the checkpoint,
dataset and output directories as volumes.

```shell
docker run \
    -it \
    --gpus all \
    -m 64g \
    --shm-size=16g \
    -v '/path/to/kitti/dataset/':/data \
    -v '/path/to/cp/':/cp \
    -v '/path/to/output/':/output \
    --name padloc \
    padloc
```

## Pre-trained Models
You can find the pre-trained model weights for PADLoC [here](https://drive.google.com/file/d/1SVuSpq74XfgMMgYOsqBFHh9gFbpY79J0/view?usp=sharing).
Download the `.tar` file and place it in the directory that will then be mounted in the Docker Container.

> &#x26a0;&#xfe0f; There is no need to extract the tarball, since the model loading method uses it directly.


## Custom environment

If you wish to setup your own Virtual or Anaconda environment, install the dependencies listed in the `environment.yaml` file.

Then, install the following packages according to how it is done in the Dockerfile:
* OpenPCDet
* PointNet2

>  &#x26a0;&#xfe0f; Versions of Open3D >= 0.15 have a different implementation of RANSAC that results in poor registration accuracy.
> Please make sure to install a version of Open3D between 0.12.0 and 0.14.2 for the best results.

> &#x26a0;&#xfe0f; Versions of SPConv >= 2.2 are not compatible with the provided pre-trained weights. 
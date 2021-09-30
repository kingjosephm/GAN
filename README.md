# Introduction
TODO

# Pipeline
The shell script `driver.sh` automates copying the latest development scripts from the local machine onto the GPU server. It prompts user's credential once, saving this as a variable. It also prompts user's password twice; the first time is to transfer the scripts onto the GPU server, and the second time is to ssh into the server itself. Once logged into the server the user should execute the docker commands (below) to run the scripts.

# Data
There are two sets of paired images for the Pix2Pix model. 1) A processed 256x256 BW/thermal image set located on the GPU server under `/home/kingj/train` and `/home/kingj/test.tar.xz`. 2) A b/w set of full-resolution (640w x 512h) paired images are located at `home/kingj/side-by-side_train`.

For the CycleGAN model these same images are stored separately under thermal and visible subdirectories in a parent directory located at `home/kingj/separated_bw`.

# Docker
To train the models, we've migrated the data described above into Docker volumes. Model output is also stored in a Docker volume.
## *Data*
### 1) Paired b/w processed training images for Pix2Pix
These images are stored in a volume called `MERGEN_GAN_train_data`. They were migrated into this volume via:

    docker volume create --name MERGEN_GAN_train_data \
      --opt type=none \
      --opt device=/home/kingj/train \
      --opt o=bind

### 2) Paired b/w full-resolution training images for Pix2Pix
These are located in a volume called `MERGEN_GAN_train_data_full`.

### Unpaired b/w training images for CycleGAN
Unpaired images (i.e. separated into thermal and visible subdirs) are located in a volume called `MERGEN_GAN_train_separated_full`.

## *Output*
Output is stored in a volume named `MERGEN_GAN_output`. This was instantiated as an empty volume via:

    docker volume create --name MERGEN_GAN_output

# Execute code in Docker container
Two options exist to train a model in a Docker container. Model output from either method will be stored in the volume `Output` under `./output/<YYYY-MM-DD-HHhmm>`, where the datetime corresponds to the time in which the Python script was executed.

## *Option 1*
Start an interactive Docker container executing Bash, then specify Python script. To begin a new interactive container:

    docker run -it \
        --name <name> \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_train_data,target=/data \
        --mount source=MERGEN_output,target=/output \
        --gpus device=GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5 \
        king0759/tf2_gpu_jupyter_mpl:v1 /bin/bash

Because we want to mount both data and scripts volumes we call `--mount` twice. Importantly, the parameters `-it` make the session interactive. `king0759/tf2_gpu_jupyter_mpl:v1` refers to the container image, which is downloaded from DockerHub if not already present on the remote host. This image was created based on the image `tensorflow/tensorflow:latest-gpu`, though with the addition of Matplotlib, Jupyterlab, and Pandas. `GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5` refers to the specific GPU ID to use; the full list of these IDs can be viewed via `nvidia-smi --list-gpus`.

Once inside the container has started, enter the command line prompt, e.g.:

    $ python3 <script> [options]

## *Option 2* (preferred)
Initialize a container to run a Python script and remove the container at the end. Volumes exist independent of container instances so the output will be stored in `Output`.

    docker run -it \
      --name <name> \
      -d \
      --rm \
      --mount type=bind,source=/home/kingj/scripts,target=/scripts \
      --mount source=MERGEN_GAN_train_data,target=/data \
      --mount source=MERGEN_GAN_output,target=/output \
      --gpus device=GPU-0c5076b3-fe4a-0cd8-e4b7-71c2037933c0 \
      king0759/tf2_gpu_jupyter_mpl:v1 \
      python3 <script> [options]

# Example: Execute Pix2Pix script
    
    docker run -it \
        --name p2p_full \
        -d --rm \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_GAN_train_data_full,target=/data \
        --mount source=MERGEN_GAN_output,target=/output \
        --gpus device=GPU-3c51591d-cfdb-f87c-ece8-8dcfdc81e67a \
        king0759/tf2_gpu_jupyter_mpl:v1 python3 ./scripts/pix2pix.py \
        --train --data=data --output=output --save-weights --steps=10000000 \
        --generator-loss='l1' --img-size=256

# Example: Execute CycleGAN script

    docker run -it \
        --name cgan \
        -d --rm \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_GAN_train_separated_full,target=/data \
        --mount source=MERGEN_GAN_output,target=/output \
        --gpus device=GPU-0c5076b3-fe4a-0cd8-e4b7-71c2037933c0 \
        king0759/tf2_gpu_jupyter_mpl:v1 python3 ./scripts/cycle_gan.py --train \
        --input-images=data/thermal --target-images=data/visible \
        --output=output --save-weights --epochs=200 --img-size=256
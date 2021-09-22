# Introduction
TODO

# Pipeline
The shell script `driver.sh` automates copying the latest development scripts from the local machine onto the GPU server. It prompts user's credential once, saving this as a variable. It also prompts user's password twice; the first time is to transfer the scripts onto the GPU server, and the second time is to ssh into the server itself. Once logged into the server the user should execute the docker commands (below) to run the scripts.

# Data
The training and test image data are located on the GPU server under `/home/kingj/train` and `/home/kingj/test`. Their compressed equivalents can also be found in this same parent directory.

# Docker
### Training images are stored in a volume named `MERGEN_GAN_train_data`
These images were migrated into this volume via:

    docker volume create --name MERGEN_GAN_train_data \
      --opt type=none \
      --opt device=/home/kingj/train \
      --opt o=bind

A very small subset of training images is stored in the volume `MERGEN_GAN_subset_data` for code development.

### Output is stored in a volume named `MERGEN_GAN_output`
This was instantiated as an empty volume via:

    docker volume create --name MERGEN_GAN_output

### Start an interactive container paired with these volumes

    docker run -it \
        --name <name> \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_train_data,target=/data \
        --mount source=MERGEN_output,target=/output \
        --gpus device=GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5 \
        king0759/tf2_gpu_jupyter_mpl:v1 /bin/bash

Because we want to mount both data and scripts volumes we call `--mount` twice. Importantly, the parameters `-it` make the session interactive. `king0759/tf2_gpu_jupyter_mpl:v1` refers to the container image, which is downloaded from DockerHub if not already present on the remote host. This image was created based on the image `tensorflow/tensorflow:latest-gpu`, though with the addition of Matplotlib, Jupyterlab, and Pandas. `GPU-7a7c102c-5f71-a0fd-2ac0-f45a63c82dc5` refers to the specific GPU ID to use; the full list of these IDs can be viewed via `nvidia-smi --list-gpus`.

### Run GAN model: Option 1
Start interactive docker container (follow above). Once this has started, enter, e.g.:

    $ python3 ./scripts/pix2pix.py --train --data=data --output=output --no-save-weights --steps=100

Output from a given run will be stored in the volume `Output` under `./output/<YYYY-MM-DD-HHhmm>`.

### Run GAN model: Option 2
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
      python3 ./scripts/pix2pix.py --train --data=data --output=output --no-save-weights --steps=100000 --generator-loss='l1'
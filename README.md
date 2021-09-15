# Introduction
TODO

# Pipeline
The shell script `driver.sh` automates copying the latest development scripts from the local machine onto the GPU server. It prompts user's credential once, saving this as a variable. It also prompts user's password twice; the first time is to transfer the scripts onto the GPU server, and the second time is to ssh into the server itself. Once logged into the server the user should execute the docker commands (below) to run the scripts.

# Data
The training and test image data are located on the GPU server under `/home/kingj/train` and `/home/kingj/test`. Their compressed equivalents can also be found in this same parent directory.

# Docker
### Training images stored in volume named `MERGEN_GAN_train_data`
These images were migrated into this volume via:

    docker volume create --name MERGEN_GAN_train_data \
      --opt type=none \
      --opt device=/home/kingj/train \
      --opt o=bind

### Output stored in volume named `MERGEN_GAN_output`
This was instantiated as an empty volume via:

    docker volume create --name MERGEN_GAN_output

### Start an interactive container paired with these volumes

    docker run -it \
        --name <name> \
        --mount type=bind,source=source=home/kingj/scripts,target=/scripts \
        --mount source=MERGEN_train_data,target=/data \
        --mount source=MERGEN_output,target=/output \
        --gpus 1 \
        tensorflow/tensorflow:latest-gpu /bin/bash

Because we want to mount both data and scripts volumes we call `--mount` twice. Importantly, the parameters `-it` make the session interactive. `tensorflow/tensorflow:latest-gpu` refers to the container image, which is downloaded from DockerHub if not already present on the remote host.

### Run GAN model
Once inside the running container, execute the shell script `container_driver.sh` via:

    sh ./scripts/container_driver.sh

Output from a given run will be stored in the container under `./output/<YYYY-MM-DD-HHhmm>`.


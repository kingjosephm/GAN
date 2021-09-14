# Introduction
TODO

# Data
The data are located on the GPU server under /home/kingj in two directories: `train` and `test`. Their compressed equivalents can also be found in this same parent directory.

# Docker container

### Launch an interactive container for the first time

    docker run -it \
    --name foo \
    --mount type=bind,source=/home/kingj/train,target=/data \
    --mount type=bind,source=home/kingj/scripts,target=/scripts \
    --gpus 1 \
    tensorflow/tensorflow:latest-gpu /bin/bash

Because we want to mount both data and scripts we call `--mount` twice. Importantly, the parameters `-it` make the session interactive. `tensorflow/tensorflow:latest-gpu` refers to the container image, which is downloaded from DockerHub if not already present on the remote host. ***Note - within the interactive session Linux commands apply.*** Depending on your machine you may need to increase memory allocation to the container via, e.g., `--memory="3g"`.

### Launch a *detached* container for the first time that calls a script

    docker run -d \
    --name foo \
    --mount type=bind,source=/home/kingj/train,target=/data \
    --mount type=bind,source=home/kingj/scripts,target=/scripts \
    --gpus 1 \
    tensorflow/tensorflow:latest-gpu python3 ./scripts/tf_pix2pix.py

Note - if there's an error in the provided script, there is no way to interact with the container as it will exit first due to the error in the script.

### Restart a previously instantiated, but stopped container in interactive mode

    docker start -i <container_ID or container_name>

### Reattach terminal to running container

    docker attach <container_ID or container_name>

### Detach termal and keep container running

    CTRL-p CTRL-q  # key sequence


### Other Docker commands
    $ docker ps  # lists running docker containers
    $ docker ps -a  # view all containers, both running and stopped
    $ docker container rm <container_name>  # deletes a container *CAUTION!*
    $ docker image ls  # view all docker images on host machine
    $ docker stats <container_name> --no-stream  # no-stream option presents just the current stats

## Linux commands
### Move data onto server <br>
    $ scp file1 file2 <credentials>:<remote_dir> # Do this before ssh remote in

### Unzip files
    $ unzip myzip.zip
    $ tar -xf train.tar.xz # unzips tar file

### Read text file (e.g. log.txt) in container

    $ input="/path/to/txt/file"
    $ while IFS= read -r line
    do
      echo "$line"
    done < "$input"

### Exit environment (including Docker interactive, which stops container)
    $ exit
# Useful docker info

### Restart a previously instantiated, but stopped container in interactive mode

    docker start -i <container_ID or container_name>

### Reattach terminal to running container

    docker attach <container_ID or container_name>

### Detach terminal but keep container running

    CTRL-p CTRL-q  # key sequence

### Copy data within container to host machine

    docker cp <container_name>:<source_path> <destination_path>

`source_path` pertains to the path within the container. Note this must be run from outside a running or stopped container. It is not possible to move data out of a removed (deleted) container.

### Other Docker commands
    $ docker ps  # lists running docker containers
    $ docker ps -a  # view all containers, both running and stopped
    $ docker container rm <container_name>  # deletes a container *CAUTION!*
    $ docker image ls  # view all docker images on host machine
    $ docker stats <container_name> --no-stream  # no-stream option presents just the current stats

# Linux commands
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
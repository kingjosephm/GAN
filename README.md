## Docker container


## Data
The data are located on the GPU server under /home/kingj in two directories: `train` and `test`. Their compressed equivalents can also be found in this same parent directory.

## Useful Linux commands
#### Move data onto server
$ `sftp` <user_credentials> # you will need to enter password <br>
$ `put` <./path/to/local/file> # move files from local to remote <br>
$ `get` <filename> # move files from remote to local

#### Unzip files
$ `unzip` myzip.zip <br>
$ `tar` -xf train.tar.xz # unzips tar file

## Useful Docker commands
$ `docker` ps  # lists running docker containers <br>
$ `docker` ps -a # view all containers, both running and stopped
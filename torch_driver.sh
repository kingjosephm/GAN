#!/bin/bash

# Script arguments about number of epochs (n) and number of epochs with LR decay (d)
while getopts e:d:s:o: flag
do
    case "${flag}" in
        e) n_epochs=${OPTARG};;
        d) n_epochs_decay=${OPTARG};;
        s) save_epoch_freq=${OPTARG};;
        o) orientation=${OPTARG};;
    esac
done

cd ..

# Install missing Python depenencies, if any
pip install -r ./scripts/pytorch-CycleGAN-and-pix2pix/requirements.txt

# Script start datetime
startdatetime=`date +"%Y-%m-%d_%Hh%M"`

# Create output directory for log
log_output=./output/$startdatetime/logs
mkdir -p $log_output

# Start log
exec 3>&1 1>> $log_output/"Log.txt" 2>&1


printf "\rTraining Pix2Pix model for $n_epochs epochs, $n_epochs_decay with decay" | tee /dev/fd/3
printf "\nStart date-time: $startdatetime\n\r" | tee /dev/fd/3
printf "\n-----------------------------\r\n" | tee /dev/fd/3


# Call script
python3 ./scripts/pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./data --model pix2pix --name experiment \
--direction $orientation --dataset_mode aligned --n_epochs $n_epochs --n_epochs_decay $n_epochs_decay \
--save_epoch_freq $save_epoch_freq --checkpoints_dir ./output/"$startdatetime" --input_nc 1 \
--output_nc 1 --verbose --display_id 0 --load_size 542 --crop_size 512 2>&1 | tee /dev/fd/3


enddatetime=`date +"%Y-%m-%d_%Hh%Mm"`
printf "\n-----------------------------\r\n" | tee /dev/fd/3
printf "\nEnd date-time: $enddatetime\n" | tee /dev/fd/3
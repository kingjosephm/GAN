# Author: Joe King
# CycleGAN in Tensorflow, credit: https://www.tensorflow.org/tutorials/generative/cyclegan

import time
import os
import random
import math
import json
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # suppresses plot
from IPython import display
from datetime import datetime

# Disable TensorFlow AUTO sharding policy warning
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


# Configure distributed training across GPUs, if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
else:  # Use the Default Strategy
    #strategy = tf.distribute.get_strategy()  # default distribution strategy
    strategy = tf.distribute.OneDeviceStrategy('/CPU:0')  # use for debugging

class CycleGAN:

    def __init__(self, config):
        self.config = config
        self.config['global_batch_size'] = self.config['batch_size'] * strategy.num_replicas_in_sync
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = self.optimizer()
        self.discriminator_optimizer = self.optimizer()
        self.loss_obj = self.loss_object()


def parse_opt():
    return

def main(opt):
    '''
    :param opt: argparse.Namespace
    :return: None
    '''

    # Directing output
    os.makedirs(opt.output, exist_ok=True)
    full_path = opt.output + '/' + datetime.now().strftime("%Y-%m-%d-%Hh%M")
    os.makedirs(full_path, exist_ok=True)  # will overwrite folder if model run within same minute

    # Log results
    log_dir = os.path.join(full_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if not opt.no_log:
        sys.stdout = open(os.path.join(log_dir, "Log.txt"), "w")
        sys.stderr = sys.stdout

    cgan = CycleGAN(vars(opt))

    # Create or read from model checkpoints
    checkpoint = tf.train.Checkpoint(generator_optimizer=cgan.generator_optimizer,
                                     discriminator_optimizer=cgan.discriminator_optimizer,
                                     generator=cgan.generator,
                                     discriminator=cgan.discriminator)

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(cgan.config, f)

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
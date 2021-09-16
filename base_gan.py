
import os
import random
import tensorflow as tf
from abc import ABC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # suppresses plot

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


class GAN(ABC):
    def __init__(self, config):
        self.config = config
        self.config['global_batch_size'] = self.config['batch_size'] * strategy.num_replicas_in_sync
        self.strategy = strategy
        self.options = options
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = self.optimizer()
        self.discriminator_optimizer = self.optimizer()
        self.loss_obj = self.loss_object()

    def load(self, image_file):
        '''
        :param image_file:
        :return:
        '''
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        try:
            image = tf.image.decode_png(image)
        except:
            image = tf.image.decode_jpeg(image)
        return image

    def show_img(self):
        '''
        View random input image
        :return:
        '''
        img_list = [i for i in os.listdir(self.config['data']) if '.png' in i or '.jpeg' in i]
        random_img = ''.join(random.sample(img_list, 1))
        input, real = self.load(self.config['data'] + f'/{random_img}')
        # Casting to int for matplotlib to display the images
        plt.figure()
        plt.imshow(input / 255.0)
        plt.figure()
        plt.imshow(real / 255.0)

    def resize(self, image, height, width):
        '''
        :param image:
        :param height:
        :param width:
        :return:
        '''
        return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Normalizing the images to [-1, 1]
    def normalize(self, image):
        '''
        :param image:
        :return:
        '''
        return (image / 127.5) - 1

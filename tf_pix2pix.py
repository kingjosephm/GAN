import json
import os
import random
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

# Pix2Pix in Tensorflow, credit: https://www.tensorflow.org/tutorials/generative/pix2pix

class p2p:
    def __init__(self, config=None):
        self.config = config

    def load(self, image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        try:
            image = tf.image.decode_png(image)
        except:
            image = tf.image.decode_jpeg(image)

        # Split each image tensor into two tensors:
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def show_img(self):
        '''
        View random input image
        :return:
        '''
        img_list = [i for i in os.listdir(self.config['IMG_PATH']) if '.png' in i or '.jpeg' in i]
        random_img = ''.join(random.sample(img_list, 1))
        input, real = self.load(self.config['IMG_PATH'] + f'/{random_img}')
        # Casting to int for matplotlib to display the images
        plt.figure()
        plt.imshow(input / 255.0)
        plt.figure()
        plt.imshow(real / 255.0)

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image

    def random_crop(self, input_image, real_image, height, width):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, height, width, 3])
        return cropped_image[0], cropped_image[1]

    # Normalizing the images to [-1, 1]
    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        # Resizing to height+30px by width+30px
        input_image, real_image = self.resize(input_image, real_image, self.config['IMG_HEIGHT']+30, self.config['IMG_WIDTH']+30)

        # Random cropping back to height, width
        input_image, real_image = self.random_crop(input_image, real_image,  self.config['IMG_HEIGHT'], self.config['IMG_WIDTH'])

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def process_images(self, image_file, train=True):
        input_image, real_image = self.load(image_file)
        if train:
            input_image, real_image = self.random_jitter(input_image, real_image)
        else: # if test images, don't apply random jitter, just resize
            input_image, real_image = self.resize(input_image, real_image, self.config['IMG_HEIGHT'], self.config['IMG_WIDTH'])
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image



def main(opt):
    parser = argparse.ArgumentParser()

if __name__ == '__main__':


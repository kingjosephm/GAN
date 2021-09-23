from PIL import Image
import os, cv2
import tensorflow as tf
import numpy as np


def split_image(img, output_path):
    height = 256  # Desired final height
    width = 256  # Desired final width
    k = 0
    subdir = {0: '/truth/', 1: '/input/'}

    im = Image.open(img)
    imgwidth, imgheight = im.size
    assert (imgwidth == 512)
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            a = im.crop(box)
            path = output_path + subdir[k]
            os.makedirs(path, exist_ok=True)
            a.save(path + os.path.split(img)[-1])
            k += 1
    return "Done with all images"

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def concat_images(path, output_path, resize_width=None, resize_height=None):
    '''
    Function to create concatenated visible (left) and thermal (right) image in same PNG for pix2pix model.
    Some poor-quality visible images were also manually deleted, so the function checks if the matching thermal images
    are still on disk and deletes these images too. Also crops thermal and visible slightly since there was a black
    stripe in all visible images.
    :param path: str, root dir path
    :param output_path: str, output path
    :param resize_width: int, optional to resize width
    :param resize_height: int, optional to resize height
    :return: None, outputs to disk
    '''
    therm = sorted([i for i in os.listdir(path) if "therm_adj" in i])  # thermal adjusted images

    for i in therm:

        # See if visual counterpart exists, otherwise delete thermal (done since many images not useful)
        if not os.path.exists(os.path.join(path, 'vis'+i[9:])):
            os.remove(os.path.join(path, i))  # remove adjusted thermal
            os.remove(os.path.join(path, 'therm'+i[9:]))  # remove original thermal
            continue
        else: # match does exist
            th = cv2.imread(os.path.join(path, i))
            vi = cv2.imread(os.path.join(path, 'vis'+ i[9:]))  # Will error if no match found

        # Verify shape
        assert(th.shape == vi.shape), f"Size mismatch! image '{i}' is not the same size as its visible counterpart!"

        # Crop images - original real images had black strip that we want to remove
        vi = vi[10:490, 40:]
        th = th[10:490, 40:]

        # Resize (Optional)
        if (resize_height is not None) and (resize_width is not None):
            th = cv2.resize(th, (resize_width, resize_height), interpolation= cv2.INTER_LINEAR)
            vi = cv2.resize(vi, (resize_width, resize_height), interpolation= cv2.INTER_LINEAR)

        # Concat and output
        conc = np.concatenate((vi, th), axis=1)
        dir_name = path.split(os.sep)[-2]
        cv2.imwrite(os.path.join(output_path, dir_name + '_' + i[9:]), conc)

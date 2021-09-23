import os
import tensorflow as tf
import numpy as np
try:
    import cv2 # not in docker container, only run locally
except ModuleNotFoundError:
    pass

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

        """
        # Use via:
        for subdir, dirs, files in os.walk(root):
            merge_images(subdir, output_path, resize_width=640, resize_height=512)
        """
def load(image_file):
    """
    :param image_file: str, path to image
    :return: tensorflow.python.framework.ops.EagerTensor
    """
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    try:
        image = tf.image.decode_png(image)
    except:
        image = tf.image.decode_jpeg(image)

    # Cast to float32 tensors
    image = tf.cast(image, tf.float32)
    return image

def split_images(path, resize_width=None, resize_height=None):
    '''
    :param path: str, path to root
    :param resize_width: int, optional to resize width
    :param resize_height: int, optional to resize height
    :return: None, outputs to disk
    '''

    images = [i for i in os.listdir(path) if ".png" in i or ".jpeg" in i]

    for x in images:

        image = load(os.path.join(path, x))

        # Split each image tensor into two tensors:
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        # Resize (Optional)
        if (resize_height is not None) and (resize_width is not None):
            input_image = tf.image.resize(input_image, [resize_height, resize_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(real_image, [resize_height, resize_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        cv2.imwrite(os.path.join(path, '../separated/visible', x), real_image.numpy())
        cv2.imwrite(os.path.join(path, '../separated/thermal', x), input_image.numpy())

    return "Done with all images"
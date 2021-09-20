from PIL import Image
import os
import tensorflow as tf


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
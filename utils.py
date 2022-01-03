import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def pix2pix_losses():
    """
    Instantiates empty dictionary with each Pix2Pix cost function
    :return: dict
    """
    return {'Generator Total Loss': [],
            'Generator Loss (Primary)': [],
            'Generator Loss (Secondary)': [],
            'Discriminator Loss': []}

def cyclegan_losses():
    """
    Instantiates empty dictionary with each CycleGAN cost function
    :return: dict
    """
    return {'X->Y Generator Loss': [],
            'Y->X Generator Loss': [],
            'Total Cycle Loss': [],
            'Total X->Y Generator Loss': [],
            'Total Y->X Generator Loss': [],
            'Discriminator X Loss': [],
            'Discriminator Y Loss': []}
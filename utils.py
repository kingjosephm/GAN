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

def make_fig(df: pd.DataFrame, title: str, output_path: str):
    '''
    Creates two line graphs in same figure using Matplotlib. Outputs as PNG to disk.
    :param df: pd.Series, mean loss by epoch
    :param title: str, title of figure. Also used to name PNG plot when outputted to disk.
    :param output_path: str, path to output PNG
    :return: None, writes figure to disk
    '''
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(df, alpha=0.7, label='Epoch Mean')
    plt.plot(df.ewm(alpha=0.1).mean(), color='red', linewidth=2, label='Weighted Epoch Mean')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title}')
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)  # Creates output directory if not existing
    plt.savefig(os.path.join(output_path, f'{title}.png'), dpi=200)
    plt.close()
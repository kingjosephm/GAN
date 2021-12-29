import tensorflow as tf
from abc import ABC, abstractmethod
from utils import InstanceNormalization
import matplotlib
matplotlib.use('Agg') # suppresses plot

"""
    Pix2Pix and CycleGAN in Tensorflow
    Credit:
        https://www.tensorflow.org/tutorials/generative/pix2pix
        https://www.tensorflow.org/tutorials/generative/cyclegan
        https://github.com/tensorflow/examples/blob/d97aa060cb00ae2299b4b32591b8489df38e85ef/tensorflow_examples/models/pix2pix/pix2pix.py

"""

tf.keras.backend.clear_session()

# Configure distributed training across GPUs, if available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

class GAN(ABC):
    def __init__(self, config):
        self.config = config
        self.loss_obj = self.loss_object()

    def load(self, image_file: str, resize: bool = False):
        """
        :param image_file: str, full path to image file
        :param resize: bool, whether to resize image on read in to ensure consistently-sized images in tensor
        :return: tf.Tensor
        """
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        try:
            image = tf.image.decode_png(image, channels=int(self.config['channels']))
        except:
            image = tf.image.decode_jpeg(image, channels=int(self.config['channels']))

        # Cast to float32 tensors
        image = tf.cast(image, tf.float32)

        if resize:
            image = self.resize(image, self.config['img_size'], self.config['img_size'])
        return image

    def resize(self, image: tf.Tensor, height: int, width: int):
        """
        :param image:
        :param height:
        :param width:
        :return:
        """
        return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Normalizing the images to [-1, 1]
    def normalize(self, image: tf.Tensor):
        """
        :param image:
        :return:
        """
        return (image / 127.5) - 1

    def downsample(self, filters: int, size: int, norm_type: str = 'batchnorm', apply_norm: bool = True):
        """Downsamples an input.
        Conv2D => Batchnorm => LeakyRelu
        Args:
          filters: number of filters
          size: filter size
          norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
          apply_norm: If True, adds the batchnorm layer
        Returns:
          Downsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(tf.keras.layers.BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters: int, size: int, norm_type: str = 'batchnorm', apply_dropout: bool = False):
        """Upsamples an input.
        Conv2DTranspose => Batchnorm => Dropout => Relu
        Args:
          filters: number of filters
          size: filter size
          norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
          apply_dropout: If True, adds the dropout layer
        Returns:
          Upsample Sequential Model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Discriminator(self, norm_type: str = 'batchnorm', target: bool = True):
        """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
        Args:
          norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
          target: Bool, indicating whether target image is an input or not.
        Returns:
          Discriminator model
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, int(self.config['channels'])], name='input_image')
        x = inp

        if target:
            tar = tf.keras.layers.Input(shape=[None, None, int(self.config['channels'])], name='target_image')
            x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer,
            use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        if norm_type.lower() == 'batchnorm':
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(
            1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        if target:
            return tf.keras.Model(inputs=[inp, tar], outputs=last)
        else:
            return tf.keras.Model(inputs=inp, outputs=last)

    def loss_object(self):
        """
        :return:
        """
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real, generated, factor: float = 1.0):
        """
        Discriminator loss.
        :param real:
        :param generated:
        :param factor: float, total_disc_loss multiplied with constant factor
        :return:
            total_discriminator_loss: float
        """
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        return (real_loss + generated_loss) * factor

    def optimizer(self, learning_rate: float = 2e-4, beta_1: float = 0.5, beta_2: float = 0.999):
        """
        Optimizer for both generator and discriminators
        :return: tf.keras Adam optimizer
        """
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    @abstractmethod
    def random_crop(self, *args, **kwargs):
        return

    @abstractmethod
    def random_jitter(self, *args, **kwargs):
        return

    @abstractmethod
    def process_images_train(self, image_file):
        return

    @abstractmethod
    def process_images_pred(self, image_file):
        return

    @abstractmethod
    def image_pipeline(self):
        return

    @abstractmethod
    def generator_loss(self, *args, **kwargs):
        return

    @abstractmethod
    def Generator(self, *args, **kwargs):
        return

    @abstractmethod
    def generate_images(self, *args, **kwargs):
        return

    @abstractmethod
    def train_step(self, *args, **kwargs):
        return

    @abstractmethod
    def fit(self, *args, **kwargs):
        return

    @abstractmethod
    def predict(self, *args, **kwargs):
        return
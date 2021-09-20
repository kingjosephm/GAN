import time
import os
import random
import json
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # suppresses plot
from IPython import display
from datetime import datetime
from base_gan import GAN


"""
    CycleGAN in Tensorflow
    Credit:
        https://www.tensorflow.org/tutorials/generative/cyclegan
        https://github.com/tensorflow/examples/blob/d97aa060cb00ae2299b4b32591b8489df38e85ef/tensorflow_examples/models/pix2pix/pix2pix.py

"""

class CycleGAN(GAN):

    def __init__(self, config):
        super().__init__(config)
        self.generator_g = self.Generator(self.config['output_channels'], norm_type='instancenorm')
        self.generator_f = self.Generator(self.config['output_channels'], norm_type='instancenorm')
        self.discriminator_x = super().Discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = super().Discriminator(norm_type='instancenorm', target=False)
        self.generator_g_optimizer = super().optimizer()
        self.generator_f_optimizer = super().optimizer()
        self.discriminator_x_optimizer = super().optimizer()
        self.discriminator_y_optimizer = super().optimizer()

    def random_crop(self, image, height, width):
        """
        :param image:
        :param height:
        :param width:
        :return:
        """
        return tf.image.random_crop(image, size=[height, width, 3])

    def random_jitter(self, image):
        """
        :param image:
        :return:
        """
        # Resizing to height+30px by width+30px
        image = super().resize(image, self.config['img_size']+30, self.config['img_size']+30)

        # Random cropping back to height, width
        image = self.random_crop(image, self.config['img_size'], self.config['img_size'])

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def process_images_train(self, image_file):
        """
        Loads individual image, applies random jitter, normalizes image. Processing for train images only.
        :param image_file:
        :return:
        """
        image = super().load(image_file, resize=True)
        image = self.random_jitter(image)
        image = super().normalize(image)
        return image

    def process_images_pred(self, image_file):
        """
        Loads individual image, resizes, normalizes image. Processing for test/pred images only.
        :param image_file:
        :return:
        """
        image = super().load(image_file, resize=True)
        image = super().resize(image, self.config['img_size'], self.config['img_size'])
        image = super().normalize(image)
        return image

    def image_pipeline(self, predict=False):
        """
        :param predict: bool, whether or not to create train/test split. False treats all images as valid for prediction.
        :return:
            train - tf.distribute.DistributedDataset object
            test - tf.distribute.DistributedDataset (or None if predict=True)
        """

        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents_X = [i for i in os.listdir(self.config['input_images']) if 'png' in i or 'jpg' in i]
        contents_Y = [i for i in os.listdir(self.config['target_images']) if 'png' in i or 'jpg' in i]

        if predict:  # all images in `train` used for prediction; they're not training images, only kept for consistency
            assert(contents_X), "No JPEG or PNG images found in input image directory!"
            train_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in contents_X]) # resize all to same dims in case images different sizes
            train_X = train_X.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            train_X = train_X.shuffle(self.config['buffer_size'])
            train_X = train_X.batch(self.config["global_batch_size"])
            train_Y = None
            test_X = None

        else:  # if train mode, break into train/test
            assert(len(contents_X) >= 2), f"Insufficient number of training examples in input image directory! " \
                                          f"At least 2 are required, but found {len(contents_X)}!"
            assert(len(contents_Y) >= 2), f"Insufficient number of training examples in target image directory! " \
                                          f"At least 2 are required, but found {len(contents_Y)}!"

            # Randomly select 1 image to view training progress
            test_X = random.sample(contents_X, 1)

            train_X = [i for i in contents_X if i not in test_X]
            train_Y = [i for i in contents_Y]

            test_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in test_X])
            train_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in train_X])
            train_Y = tf.data.Dataset.from_tensor_slices([self.config['target_images'] + '/' + i for i in train_Y])

            # process test images
            test_X = test_X.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            test_X = test_X.cache().shuffle(self.config['buffer_size'])
            test_X = test_X.batch(self.config["global_batch_size"])
            test_X = test_X.with_options(self.options)

            # process training images
            train_X = train_X.cache().map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train_Y = train_Y.cache().map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train_X = train_X.shuffle(self.config['buffer_size'])
            train_Y = train_Y.shuffle(self.config['buffer_size'])
            train_X = train_X.batch(self.config["global_batch_size"])
            train_Y = train_Y.batch(self.config["global_batch_size"])
            train_X = train_X.with_options(self.options)
            train_Y = train_Y.with_options(self.options)

        return train_X, train_Y, test_X

    def Generator(self, output_channels, norm_type='batchnorm'):
        """
        Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          output_channels: Output channels
          norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        Returns:
          Generator model
        """

        with self.strategy.scope():

            down_stack = [
                super().downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
                super().downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
                super().downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
                super().downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
                super().downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
                super().downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
                super().downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
                super().downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
            ]

            up_stack = [
                super().upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
                super().upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
                super().upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
                super().upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
                super().upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
                super().upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
                super().upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
            ]

            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(
                output_channels, 4, strides=2,
                padding='same', kernel_initializer=initializer,
                activation='tanh')  # (bs, 256, 256, 3)

            concat = tf.keras.layers.Concatenate()

            inputs = tf.keras.layers.Input(shape=[None, None, 3])
            x = inputs

            # Downsampling through the model
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)

            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                x = concat([x, skip])

            x = last(x)

            return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, generated):
        """
        :param generated:
        :return:
        """
        with self.strategy.scope():
            return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        :param real_image:
        :param cycled_image:
        :return:
        """
        with self.strategy.scope():
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
            LAMBDA = 10
            return loss1 * LAMBDA

    def identity_loss(self, real_image, same_image):
        """
        :param real_image:
        :param same_image:
        :return:
        """
        with self.strategy.scope():
            loss = tf.reduce_mean(tf.abs(real_image - same_image))
            LAMBDA = 10
            return LAMBDA * 0.5 * loss

    def generate_images(self, model, test_input, epoch, output_path):
        """
        :param model:
        :param test_input:
        :param epoch:
        :param output_path:
        :return:
        """
        prediction = model(test_input, training=True)
        plt.figure(figsize=(12, 6))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        plot_path = os.path.join(output_path, 'test_images')
        os.makedirs(plot_path, exist_ok=True) # dir should not exist
        plt.savefig(os.path.join(plot_path, f'epoch{epoch}.png'), dpi=80)

    @tf.function
    def train_step(self, real_x, real_y, epoch, summary_writer):
        """
        :param real_x:
        :param real_y:
        :param epoch:
        :param summary_writer:
        :return:
        """
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x, 0.5)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y, 0.5)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                  self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      self.discriminator_y.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_g_loss', tf.reduce_sum(gen_g_loss), step=epoch)
            tf.summary.scalar('gen_f_loss', tf.reduce_sum(gen_f_loss), step=epoch)
            tf.summary.scalar('total_cycle_loss', tf.reduce_sum(total_cycle_loss), step=epoch)
            tf.summary.scalar('total_gen_g_loss', tf.reduce_sum(total_gen_g_loss), step=epoch)
            tf.summary.scalar('total_gen_f_loss', tf.reduce_sum(total_gen_f_loss), step=epoch)
            tf.summary.scalar('disc_x_loss', disc_x_loss, step=epoch)
            tf.summary.scalar('disc_y_loss', disc_y_loss, step=epoch)

    def fit(self, train_X, train_Y, test_X, epochs, summary_writer, output_path, checkpoint_manager=None, save_weights=True):

        print("\nTraining...\n", flush=True)

        example_X = next(iter(test_X.take(1)))
        start = time.time()

        for epoch in range(epochs):

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((train_X, train_Y)):
                self.train_step(image_x, image_y, epoch, summary_writer)
                if n % 10 == 0:
                    print('.', end='', flush=True)
                n += 1

            display.clear_output(wait=True)

            # Every 5 epochs save weights and generate predicted image
            if (epoch + 1) % 5 == 0:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator_g, example_X, epoch, output_path)

            if (epoch + 1) == self.config['epochs']:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator_g, example_X, epoch, output_path)

            print('Cumulative training duration at epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

    def predict(self, pred_ds, output_path):
        pass

def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--input-images', type=str, help='path to input images', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per replica')
    parser.add_argument('--buffer-size', type=int, default=1000, help='buffer size')
    parser.add_argument('--output-channels', type=int, default=3, help='number of color channels to output')
    parser.add_argument('--no-log', action='store_true', help='turn off script logging, e.g. for CLI debugging')
    parser.add_argument('--generator-loss', type=str, default='l1', choices=['l1', 'ssim'], help='combined generator loss function')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    parser.add_argument('--target-images', type=str, help='path to target images', required='--train' in sys.argv)
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train', required='--train' in sys.argv)
    group2 = parser.add_mutually_exclusive_group(required='--train' in sys.argv)
    group2.add_argument('--save-weights', action='store_true', help='save model checkpoints and weights')
    group2.add_argument('--no-save-weights', action='store_true', help='do not save model checkpoints or weights')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    return parser.parse_args()

def main(opt):
    """
    :param opt: argparse.Namespace
    :return: None
    """

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
    checkpoint = tf.train.Checkpoint(generator_g=cgan.generator_g,
                                   generator_f=cgan.generator_f,
                                   discriminator_x=cgan.discriminator_x,
                                   discriminator_y=cgan.discriminator_y,
                                   generator_g_optimizer=cgan.generator_g_optimizer,
                                   generator_f_optimizer=cgan.generator_f_optimizer,
                                   discriminator_x_optimizer=cgan.discriminator_x_optimizer,
                                   discriminator_y_optimizer=cgan.discriminator_y_optimizer)

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(cgan.config, f)

    if opt.predict: # if predict mode
        prediction_dataset, _, _ = cgan.image_pipeline(predict=True)
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()
        cgan.predict(prediction_dataset, full_path)

    if opt.train: # if train mode
        train_X, train_Y, test_X = cgan.image_pipeline(predict=False)

        # Outputting model checkpoints
        if opt.save_weights:
            checkpoint_dir = os.path.join(full_path, 'training_checkpoints')
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        else:
            manager = None

        # Summary witer file for tensorboard
        summary_writer = tf.summary.create_file_writer(log_dir)

        cgan.fit(train_X=train_X,
                 train_Y=train_Y,
                 test_X=test_X,
                 epochs=cgan.config['epochs'],
                 summary_writer=summary_writer,
                 output_path=full_path,
                 checkpoint_manager=manager,
                 save_weights=cgan.config['save_weights'])

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
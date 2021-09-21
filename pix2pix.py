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
from datetime import datetime
from base_gan import GAN
import pdb

"""
    Pix2Pix in Tensorflow
    Credit:
        https://www.tensorflow.org/tutorials/generative/pix2pix
        https://github.com/tensorflow/examples/blob/d97aa060cb00ae2299b4b32591b8489df38e85ef/tensorflow_examples/models/pix2pix/pix2pix.py

"""

# Disable TensorFlow AUTO sharding policy warning
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

class Pix2Pix(GAN):
    def __init__(self, config):
        super().__init__(config)
        self.options = options
        self.generator = self.Generator()
        self.discriminator = super().Discriminator(target=True)
        self.generator_optimizer = super().optimizer()
        self.discriminator_optimizer = super().optimizer()
        self.model_metrics = {'Gen total loss': [],
                              'Gen loss': [],
                              'Gen loss2': [],
                              'Disc loss': []}

    def split_img(self, image_file):
        """
        :param image_file:
        :return:
        """
        image = super().load(image_file)

        # Split each image tensor into two tensors:
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        return input_image, real_image

    def random_crop(self, input_image, real_image, height, width):
        """
        :param input_image:
        :param real_image:
        :param height:
        :param width:
        :return:
        """
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, height, width, 3])
        return cropped_image[0], cropped_image[1]

    @tf.function()
    def random_jitter(self, input_image, real_image):
        """
        :param input_image:
        :param real_image:
        :return:
        """
        # Resizing to height+30px by width+30px
        input_image = super().resize(input_image, self.config['img_size']+30, self.config['img_size']+30)
        real_image = super().resize(real_image, self.config['img_size']+30, self.config['img_size']+30)

        # Random cropping back to height, width
        input_image, real_image = self.random_crop(input_image, real_image,  self.config['img_size'], self.config['img_size'])

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def process_images_train(self, image_file):
        """
        Loads individual image, applies random jitter, normalizes image. Processing for train images only.
        :param image_file:
        :return:
        """
        input_image, real_image = self.split_img(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image = super().normalize(input_image)
        real_image = super().normalize(real_image)
        return input_image, real_image

    def process_images_pred(self, image_file):
        """
        Loads individual image, resizes, normalizes image. Processing for test/pred images only.
        :param image_file:
        :return:
        """
        input_image, real_image = self.split_img(image_file)
        input_image = super().resize(input_image, self.config['img_size'], self.config['img_size'])
        real_image = super().resize(real_image, self.config['img_size'], self.config['img_size'])
        input_image = super().normalize(input_image)
        real_image = super().normalize(real_image)
        return input_image, real_image

    def image_pipeline(self, predict=False):
        """
        :param predict: bool, whether or not to create train/test split. False treats all images as valid for prediction.
        :return:
            train - tf.distribute.DistributedDataset object
            test - tf.distribute.DistributedDataset (or None if predict=True)
        """

        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents = [i for i in os.listdir(self.config['data']) if 'png' in i or 'jpg' in i]

        if predict:  # all images in `train` used for prediction; they're not training images, only kept for consistency
            assert(contents), "No JPEG or PNG images found in data directory!"
            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in contents])
            train = train.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.shuffle(self.config['buffer_size'])
            train = train.batch(self.config["global_batch_size"])
            test = None

        else:  # if train mode, break into train/test
            assert(len(contents) >=2), f"Insufficient number of training examples in data directory! " \
                                          f"At least 2 are required, but found {len(contents)}!"

            # Randomly select 1 image to view training progress
            test = random.sample(contents, 1)
            train = [i for i in contents if i not in test]

            test = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in test])
            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in train])

            # process test images
            test = test.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            test = test.shuffle(self.config['buffer_size'])
            test = test.batch(self.config["global_batch_size"]).repeat()
            test = test.with_options(self.options)

            # process training images
            train = train.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.shuffle(self.config['buffer_size'])
            train = train.batch(self.config["global_batch_size"]).repeat()
            train = train.with_options(self.options)

        return train, test

    def Generator(self):
        """
        Modified U-Net encoder.
        :return: tf.keras Model class
        """

        with self.strategy.scope():

            inputs = tf.keras.layers.Input(shape=[256, 256, 3])

            down_stack = [
                super().downsample(64, 4, apply_norm=False),  # (batch_size, 128, 128, 64)
                super().downsample(128, 4),  # (batch_size, 64, 64, 128)
                super().downsample(256, 4),  # (batch_size, 32, 32, 256)
                super().downsample(512, 4),  # (batch_size, 16, 16, 512)
                super().downsample(512, 4),  # (batch_size, 8, 8, 512)
                super().downsample(512, 4),  # (batch_size, 4, 4, 512)
                super().downsample(512, 4),  # (batch_size, 2, 2, 512)
                super().downsample(512, 4),  # (batch_size, 1, 1, 512)
            ]

            up_stack = [
                super().upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
                super().upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
                super().upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
                super().upsample(512, 4),  # (batch_size, 16, 16, 1024)
                super().upsample(256, 4),  # (batch_size, 32, 32, 512)
                super().upsample(128, 4),  # (batch_size, 64, 64, 256)
                super().upsample(64, 4),  # (batch_size, 128, 128, 128)
            ]

            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(self.config['output_channels'], 4,
                                                   strides=2,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   activation='tanh')  # (batch_size, 256, 256, 3)

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
                x = tf.keras.layers.Concatenate()([x, skip])

            x = last(x)

            model = tf.keras.Model(inputs=inputs, outputs=x)

        return model

    def generator_loss(self, disc_generated_output, gen_output, target, input_image):
        """
        Generator loss
        :param disc_generated_output:
        :param gen_output:
        :param target:
        :param input_image:
        :return:
        """
        with self.strategy.scope():

            gan_loss = tf.reduce_sum(self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output))  * (1. / self.config['global_batch_size'])

            if self.config['generator_loss']=='l1':
                # Mean absolute error
                gan_loss2 = tf.reduce_mean(tf.abs(target - gen_output))
            elif self.config['generator_loss']=='ssim':
                # SSIM loss, see https://www.tensorflow.org/api_docs/python/tf/image/ssim
                gan_loss2 = (1 - tf.reduce_sum(tf.image.ssim(input_image, target, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)))

            total_gen_loss = gan_loss + (100 * gan_loss2) # 100=LAMBDA

        return total_gen_loss, gan_loss, gan_loss2

    @tf.function
    def train_step(self, input_image, target, step, summary_writer):
        """
        :param input_image:
        :param target:
        :param step:
        :param summary_writer: tf.summary_writer object
        :return:
        """

        # TODO - consider different numbers of generator or discriminator steps each time
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_gan_loss2 = self.generator_loss(disc_generated_output, gen_output, target, input_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        # Model metrics to use in tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 100)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 100)
            tf.summary.scalar('gen_gan_loss2', gen_gan_loss2, step=step // 100) # L1 loss or SSIM
            tf.summary.scalar('disc_loss', disc_loss, step=step // 100)

        return gen_total_loss, gen_gan_loss, gen_gan_loss2, disc_loss  # return model metrics as unable to convert to numpy within @tf.function

    def generate_images(self, model, test_input, tar, step, output_path):
        """
        :param model:
        :param test_input:
        :param tar:
        :param step:
        :param output_path:
        :return:
        """
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 6))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        plot_path = os.path.join(output_path, 'test_images')
        os.makedirs(plot_path, exist_ok=True) # dir should not exist
        plt.savefig(os.path.join(plot_path, f'step_{step}.png'), dpi=80)
        plt.close()

    def fit(self, train_ds, test_ds, steps, summary_writer, output_path, checkpoint_manager=None, save_weights=True):
        """
        :param train_ds:
        :param test_ds:
        :param steps:
        :param summary_writer: tf.summary_writer object
        :param output_path: str, path to output test images across training steps
        :param checkpoint_manager:
        :param save_weights: bool, whether to save model weights per 5k training steps and at end, along with model checkpoints
        :return:
        """

        print("\nTraining...\n", flush=True)

        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step % 1000 == 0) and (step > 0):

                print(f'\nCumulative training time at step {step+1}: {time.time() - start:.2f} sec\n')

            gen_total_loss, gen_gan_loss, gen_gan_loss2, disc_loss = self.train_step(input_image, target, step, summary_writer)

            # Performance metrics from step into dict
            # Note - must be done outside self.train_step() as numpy operations do not work in tf.function
            self.model_metrics['Gen total loss'].append(tf.reduce_sum(gen_total_loss, axis=None).numpy().tolist())
            self.model_metrics['Gen loss'].append(tf.reduce_sum(gen_gan_loss, axis=None).numpy().tolist())
            self.model_metrics['Gen loss2'].append(tf.reduce_sum(gen_gan_loss2, axis=None).numpy().tolist())
            self.model_metrics['Disc loss'].append(tf.reduce_sum(disc_loss, axis=None).numpy().tolist())

            # Save (checkpoint) the model every 5k steps and at end
            # Also saves generated training image
            if (step + 1) % 50000 == 0:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator, example_input, example_target, step+1, output_path)

            # At end save checkpoint and final test image
            if (step + 1) == self.config['steps']:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator, example_input, example_target, step+1, output_path)
                print(f'Cumulative training time at end of {step} steps: {time.time() - start:.2f} sec\n')

    def predict(self, pred_ds, output_path):
        """
        :param pred_ds:
        :param output_path:
        :return:
        """
        print("\nRendering images using pretrained weights\n")

        img_nr = 0
        for input, target in pred_ds:
            prediction = self.generator(input, training=False)

            # Three image subplots
            plt.figure(figsize=(15, 6))
            display_list = [input[0], target[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(title[i])
                # Getting the pixel values in the [0, 1] range to plot.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')

            plot_path = os.path.join(output_path, 'prediction_images')
            os.makedirs(plot_path, exist_ok=True)  # dir should not exist
            plt.savefig(os.path.join(plot_path, f'img_{img_nr}.png'), dpi=80)
            plt.close()

            # Just prediction image
            plt.figure(figsize=(6, 6))
            plt.imshow(prediction[0] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig(os.path.join(plot_path, f'prediction_{img_nr}.png'), dpi=80)
            plt.close()
            img_nr += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--data', type=str, help='path to data', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per replica')
    parser.add_argument('--buffer-size', type=int, default=400, help='buffer size')
    parser.add_argument('--output-channels', type=int, default=3, help='number of color channels to output')
    parser.add_argument('--no-log', action='store_true', help='turn off script logging, e.g. for CLI debugging')
    parser.add_argument('--generator-loss', type=str, default='l1', choices=['l1', 'ssim'], help='combined generator loss function')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    group2 = parser.add_mutually_exclusive_group(required='--train' in sys.argv)
    group2.add_argument('--save-weights', action='store_true', help='save model checkpoints and weights')
    group2.add_argument('--no-save-weights', action='store_true', help='do not save model checkpoints or weights')
    parser.add_argument('--steps', type=int, default=10, help='number of training steps to take')
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

    p2p = Pix2Pix(vars(opt))

    # Create or read from model checkpoints
    checkpoint = tf.train.Checkpoint(generator_optimizer=p2p.generator_optimizer,
                                     discriminator_optimizer=p2p.discriminator_optimizer,
                                     generator=p2p.generator,
                                     discriminator=p2p.discriminator)

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(p2p.config, f)

    if opt.predict: # if predict mode
        prediction_dataset, _ = p2p.image_pipeline(predict=True)
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()
        p2p.predict(prediction_dataset, full_path)

    if opt.train: # if train mode
        train_dataset, test_dataset = p2p.image_pipeline(predict=False)

        # Outputting model checkpoints
        if opt.save_weights:
            checkpoint_dir = os.path.join(full_path, 'training_checkpoints')
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        else:
            manager = None

        # Summary writer file for tensorboard
        summary_writer = tf.summary.create_file_writer(log_dir)

        p2p.fit(train_ds=train_dataset,
                test_ds=test_dataset,
                steps=p2p.config['steps'],
                summary_writer=summary_writer,
                output_path=full_path,
                checkpoint_manager=manager,
                save_weights=p2p.config['save_weights'])

        # Output model metrics dict to log dir
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(p2p.model_metrics, f)

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
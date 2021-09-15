# Author: Joe King
# Pix2Pix in Tensorflow, credit: https://www.tensorflow.org/tutorials/generative/pix2pix

import time
import os
import random
import math
import json
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # suppresses plot
from IPython import display
from datetime import datetime

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


class p2p:
    def __init__(self, config):
        self.config = config
        self.config['global_batch_size'] = self.config['batch_size'] * strategy.num_replicas_in_sync
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
        img_list = [i for i in os.listdir(self.config['data']) if '.png' in i or '.jpeg' in i]
        random_img = ''.join(random.sample(img_list, 1))
        input, real = self.load(self.config['data'] + f'/{random_img}')
        # Casting to int for matplotlib to display the images
        plt.figure()
        plt.imshow(input / 255.0)
        plt.figure()
        plt.imshow(real / 255.0)

    def resize(self, input_image, real_image, height, width):
        '''
        :param input_image:
        :param real_image:
        :param height:
        :param width:
        :return:
        '''
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image

    def random_crop(self, input_image, real_image, height, width):
        '''
        :param input_image:
        :param real_image:
        :param height:
        :param width:
        :return:
        '''
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, height, width, 3])
        return cropped_image[0], cropped_image[1]

    # Normalizing the images to [-1, 1]
    def normalize(self, input_image, real_image):
        '''
        :param input_image:
        :param real_image:
        :return:
        '''
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        '''
        :param input_image:
        :param real_image:
        :return:
        '''
        # Resizing to height+30px by width+30px
        input_image, real_image = self.resize(input_image, real_image, self.config['img_size']+30, self.config['img_size']+30)

        # Random cropping back to height, width
        input_image, real_image = self.random_crop(input_image, real_image,  self.config['img_size'], self.config['img_size'])

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def process_images_train(self, image_file):
        '''
        Loads individual image, applies random jitter, normalizes image. Processing for train images only.
        :param image_file:
        :return:
        '''
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def process_images_pred(self, image_file):
        '''
        Loads individual image, resizes, normalizes image. Processing for test/pred images only.
        :param image_file:
        :return:
        '''
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.config['img_size'], self.config['img_size'])
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def image_pipeline(self, predict=False):
        '''
        :param predict: bool, whether or not to create train/test split. False treats all images as valid for prediction.
        :return:
            train - tf.distribute.DistributedDataset object
            test - tf.distribute.DistributedDataset (or None if predict=True)
        '''

        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents = [i for i in os.listdir(self.config['data']) if 'png' in i or 'jpg' in i]

        if predict:  # all images in `train` used for prediction
            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in contents])
            train = train.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.shuffle(self.config['buffer_size'])
            train = train.batch(self.config["global_batch_size"])
            test = None

        else:  # if train mode, break into train/test
            assert(self.config['test_examples'] > 0), "TEST_SIZE must be strictly > 0!"
            if self.config['test_examples'] > 1: # if int test samples
                test = random.sample(contents, self.config['test_examples'])
            else: # fraction
                test = random.sample(contents, math.ceil(len(contents) * self.config['test_examples']))
            train = [i for i in contents if i not in test]

            test = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in test])
            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in train])

            # process test images
            test = test.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            test = test.shuffle(self.config['buffer_size'])
            test = test.batch(self.config["global_batch_size"]).repeat()
            test = test.with_options(options)
            #test = iter(strategy.experimental_distribute_dataset(test))  # creates tf.distribute.DistributedDataset object

            # process training images
            train = train.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.shuffle(self.config['buffer_size'])
            train = train.batch(self.config["global_batch_size"]).repeat()
            train = train.with_options(options)
            #train = iter(strategy.experimental_distribute_dataset(train))

        return train, test

    def downsample(self, filters, size, apply_batchnorm=True):
        '''
        Builds encoder portion of pix2pix model.
        :param filters:
        :param size:
        :param apply_batchnorm:
        :return:
        '''
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        '''
        Builds decoder portion of pix2pix model.
        :param filters:
        :param size:
        :param apply_dropout:
        :return:
        '''
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        '''
        Define generator by combining down- and upsamplers.
        :return: tf.keras Model class
        '''

        with strategy.scope():

            inputs = tf.keras.layers.Input(shape=[256, 256, 3])

            down_stack = [
                self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
                self.downsample(128, 4),  # (batch_size, 64, 64, 128)
                self.downsample(256, 4),  # (batch_size, 32, 32, 256)
                self.downsample(512, 4),  # (batch_size, 16, 16, 512)
                self.downsample(512, 4),  # (batch_size, 8, 8, 512)
                self.downsample(512, 4),  # (batch_size, 4, 4, 512)
                self.downsample(512, 4),  # (batch_size, 2, 2, 512)
                self.downsample(512, 4),  # (batch_size, 1, 1, 512)
            ]

            up_stack = [
                self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
                self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
                self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
                self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
                self.upsample(256, 4),  # (batch_size, 32, 32, 512)
                self.upsample(128, 4),  # (batch_size, 64, 64, 256)
                self.upsample(64, 4),  # (batch_size, 128, 128, 128)
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

    def loss_object(self):
        '''
        :return:
        '''
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def optimizer(self):
        '''
        Optimizer for both generator and discriminators
        :return: tf.keras Adam optimizer
        '''
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # TODO - learning rate by batch size, if >GPU
        return optimizer

    def generator_loss(self, disc_generated_output, gen_output, target):
        '''
        Generator loss
        :param disc_generated_output:
        :param gen_output:
        :param target:
        :return:
        '''
        with strategy.scope():

            gan_loss = tf.reduce_sum(self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output))  * (1. / self.config['global_batch_size'])

            # Mean absolute error
            l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

            total_gen_loss = gan_loss + (100 * l1_loss) # 100=LAMBDA

        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self):
        '''
        Discrimator of pix2pix is a convolutional PatchGAN classifier - it tries to classify if each image patch
        is real or not real
        :return:
        '''

        with strategy.scope():
            initializer = tf.random_normal_initializer(0., 0.02)

            inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
            tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

            x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

            down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
            down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
            down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

            zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
            conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                          kernel_initializer=initializer,
                                          use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

            batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

            leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

            zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

            last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                          kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

            model = tf.keras.Model(inputs=[inp, tar], outputs=last)

        return model

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        '''
        Discriminator loss.
        :param disc_real_output:
        :param disc_generated_output:
        :return:
        '''
        with strategy.scope():

            real_loss = tf.reduce_sum(self.loss_obj(tf.ones_like(disc_real_output), disc_real_output))  * (1. / self.config['global_batch_size'])
            generated_loss = tf.reduce_sum(self.loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)) * (1. / self.config['global_batch_size'])

            total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    @tf.function
    def train_step(self, input_image, target, step, summary_writer):
        '''
        :param input_image:
        :param target:
        :param step:
        :param summary_writer:
        :return:
        '''

        # TODO - consider different numbers of generator or discriminator steps each time
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 100)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 100)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 100)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 100)

    def generate_images(self, model, test_input, tar, step, output_path):
        '''
        :param model:
        :param test_input:
        :param tar:
        :param step:
        :param output_path:
        :return:
        '''
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

        plot_path = os.path.join(output_path, 'test_images') # uses checkpoint manager path as contains strf datetime
        os.makedirs(plot_path, exist_ok=True) # dir should not exist
        plt.savefig(os.path.join(plot_path, f'step_{step}.png'), dpi=80)

    def fit(self, train_ds, test_ds, steps, summary_writer, output_path, checkpoint_manager=None, save_weights=True):
        '''
        :param train_ds:
        :param test_ds:
        :param steps:
        :param summary_writer:
        :param output_path: str, path to output test images across training steps
        :param checkpoint_manager:
        :param save_weights: bool, whether to save model weights per 5k training steps and at end, along with model checkpoints
        :return:
        '''

        print("\nTraining...\n", flush=True)

        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step % 1000 == 0) and (step > 0):
                display.clear_output(wait=True)

                print(f"Step: {step // 1000}k")
                print(f'\nCumulative training time: {time.time() - start:.2f} sec\n')

            self.train_step(input_image, target, step, summary_writer)

            # Training step
            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps and at end
            # Also saves generated training image
            if (step + 1) % 5000 == 0:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator, example_input, example_target, step, output_path)

            # At end save checkpoint and final test image
            if (step + 1) == self.config['steps']:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator, example_input, example_target, step, output_path)
                print(f'Cumulative training time at end of {step} steps: {time.time() - start:.2f} sec\n')

    def predict(self, pred_ds, output_path):
        '''
        :param pred_ds:
        :param output_path:
        :return:
        '''
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
            print('.', end='', flush=True)
            img_nr += 1
        print("\r")

def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--data', type=str, help='path to data', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per replica')
    parser.add_argument('--buffer-size', type=int, default=400, help='buffer size')
    parser.add_argument('--output-channels', type=int, default=3, help='number of color channels to output')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    group2 = parser.add_mutually_exclusive_group(required='--train' in sys.argv)
    group2.add_argument('--save-weights', action='store_true', help='save model checkpoints and weights')
    group2.add_argument('--no-save-weights', action='store_true', help='do not save model checkpoints or weights')
    parser.add_argument('--test-examples', type=int, default=5, help='number of test examples')
    parser.add_argument('--steps', type=int, default=10, help='number of training steps to take')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    opt = parser.parse_args()
    return opt


def main(opt):
    '''
    :param opt: argparse.Namespace
    :return: None
    '''

    # Directing output
    os.makedirs(opt.output, exist_ok=True)
    full_path = opt.output + '/' + datetime.now().strftime("%Y-%m-%d-%Hh%M")
    os.makedirs(full_path, exist_ok=True)  # will overwrite folder if model run within same minute

    # Log results
    log_dir = os.path.join(full_path, 'logs')
    os.makedirs(log_dir, exist_ok=False)  # dir should not exist, but just in case
    sys.stdout = open(os.path.join(log_dir, "Log.txt"), "w")
    sys.stderr = sys.stdout

    pix2pix = p2p(vars(opt))

    # Create or read from model checkpoints
    checkpoint = tf.train.Checkpoint(generator_optimizer=pix2pix.generator_optimizer,
                                     discriminator_optimizer=pix2pix.discriminator_optimizer,
                                     generator=pix2pix.generator,
                                     discriminator=pix2pix.discriminator)

    # Output config to logging dir
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(pix2pix.config, f)

    if opt.predict: # if predict mode
        prediction_dataset, _ = pix2pix.image_pipeline(predict=True)
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()
        pix2pix.predict(prediction_dataset, full_path)

    if opt.train: # if train mode
        train_dataset, test_dataset = pix2pix.image_pipeline(predict=False)

        # Outputting model checkpoints
        if opt.save_weights:
            checkpoint_dir = os.path.join(full_path, 'training_checkpoints')
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        else:
            manager = None

        # Summary witer file for tensorboard
        summary_writer = tf.summary.create_file_writer(log_dir)

        pix2pix.fit(train_ds=train_dataset,
                    test_ds=test_dataset,
                    steps=pix2pix.config['steps'],
                    summary_writer=summary_writer,
                    output_path=full_path,
                    checkpoint_manager=manager,
                    save_weights=pix2pix.config['save_weights'])

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
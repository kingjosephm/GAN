import time
import os
import random
import json
import sys
import argparse
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # suppresses plot
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
        self.generator_g = self.Generator(int(self.config['channels']), norm_type='instancenorm')
        self.generator_f = self.Generator(int(self.config['channels']), norm_type='instancenorm')
        self.discriminator_x = super().Discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = super().Discriminator(norm_type='instancenorm', target=False)
        self.generator_g_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.generator_f_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.discriminator_x_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.discriminator_y_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.model_metrics = {'X->Y Generator Loss': [],
                              'Y->X Generator Loss': [],
                              'Total Cycle Loss': [],
                              'Total X->Y Generator Loss': [],
                              'Total Y->X Generator Loss': [],
                              'Discriminator X Loss': [],
                              'Discriminator Y Loss': []}


    def random_crop(self, image, height, width):
        """
        :param image:
        :param height:
        :param width:
        :return:
        """
        return tf.image.random_crop(image, size=[height, width, int(self.config['channels'])])

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
            train - tf.python.data.ops.dataset_ops.BatchDataset
            test - tf.python.data.ops.dataset_ops.BatchDataset (or None if predict=True)
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
            test_X = test_X.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE).cache()
            test_X = test_X.shuffle(self.config['buffer_size'])
            test_X = test_X.batch(self.config["global_batch_size"])

            # process training images
            train_X = train_X.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE).cache()
            train_Y = train_Y.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE).cache()
            train_X = train_X.shuffle(self.config['buffer_size'])
            train_Y = train_Y.shuffle(self.config['buffer_size'])
            train_X = train_X.batch(self.config["global_batch_size"])
            train_Y = train_Y.batch(self.config["global_batch_size"])

            # Ensure channels align
            assert (train_X.element_spec.shape[-1] == int(self.config['channels'])), f"Mismatching number of channels between image and user argument! Number of channels in input image is {train_X.element_spec.shape[-1]}, while user specified {self.config['channels']} channels!"
            assert (train_Y.element_spec.shape[-1] == int(self.config['channels'])), f"Mismatching number of channels between image and user argument! Number of channels in target image is {train_Y.element_spec.shape[-1]}, while user specified {self.config['channels']} channels!"
            # Note - test_X, test_Y have None for channel dim at this point

        return train_X, train_Y, test_X

    def Generator(self, output_channels, norm_type='batchnorm'):
        """
        Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          output_channels: int, Output channels
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
                activation='tanh')  # (bs, 256, 256, n_channels)

            concat = tf.keras.layers.Concatenate()

            inputs = tf.keras.layers.Input(shape=[None, None, int(self.config['channels'])])
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
            return loss1 * self.config['lambda']

    def identity_loss(self, real_image, same_image):
        """
        :param real_image:
        :param same_image:
        :return:
        """
        with self.strategy.scope():
            loss = tf.reduce_mean(tf.abs(real_image - same_image))
            return self.config['lambda'] * 0.5 * loss

    def generate_images(self, model, test_input, image_nr, output_path, img_file_prefix='epoch'):
        """
        :param model:
        :param test_input:
        :param image_nr: int, either epoch number (train only) of image identifier number (predict mode)
        :param output_path:
        :param img_file_prefix: str, output image file suffix, whether 'epoch' (train) or 'img' (predict)
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
            if self.config['channels'] == '1':
                plt.imshow(display_list[i] * 0.5 + 0.5, cmap=plt.get_cmap('gray'))
            else:
                plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

        if img_file_prefix == 'epoch': # train mode, make subdir
            plot_path = os.path.join(output_path, 'test_images')
        else: # predict mode, don't make subdir
            plot_path = output_path
        os.makedirs(plot_path, exist_ok=True) # dir should not exist
        plt.savefig(os.path.join(plot_path, f"{img_file_prefix}_{image_nr}.png"), dpi=200)
        plt.close()

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
            tf.summary.scalar('X->Y Generator Loss', tf.reduce_sum(gen_g_loss), step=epoch)
            tf.summary.scalar('Y->X Generator Loss', tf.reduce_sum(gen_f_loss), step=epoch)
            tf.summary.scalar('Total Cycle Loss', tf.reduce_sum(total_cycle_loss), step=epoch)
            tf.summary.scalar('Total X->Y Generator Loss', tf.reduce_sum(total_gen_g_loss), step=epoch)
            tf.summary.scalar('Total Y->X Generator Loss', tf.reduce_sum(total_gen_f_loss), step=epoch)
            tf.summary.scalar('Discriminator X Loss', disc_x_loss, step=epoch)
            tf.summary.scalar('Discriminator Y Loss', disc_y_loss, step=epoch)

        return gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, \
               disc_x_loss, disc_y_loss

    def fit(self, train_X, train_Y, test_X, epochs, summary_writer, output_path, checkpoint_manager=None, save_weights=True):

        print("\nTraining...\n", flush=True)

        example_X = next(iter(test_X.take(1)))

        assert (example_X.shape[-1] == int(self.config['channels'])), f"Mismatching number of channels between image and user argument! Number of channels in input image is {example_X.shape[-1]}, while user specified {self.config['channels']} channels!"

        start = time.time()

        for epoch in range(epochs):

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((train_X, train_Y)):

                gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, \
                disc_x_loss, disc_y_loss = self.train_step(image_x, image_y, epoch, summary_writer)

                # Performance metrics from step into dict
                # Note - must be done outside self.train_step() as numpy operations do not work in tf.function
                self.model_metrics['X->Y Generator Loss'].append(tf.reduce_sum(gen_g_loss, axis=None).numpy().tolist())
                self.model_metrics['Y->X Generator Loss'].append(tf.reduce_sum(gen_f_loss, axis=None).numpy().tolist())
                self.model_metrics['Total Cycle Loss'].append(tf.reduce_sum(total_cycle_loss, axis=None).numpy().tolist())
                self.model_metrics['Total X->Y Generator Loss'].append(tf.reduce_sum(total_gen_g_loss, axis=None).numpy().tolist())
                self.model_metrics['Total Y->X Generator Loss'].append(tf.reduce_sum(total_gen_f_loss, axis=None).numpy().tolist())
                self.model_metrics['Discriminator X Loss'].append(tf.reduce_sum(disc_x_loss, axis=None).numpy().tolist())
                self.model_metrics['Discriminator Y Loss'].append(tf.reduce_sum(disc_y_loss, axis=None).numpy().tolist())

            # Every 5 epochs save weights and generate predicted image
            if (epoch + 1) % 5 == 0:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator_g, example_X, epoch+1, output_path)

            if (epoch + 1) == self.config['epochs']:
                if save_weights:
                    checkpoint_manager.save()
                self.generate_images(self.generator_g, example_X, epoch+1, output_path)

            print('Cumulative training duration at epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
            n += 1

    def predict(self, pred_ds, output_path):
        """
        :param pred_ds: tf.python.data.ops.dataset_ops.BatchDataset
        :param output_path: str, output path for image
        :return:
        """
        print("\nRendering images using pretrained weights\n")

        plot_path = os.path.join(output_path, 'prediction_images')
        os.makedirs(plot_path)

        img_nr = 0
        for image in pred_ds:

            assert (image.shape[-1] == int(self.config['channels'])), f"Mismatching number of channels between image and user argument! Number of channels in input image is {image.shape[-1]}, while user specified {self.config['channels']} channels!"

            # Output combined image
            self.generate_images(self.generator_g, image, img_nr, plot_path, img_file_prefix='img')

            # Just prediction image
            prediction = self.generator_g(image, training=True)
            plt.figure(figsize=(6, 6))
            if self.config['channels'] == '1':
                plt.imshow(prediction[0] * 0.5 + 0.5, cmap=plt.get_cmap('gray'))
            else:
                plt.imshow(prediction[0] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig(os.path.join(plot_path, f'prediction_{img_nr}.png'), dpi=200)
            plt.close()
            img_nr += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--input-images', type=str, help='path to input images', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per replica')
    parser.add_argument('--buffer-size', type=int, default=1000, help='buffer size')
    parser.add_argument('--channels', type=str, default='1', choices=['1', '3'], help='number of color channels to read in and output')
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
    parser.add_argument('--lambda', type=int, default=10, help='lambda parameter value')
    parser.add_argument('--learning-rate', type=float, default=0.0002, help='learning rate for Adam optimizer for generators and discriminators')
    parser.add_argument('--beta-1', type=float, default=0.5, help='exponential decay rate for 1st moment of Adam optimizer for generators and discriminators')
    parser.add_argument('--beta-2', type=float, default=0.999,  help='exponential decay rate for 2st moment of Adam optimizer for generators and discriminators')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    return parser.parse_args()

def make_fig(df, title, output_path):
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
    plt.title(f'CycleGAN {title}')
    os.makedirs(output_path, exist_ok=True)  # Creates output directory if not existing
    plt.savefig(os.path.join(output_path, f'{title}.png'), dpi=200)
    plt.close()

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
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()  # Note - if crashes here this b/c mismatch in channel size between weights and instantiated CycleGAN class
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

        # Output model metrics dict to log dir
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(cgan.model_metrics, f)

        # Output performance metrics figures
        for key in cgan.model_metrics.keys():
            df = pd.DataFrame(cgan.model_metrics[key]).reset_index()
            obs_per_epoch = len(df) / cgan.config['epochs']  # Number of data points per epoch = N_X-1
            df['epoch'] = ((df['index'] // obs_per_epoch) + 1).astype('int')
            agg = df.groupby('epoch')[0].mean()  # mean loss by epoch across obs
            make_fig(agg, title=key, output_path=os.path.join(full_path, 'figs'))

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
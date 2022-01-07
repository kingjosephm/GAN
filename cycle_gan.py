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
from utils import cyclegan_losses, make_fig
import numpy as np


"""
    CycleGAN in Tensorflow
    Credit:
        https://www.tensorflow.org/tutorials/generative/cyclegan
        https://github.com/tensorflow/examples/blob/d97aa060cb00ae2299b4b32591b8489df38e85ef/tensorflow_examples/models/pix2pix/pix2pix.py

"""

class CycleGAN(GAN):

    def __init__(self, config):
        super().__init__(config)
        self.generator_g = super().Generator(norm_type='instancenorm', shape=(None, None, int(self.config['channels'])))
        self.generator_f = super().Generator(norm_type='instancenorm', shape=(None, None, int(self.config['channels'])))
        self.discriminator_x = super().Discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = super().Discriminator(norm_type='instancenorm', target=False)
        self.generator_g_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.generator_f_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.discriminator_x_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.discriminator_y_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])


    def random_crop(self, image: tf.Tensor, height: int, width: int):
        """
        :param image: tf.Tensor, training image
        :param height: int, height
        :param width: int, width
        :return: tf.Tensor
        """
        return tf.image.random_crop(image, size=[height, width, int(self.config['channels'])])

    def random_jitter(self, image: tf.Tensor):
        """
        :param image: tf.Tensor, training image
        :return: tf.Tensor
        """
        # Resizing to height+30px by width+30px
        image = super().resize(image, self.config['img_size']+30, self.config['img_size']+30)

        # Random cropping back to height, width
        image = self.random_crop(image, self.config['img_size'], self.config['img_size'])

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def process_images_train(self, image_file: str):
        """
        Loads individual image, applies random jitter, normalizes image. Processing for train images only.
        :param image_file: str, full path to image
        :return: tf.Tensor
        """
        image = super().load(image_file, resize=True)
        image = self.random_jitter(image)
        image = super().normalize(image)
        return image

    def process_images_pred(self, image_file: str):
        """
        Loads individual image, resizes, normalizes image. Processing for test/pred images only.
        :param image_file: str, full path to image
        :return: tf.Tensor
        """
        image = super().load(image_file, resize=True)
        image = super().resize(image, self.config['img_size'], self.config['img_size'])
        image = super().normalize(image)
        return image

    def image_pipeline(self, predict: bool = False):
        """
        :param predict: bool, whether or not to create train/test split. False treats all images as valid for prediction.
        :return:
            train - tf.python.data.ops.dataset_ops.BatchDataset
            test - tf.python.data.ops.dataset_ops.BatchDataset (or None if predict=True)
        """

        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents_X = [i for i in os.listdir(self.config['input_images']) if 'png' in i or 'jpg' in i]
        assert contents_X, "No images found in input image directory!"

        if predict:  # all images in `train` used for prediction; they're not training images, only kept for consistency
            train_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in contents_X])
            train_X = train_X.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)

            test = None
            val_X = None
            val_Y = None
            train_Y = None

        else:  # if train mode, break into train/test
            contents_Y = [i for i in os.listdir(self.config['target_images']) if 'png' in i or 'jpg' in i]
            assert contents_Y, "No images found in target image directory!"

            random.seed(self.config['seed'])

            # Create subsets
            test = random.sample(contents_X, self.config['test_img'])

            val_obs_X = np.ceil((len(contents_X) - self.config['test_img']) * self.config['validation_size'])
            val_obs_Y = np.ceil(len(contents_Y) * self.config['validation_size'])
            val_X = random.sample([i for i in contents_X if i not in test], int(val_obs_X))
            val_Y = random.sample([i for i in contents_Y], int(val_obs_Y))

            train_X = [i for i in contents_X if i not in test and i not in val_X]
            train_Y = [i for i in contents_Y if i not in val_Y]

            # Convert to tf.Dataset
            test = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in test])
            val_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in val_X])
            val_Y = tf.data.Dataset.from_tensor_slices([self.config['target_images'] + '/' + i for i in val_Y])
            train_X = tf.data.Dataset.from_tensor_slices([self.config['input_images'] + '/' + i for i in train_X])
            train_Y = tf.data.Dataset.from_tensor_slices([self.config['target_images'] + '/' + i for i in train_Y])

            # Process each subset
            test = test.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            test = test.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

            val_X = val_X.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            val_Y = val_Y.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            val_X = val_X.shuffle(self.config['buffer_size'], reshuffle_each_iteration=True)
            val_Y = val_Y.shuffle(self.config['buffer_size'], reshuffle_each_iteration=True)
            val_X = val_X.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_Y = val_Y.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

            train_X = train_X.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train_Y = train_Y.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train_X = train_X.shuffle(self.config['buffer_size'], reshuffle_each_iteration=True)
            train_Y = train_Y.shuffle(self.config['buffer_size'], reshuffle_each_iteration=True)
            train_X = train_X.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)
            train_Y = train_Y.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_X, train_Y, val_X, val_Y, test

    def generator_loss(self, generated):
        """
        :param generated:
        :return:
        """
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        :param real_image:
        :param cycled_image:
        :return:
        """
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return loss1 * self.config['lambda']

    def identity_loss(self, real_image, same_image):
        """
        :param real_image:
        :param same_image:
        :return:
        """
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.config['lambda'] * 0.5 * loss

    def generate_images(self, model: tf.keras.Model, test_input: tf.Tensor, path_filename: str):
        """
        :param model: tf.keras.Model
        :param test_input: tf.Tensor
        :param path_filename: str, full path and file name including suffix
        :return: None
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
            plt.tight_layout()

        plt.savefig(path_filename, dpi=200)
        plt.close()

    @tf.function
    def train_step(self, real_x: tf.Tensor, real_y: tf.Tensor, training: bool = True):
        """
        :param real_x: tf.Tensor, input images
        :param real_y: tf.Tensor, target images
        :param training: bool, whether or not training mode
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

        if training:  # don't want to update weights if not training

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

        return gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, \
               disc_x_loss, disc_y_loss

    def fit(self, train_X: tf.Tensor, train_Y: tf.Tensor, val_X: tf.Tensor, val_Y: tf.Tensor, test: tf.Tensor,
            output_path: str, checkpoint_manager=None):

        print("\nTraining...\n", flush=True)

        test = next(iter(test.take(1)))  # returns batched dataset

        start = time.time()

        # Cost functions: average loss per epoch
        train_cost_functions = cyclegan_losses()
        val_cost_functions = cyclegan_losses()

        for epoch in range(self.config['epochs']):

            mini_batch_count = 1
            train_losses = cyclegan_losses()
            val_losses = cyclegan_losses()

            for image_x, image_y in tf.data.Dataset.zip((train_X, train_Y)):

                gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, \
                disc_x_loss, disc_y_loss = self.train_step(image_x, image_y)

                train_losses['X->Y Generator Loss'].append(gen_g_loss.numpy().tolist())
                train_losses['Y->X Generator Loss'].append(gen_f_loss.numpy().tolist())
                train_losses['Total Cycle Loss'].append(total_cycle_loss.numpy().tolist())
                train_losses['Total X->Y Generator Loss'].append(total_gen_g_loss.numpy().tolist())
                train_losses['Total Y->X Generator Loss'].append(total_gen_f_loss.numpy().tolist())
                train_losses['Discriminator X Loss'].append(disc_x_loss.numpy().tolist())
                train_losses['Discriminator Y Loss'].append(disc_y_loss.numpy().tolist())

                if mini_batch_count % 100 == 0:  # counter for every 100 mini-batches
                    print('.', end='', flush=True)

                mini_batch_count += 1

            # Append average of training loss functions per mini-batch to train cost function
            for key in train_losses.keys():
                train_cost_functions[key].append(sum(train_losses[key]) / len(train_losses[key]))

            # Evaluate using validation dataset
            for image_x, image_y in tf.data.Dataset.zip((val_X, val_Y)):

                gen_g_loss, gen_f_loss, total_cycle_loss, total_gen_g_loss, total_gen_f_loss, \
                disc_x_loss, disc_y_loss = self.train_step(image_x, image_y, False)

                val_losses['X->Y Generator Loss'].append(gen_g_loss.numpy().tolist())
                val_losses['Y->X Generator Loss'].append(gen_f_loss.numpy().tolist())
                val_losses['Total Cycle Loss'].append(total_cycle_loss.numpy().tolist())
                val_losses['Total X->Y Generator Loss'].append(total_gen_g_loss.numpy().tolist())
                val_losses['Total Y->X Generator Loss'].append(total_gen_f_loss.numpy().tolist())
                val_losses['Discriminator X Loss'].append(disc_x_loss.numpy().tolist())
                val_losses['Discriminator Y Loss'].append(disc_y_loss.numpy().tolist())

            # Append average of val loss functions per mini-batch to val cost function
            for key in val_losses.keys():
                val_cost_functions[key].append(sum(val_losses[key]) / len(val_losses[key]))

            # Make directory for output of test images
            test_img_path = output_path+'/test_images'
            os.makedirs(test_img_path, exist_ok=True)

            # Every 5 epochs save weights and generate predicted image
            if ((epoch + 1) % 5 == 0) and ((epoch + 1) != self.config['epochs']):
                if checkpoint_manager is not None:
                    checkpoint_manager.save()
                self.generate_images(self.generator_g, np.expand_dims(test[0], axis=0), path_filename=os.path.join(test_img_path,
                                                                                        f"epoch_{epoch+1}.png"))  # use first image of batched ds

            if (epoch + 1) == self.config['epochs']:
                if checkpoint_manager is not None:
                    checkpoint_manager.save()

            print(f'\nCumulative training duration at end of epoch {epoch + 1}: {(time.time() - start)/60:.2f} min')
            print(f"Train X->Y generator loss: {round(train_cost_functions['Total X->Y Generator Loss'][-1], 2)}, train discriminator X loss: {round(train_cost_functions['Discriminator X Loss'][-1], 2)}")
            print(f"Train Y->X generator loss: {round(train_cost_functions['Total Y->X Generator Loss'][-1], 2)}, train discriminator Y loss: {round(train_cost_functions['Discriminator Y Loss'][-1], 2)}")
            print(f"Val X->Y generator loss: {round(val_cost_functions['Total X->Y Generator Loss'][-1], 2)}, val discriminator X loss: {round(val_cost_functions['Discriminator X Loss'][-1], 2)}")
            print(f"Val Y->X generator loss: {round(val_cost_functions['Total Y->X Generator Loss'][-1], 2)}, val discriminator Y loss: {round(val_cost_functions['Discriminator Y Loss'][-1], 2)}\n")

        return train_cost_functions, val_cost_functions

    def predict(self, predict_ds: tf.Tensor, output_path: str):
        """
        :param predict_ds: tf.python.data.ops.dataset_ops.BatchDataset
        :param output_path: str, output path for image
        :return:
        """
        print("\nRendering images using pretrained weights\n")

        plot_path = os.path.join(output_path, 'prediction_images')
        os.makedirs(plot_path)

        # Output images
        img_counter = 0
        for i in predict_ds:
            self.generate_images(self.generator_g, np.expand_dims(i, axis=0),
                                 path_filename=plot_path + "/" + f"img{img_counter}.png")  # function expects batched data
            img_counter += 1


def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--input-images', type=str, help='path to input images', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--buffer-size', type=int, default=99999, help='buffer size')
    parser.add_argument('--channels', type=str, default='1', choices=['1', '3'], help='number of color channels to read in and output')
    parser.add_argument('--logging', type=str, default='true', choices=['true', 'false'], help='turn on/off script logging, e.g. for CLI debugging')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    parser.add_argument('--target-images', type=str, help='path to target images', required='--train' in sys.argv)
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train', required='--train' in sys.argv)
    parser.add_argument('--validation-size', type=float, default=0.1, help='validation set size as share of number of training images')
    parser.add_argument('--test-img', type=int, default=5, help='number of test images to sample')
    parser.add_argument('--save-weights', type=str, default='true', choices=['true', 'false'], help='save model checkpoints and weights')
    parser.add_argument('--lambda', type=int, default=10, help='lambda parameter value')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='learning rate for Adam optimizer for generators and discriminators')
    parser.add_argument('--beta-1', type=float, default=0.5, help='exponential decay rate for 1st moment of Adam optimizer for generators and discriminators')
    parser.add_argument('--beta-2', type=float, default=0.999,  help='exponential decay rate for 2st moment of Adam optimizer for generators and discriminators')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)

    args = parser.parse_args()

    assert (args.img_size == 256) or (args.img_size == 512), "img-size currently only supported for 256 x 256 or 512 x 512 pixels!"
    assert (args.validation_size > 0.0 and args.validation_size <= 0.3), "validation size is a proportion and bounded between 0-0.3!"
    assert (args.test_img >= 1), "test-img is an integer and must be >=1!"

    return args

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
    if opt.logging == 'true':
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

    if opt.predict:  # if predict mode
        prediction_dataset, _, _, _, _ = cgan.image_pipeline(predict=True)
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()  # Note - if crashes here this b/c mismatch in channel size between weights and instantiated CycleGAN class
        cgan.predict(prediction_dataset, full_path)

    if opt.train:  # if train mode
        train_X, train_Y, val_X, val_Y, test = cgan.image_pipeline(predict=False)

        # Outputting model checkpoints
        if opt.save_weights == 'true':
            checkpoint_dir = os.path.join(full_path, 'training_checkpoints')
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        else:
            manager = None

        train_metrics, val_metrics = cgan.fit(train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y,
                                              test=test, output_path=full_path, checkpoint_manager=manager)

        # Output final test images
        final_test_imgs = full_path+'/final_test_imgs'
        os.makedirs(final_test_imgs, exist_ok=False)

        img_counter = 0
        for i in test.unbatch():
            cgan.generate_images(cgan.generator_g, np.expand_dims(i, axis=0), final_test_imgs + "/" + f"img{img_counter}.png")  # function expects batched data
            img_counter += 1

        # Output model metrics dict to log dir
        with open(os.path.join(log_dir, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f)
        with open(os.path.join(log_dir, 'val_metrics.json'), 'w') as f:
            json.dump(val_metrics, f)

        # Output performance metrics figures
        for key in train_metrics.keys():
            tr = pd.DataFrame(train_metrics[key]).reset_index()
            va = pd.DataFrame(val_metrics[key]).reset_index()

            # non-zero-based epoch index
            tr['index'] = tr['index'] + 1
            tr = tr.set_index('index')

            va['index'] = va['index'] + 1
            va = va.set_index('index')

            make_fig(tr, va, title='CycleGAN ' + key, output_path=os.path.join(full_path, 'figs'))

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
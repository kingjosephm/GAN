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
from utils import pix2pix_losses, make_fig
import numpy as np

"""
    Pix2Pix in Tensorflow
    Credit:
        https://www.tensorflow.org/tutorials/generative/pix2pix
        https://github.com/tensorflow/examples/blob/d97aa060cb00ae2299b4b32591b8489df38e85ef/tensorflow_examples/models/pix2pix/pix2pix.py

"""


class Pix2Pix(GAN):
    def __init__(self, config):
        super().__init__(config)
        self.generator = super().Generator(shape=(self.config['img_size'], self.config['img_size'], int(self.config['channels'])))
        self.discriminator = super().Discriminator(target=True)
        self.generator_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])
        self.discriminator_optimizer = super().optimizer(learning_rate=self.config['learning_rate'], beta_1=self.config['beta_1'], beta_2=self.config['beta_2'])

    def split_img(self, image_file: str):
        """
        :param image_file: str, full path to image file
        :return:
            input_image - input (thermal) image, tensorflow.python.framework.ops.EagerTensor
            real_image - target (real) image, tensorflow.python.framework.ops.EagerTensor
        """
        image = super().load(image_file, resize=False)

        # Split each image tensor into two tensors:
        w = tf.shape(image)[1]
        w = w // 2

        if self.config['input_img_orient'] == 'left':
            input_image = image[:, :w, :]
            real_image = image[:, w:, :]
        else:
            input_image = image[:, w:, :]
            real_image = image[:, :w, :]

        return input_image, real_image

    def random_crop(self, input_image: tf.Tensor, real_image: tf.Tensor, height: int, width: int):
        """
        :param input_image: tf.Tensor, thermal image
        :param real_image: tf.Tensor, visible grayscale image
        :param height: int, image height
        :param width: int, image width
        :return: stacked tf.Tensor
        """
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, height, width, int(self.config['channels'])])
        return cropped_image[0], cropped_image[1]

    @tf.function()
    def random_jitter(self, input_image: tf.Tensor, real_image: tf.Tensor):
        """
        :param input_image: tf.Tensor, thermal image
        :param real_image: tf.Tensor, visible grayscale image
        :return: tf.Tensor (2)
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

    def process_images_train(self, image_file: str):
        """
        Loads matched image pair, applies random jitter, normalizes images.
        :param image_file: str, full path to thermal image
        :return: tf.Tensor (2)
        """
        input_image, real_image = self.split_img(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image = super().normalize(input_image)
        real_image = super().normalize(real_image)
        return input_image, real_image

    def process_images_pred(self, image_file: str):
        """
        Loads matched image pair, *no augmentation*, resizes, normalizes image. Processing for test/pred images only.
        :param image_file: str, full path to visible image
        :return: tf.Tensor (2)
        """
        input_image, real_image = self.split_img(image_file)
        input_image = super().resize(input_image, self.config['img_size'], self.config['img_size'])
        real_image = super().resize(real_image, self.config['img_size'], self.config['img_size'])
        input_image = super().normalize(input_image)
        real_image = super().normalize(real_image)
        return input_image, real_image

    def image_pipeline(self, predict: bool = False):
        """
        :param predict: bool, whether or not to create train/test split. False treats all images as valid for prediction.
        :return:
            train - tf.distribute.DistributedDataset object
            test - tf.distribute.DistributedDataset (or None if predict=True)
        """

        print("\nReading in and processing images.\n", flush=True)

        # list of images in dir
        contents = [i for i in os.listdir(self.config['data']) if 'png' in i or 'jpg' in i]
        assert contents, "No images found in data directory!"

        if predict:  # all images in `train` used for prediction; they're not training images, only kept for consistency

            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in contents])
            train = train.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)

            val = None
            test = None

        else:  # if train mode, break into train/val subsets

            random.seed(self.config['seed'])

            # Get subsets
            test = random.sample(contents, self.config['test_img'])

            val_obs = np.ceil((len(contents)-self.config['test_img']) * self.config['validation_size'])
            val = random.sample([i for i in contents if i not in test], int(val_obs))

            train = [i for i in contents if i not in test and i not in val]
            train = random.sample(train, len(train))  # shuffle in lieu of tf.data.shuffle

            # Convert to tf.Dataset
            test = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in test])
            val = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in val])
            train = tf.data.Dataset.from_tensor_slices([self.config['data'] + '/' + i for i in train])

            # Process each subset
            test = test.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            test = test.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

            val = val.map(self.process_images_pred, num_parallel_calls=tf.data.AUTOTUNE)
            val = val.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

            # Process training images
            train = train.map(self.process_images_train, num_parallel_calls=tf.data.AUTOTUNE)
            train = train.batch(self.config["batch_size"]).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train, val, test

    def generator_loss(self, disc_generated_output, gen_output, target, input_image):
        """
        Generator loss
        :param disc_generated_output:
        :param gen_output:
        :param target:
        :param input_image:
        :return:
        """

        gan_loss = self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)

        if self.config['generator_loss']=='l1':
            # Mean absolute error
            gan_loss2 = tf.reduce_mean(tf.abs(target - gen_output))
        else:  # ssim
            # SSIM loss, see https://www.tensorflow.org/api_docs/python/tf/image/ssim
            gan_loss2 = tf.image.ssim(input_image, target, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

        total_gen_loss = gan_loss + (self.config['lambda'] * gan_loss2)

        return total_gen_loss, gan_loss, gan_loss2

    @tf.function
    def train_step(self, input_image: tf.Tensor, target: tf.Tensor, training: bool = True):
        """
        :param input_image: tf.Tensor, thermal image
        :param target: tf.Tensor, grayscale true image
        :param training: bool, whether or not training mode
        :return:
        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_gan_loss2 = self.generator_loss(disc_generated_output, gen_output, target, input_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output, 0.5)  # TF example lacked, but original code divides by 2, see https://github.com/phillipi/pix2pix/blob/89ff2a81ce441fbe1f1b13eca463b87f1e539df8/train.lua#L254

        if training:  # don't want to update weights if not training

            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                    self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                        self.discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_gan_loss2, disc_loss

    def generate_images(self, model: tf.keras.Model, test_input: np.array, tar: np.array, path_filename: str):
        """
        :param model:
        :param test_input: array-like, tf.Tensor or numpy array, batched dataset of generated images
        :param tar: array-like, tf.Tensor or numpy array, batched dataset of true images
        :param path_filename: str, full path and file name including suffix
        :return: None
        """
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 6))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            if self.config['channels'] == '1':
                plt.imshow(display_list[i] * 0.5 + 0.5, cmap=plt.get_cmap('gray'))
            else:
                plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            plt.tight_layout()

        plt.savefig(path_filename, dpi=200)
        plt.close()

    def fit(self, train_ds: tf.Tensor, val_ds: tf.Tensor, test_ds: tf.Tensor, output_path: str, checkpoint_manager=None):
        """
        :param train_ds: tf.Tensor, training dataset
        :param val_ds: tf.Tensor, validation dataset
        :param test_ds: tf.Tensor, test dataset
        :param output_path: str, path to output test images across training epochs
        :param checkpoint_manager: tf.train.CheckpointManager or None
        :return:
        """

        print("\nTraining...\n", flush=True)

        example_input, example_target = next(iter(test_ds.take(1)))  # returns batched dataset
        start = time.time()

        # Cost functions: average loss per epoch
        train_cost_functions = pix2pix_losses()
        val_cost_functions = pix2pix_losses()

        for epoch in range(self.config['epochs']):

            mini_batch_count = 1
            train_losses = pix2pix_losses()
            val_losses = pix2pix_losses()

            for step, (input_image, target) in enumerate(train_ds):  # each step is a mini-batch
                gen_total_loss, gen_gan_loss, gen_gan_loss2, disc_loss = self.train_step(input_image, target, True)

                train_losses['Generator Total Loss'].append(gen_total_loss.numpy().tolist())
                train_losses['Generator Loss (Primary)'].append(gen_gan_loss.numpy().tolist())
                train_losses['Generator Loss (Secondary)'].append(gen_gan_loss2.numpy().tolist())
                train_losses['Discriminator Loss'].append(disc_loss.numpy().tolist())

                if mini_batch_count % 100 == 0:  # counter for every 100 mini-batches per epoch
                    print('.', end='', flush=True)

                mini_batch_count += 1

            # Append average of training loss functions per mini-batch to train cost function
            for key in train_losses.keys():
                train_cost_functions[key].append(sum(train_losses[key]) / len(train_losses[key]))

            # Evaluate using validation dataset
            for step, (input_image, target) in enumerate(val_ds):
                gen_total_loss, gen_gan_loss, gen_gan_loss2, disc_loss = self.train_step(input_image, target, False)

                val_losses['Generator Total Loss'].append(gen_total_loss.numpy().tolist())
                val_losses['Generator Loss (Primary)'].append(gen_gan_loss.numpy().tolist())
                val_losses['Generator Loss (Secondary)'].append(gen_gan_loss2.numpy().tolist())
                val_losses['Discriminator Loss'].append(disc_loss.numpy().tolist())

            # Append average of val loss functions per mini-batch to val cost function
            for key in val_losses.keys():
                val_cost_functions[key].append(sum(val_losses[key]) / len(val_losses[key]))

            # Make directory for output of test images
            test_img_path = output_path+'/test_images'
            os.makedirs(test_img_path, exist_ok=True)

            # Every 5 epochs save weights and generate predicted image(s)
            if ((epoch + 1) % 5 == 0) and ((epoch + 1) != self.config['epochs']):
                if checkpoint_manager is not None:
                    checkpoint_manager.save()
                self.generate_images(self.generator, np.expand_dims(example_input[0], axis=0),
                                     np.expand_dims(example_target[0], axis=0),
                                     path_filename=os.path.join(test_img_path, f"epoch_{epoch+1}.png"))  # use first image of batched ds

            if (epoch + 1) == self.config['epochs']:
                if checkpoint_manager is not None:
                    checkpoint_manager.save()

            print(f'\nCumulative training duration at end of epoch {epoch + 1}: {(time.time() - start)/60:.2f} min')
            print(f"Train generator loss: {round(train_cost_functions['Generator Total Loss'][-1], 2)}, train discriminator loss: {round(train_cost_functions['Discriminator Loss'][-1], 2)}")
            print(f"Val generator loss: {round(val_cost_functions['Generator Total Loss'][-1], 2)}, val discriminator loss: {round(val_cost_functions['Discriminator Loss'][-1], 2)}\n")

        return train_cost_functions, val_cost_functions

    def predict(self, predict_ds: tf.Tensor, output_path: str):
        """
        Generates generated PNG images using supplied image dataset.
        :param predict_ds: tensorflow.python.data.ops.dataset_ops.ParallelMapDataset
        :param output_path: str, output path
        :return: None
        """
        plot_path = os.path.join(output_path, 'prediction_images')
        os.makedirs(plot_path, exist_ok=False)

        # Output images
        img_counter = 0
        for i in predict_ds:
            self.generate_images(self.generator, np.expand_dims(i[0], axis=0), np.expand_dims(i[1], axis=0), plot_path + "/" + f"img{img_counter}.png")  # function expects batched data
            img_counter += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    # Needed in all cases
    parser.add_argument('--data', type=str, help='path to data', required=True)
    parser.add_argument('--output', type=str, help='path to output results', required=True)
    parser.add_argument('--img-size', type=int, default=256, help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per replica')
    parser.add_argument('--buffer-size', type=int, default=99999, help='buffer size')
    parser.add_argument('--channels', type=str, default='1', choices=['1', '3'], help='number of color channels to read in and output')
    parser.add_argument('--logging', type=str, default='true', choices=['true', 'false'], help='turn on/off script logging, e.g. for CLI debugging')
    parser.add_argument('--generator-loss', type=str, default='l1', choices=['l1', 'ssim'], help='combined generator loss function')
    parser.add_argument('--input-img-orient', type=str, default='left', choices=['left', 'right'], help='whether input image is on left (i.e. target right) or vice-versa')
    parser.add_argument('--seed', type=int, default=123, help='seed value for random number generator')
    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train model using data')
    group.add_argument('--predict', action='store_true', help='use pretrained weights to make predictions on data')
    # Train params
    parser.add_argument('--save-weights', type=str, default='true', choices=['true', 'false'], help='save model checkpoints and weights')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train', required='--train' in sys.argv)
    parser.add_argument('--lambda', type=int, default=100, help='lambda value for secondary generator loss (L1)')
    parser.add_argument('--validation-size', type=float, default=0.1, help='validation set size as share of number of training images')
    parser.add_argument('--test-img', type=int, default=5, help='number of test images to sample')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='learning rate for Adam optimizer for generator and discriminator')
    parser.add_argument('--beta-1', type=float, default=0.5, help='exponential decay rate for 1st moment of Adam optimizer for generator and discriminator')
    parser.add_argument('--beta-2', type=float, default=0.999,  help='exponential decay rate for 2st moment of Adam optimizer for generator and discriminator')
    # Predict param
    parser.add_argument('--weights', type=str, help='path to pretrained model weights for prediction',
                        required='--predict' in sys.argv)
    args = parser.parse_args()

    # Verify image size
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
    os.makedirs(log_dir, exist_ok=False)
    if opt.logging == 'true':
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
        prediction_dataset, _, _ = p2p.image_pipeline(predict=True)
        checkpoint.restore(tf.train.latest_checkpoint(opt.weights)).expect_partial()  # Note - if crashes here this b/c mismatch in channel size between weights and instantiated Pix2Pix class
        p2p.predict(prediction_dataset, full_path)

    if opt.train: # if train mode
        train, validation, test = p2p.image_pipeline(predict=False)

        # Outputting model checkpoints
        if opt.save_weights == 'true':
            checkpoint_dir = os.path.join(full_path, 'training_checkpoints')
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        else:
            manager = None

        train_metrics, val_metrics = p2p.fit(train_ds=train, val_ds=validation, test_ds=test, output_path=full_path,
                                             checkpoint_manager=manager)

        # Output final test images
        final_test_imgs = full_path+'/final_test_imgs'
        os.makedirs(final_test_imgs, exist_ok=False)

        img_counter = 0
        for i in test.unbatch():
            p2p.generate_images(p2p.generator, np.expand_dims(i[0], axis=0), np.expand_dims(i[1], axis=0), final_test_imgs + "/" + f"img{img_counter}.png")  # function expects batched data
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

            make_fig(tr, va, title='Pix2Pix ' + key, output_path=os.path.join(full_path, 'figs'))

    print("Done.")

if __name__ == '__main__':

    opt = parse_opt()
    main(opt)
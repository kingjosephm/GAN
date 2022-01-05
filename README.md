# Introduction
The code in this repository aims to develop a generative adversarial network (GAN) model to convert thermal-spectrum images to the visible spectrum. To accomplish this, we curate an image training dataset of matched thermal and visible images. This cGAN model will serve as the *first model* in a three-part image classification pipeline of motor vehicle makes and models: 1) images are output from a thermal camera and supplied to the trained cGAN model for conversion to the visible spectrum; 2) the [YOLOv5 algorithm](https://github.com/ultralytics/yolov5) is used on converted visible images to generate bounding box coordinates around any passenger motor vehicles present in the image; 3) images are cropped to the YOLOv5 bounding box area and the make-model of the vehicle is classified using a [second model](https://github.boozallencsn.com/MERGEN/vehicle_make_model_classifier). A mockup of this workflow can be found in the [vehicle_image_pipeline repository](https://github.boozallencsn.com/MERGEN/vehicle_image_pipeline). Because of a dearth of adequately-sized, representative labeled thermal images of passenger vehicles, we train the vehicle make-model classifier model using labeled RGB images. For this reason, an upstream model is needed to convert thermal images to the visible spectrum prior to classification.


# Document contents and intended audience
This document aims to serve four purposes. First, describe the code contained in this repository, how it was run on Booz Allen hardware, how to possibly run it on other hardware, and general performance expectations. Second, offer a brief explanation of the theory behind the code and the code relates to this theory.  Third, detail the training date used by this code and how they were obtained. Fourth, offer a summary of the existing results, prior experiments, outstanding issues, and potential future uses of the code. This document is written for a semi-technical audience that is presumed to have (some) familiarity with the Python language, deep neural networks, and computer vision models. We also aim to relate this repository to other repositories and efforts within the MERGEN project.


# Repository branches
- `main` **[this branch]**: implements Pix2Pix and CycleGAN models in TensorFlow
- `pytorch`: implements Pix2Pix and CycleGAN models using PyTorch, code developed by [Jn-Yan Zhu and colleagues](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- `cycle_gan_inline_plts`: outdated, makes inline plots for matched images for CycleGAN


# Pix2Pix and CycleGAN
Originally developed by [Isola et al. (2017)](https://arxiv.org/abs/1611.07004), **Pix2Pix** is a generative adversarial network (GAN) created for general purpose image-to-image translation. As explained by [Jason Brownlee](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/), "[t]he GAN architecture is comprised of a generator model for outputting new plausible synthetic images, and a discriminator model that classifies images as real (from the dataset) or fake (generated). The discriminator model is updated directly, whereas the generator model is updated via the discriminator model. As such, the two models are trained simultaneously in an adversarial process where the generator seeks to better fool the discriminator and the discriminator seeks to better identify the counterfeit images. The Pix2Pix model is a type of conditional GAN, or cGAN, where the generation of the output image is conditional on an input, in this case, a source image. The discriminator is provided both with a source image and the target image and must determine whether the target is a plausible transformation of the source image. The generator is trained via adversarial loss, which encourages the generator to generate plausible images in the target domain. The generator is also updated via L1 loss measured between the generated image and the expected output image. This additional loss encourages the generator model to create plausible translations of the source image. The Pix2Pix GAN has been demonstrated on a range of image-to-image translation tasks such as converting maps to satellite photographs, black and white photographs to color, and sketches of products to product photographs."

Whereas Pix2Pix relies on *paired images* (i.e. an image in one domain and an aligned matching image in another domain), **CycleGAN** is an image-to-image translation model for *unpaired images*. Developed by [Jun-Yan Zhu et al. (2017)](https://arxiv.org/abs/1703.10593), CycleGAN is advantageous when paired examples don't exist or are challenging to gather. CycleGAN uses the same GAN architecture as Pix2Pix, however, it trains two generators and two discriminators simultaneously. "One generator takes images from the first domain as input and outputs images for the second domain, and the other generator takes images from the second domain as input and generates images for the first domain. Discriminator models are then used to determine how plausible the generated images are and update the generator models accordingly. This extension alone might be enough to generate plausible images in each domain, but not sufficient to generate translations of the input images. [...] The CycleGAN uses an additional extension to the architecture called cycle consistency. This is the idea that an image output by the first generator could be used as input to the second generator and the output of the second generator should match the original image. The reverse is also true: that an output from the second generator can be fed as input to the first generator and the result should match the input to the second generator. Cycle consistency is a concept from machine translation where a phrase translated from English to French should translate from French back to English and be identical to the original phrase. The reverse process should also be true. [...] The CycleGAN encourages cycle consistency by adding an additional loss to measure the difference between the generated output of the second generator and the original image, and the reverse. This acts as a regularization of the generator models, guiding the image generation process in the new domain toward image translation" ([Brownlee 2019](https://machinelearningmastery.com/what-is-cyclegan/)).


# Data
We train our GAN models using paired thermal and visible images from [Teledyne FLIR](https://www.flir.com/oem/adas/adas-dataset-form/). FLIR matched images are developed primarily for object detection exercises, as they include object bounding boxes and labels. However, we discard this information as it is not relevant to the current effort. FLIR offers a [free version](https://www.flir.com/oem/adas/adas-dataset-form/) of their matched thermal-visible image data, which comprises 14,452 images taken in and around Santa Barbara, California. We use this dataset and combine it with FLIR's proprietary Europe dataset, which contains 14,353 matched images captured in London, Paris, Madrid, and several other Spanish cities. Both California and European original FLIR images are located on the MERGEN OneDrive Data folder in a directory titled `FLIR_ADAS_DATASET`.

FLIR raw visible (i.e. RGB spectrum) images are 1024h x 1224w pixels, whereas thermal raw images are 512h x 640w pixels. Visible images were produced with a camera with a wider field-of-view and so alignment is necessary for (at least) the Pix2Pix model. The script, `./create_training_imgs/curate_FLIR_data.py`, pairs matched thermal and visible images, and aligns these to the overlapping portion of the visible image. Both thermal and RGB images are converted to grayscale (1-channel) and concatenated horizontally with the thermal on the left, visible on the right. These are stored on the MERGEN OneDrive Data folder in `FLIR_matched_rgb_thermal.zip`. In some cases, a matched image was not found, or it was corrupt. Correspondingly, the total number of matched thermal-visible images totals **28,279**. The script, `./create_training_imgs/separate_FLIR_data.py`, takes as input the concatenated matched images and separates them into child thermal and visible directories. Matched thermal and visible images retain the same file name as the original image for linkage purposes. These are stored on OneDrive's MERGEN Data folder in `FLIR_separated.zip`. Each curated FLIR image is 512h x 640w pixels (1-channel), which must be resized by both GAN models to 256x256 or 512x512 on read-in.

Other data have also been previously used in this project. These data were collected by hand during the daytime only in local parking lots around the DC-Maryland-Virgina area and generally feature closely-cropped images of passenger vehicles. The originals of these images are stored on OneDrive in `original_thermal_visible_GAN_images.zip`. Concatenated thermal-visible images, not perfectly aligned, are found on OneDrive in `curated_thermal_visible_GAN_images.zip`. These images are each 512h x 640w pixels.

# GPU cluster & Docker
### Data
Concatenated, matched thermal-visible images are stored on the GPU cluster under `/home/kingj/FLIR_matched_gray_thermal`, while the separated versions are stored under `/home/kingj/FLIR_separated`.

These are stored in the following Docker volumes:

    FLIR_data_matched <- concatenated thermal/visible images
    FLIR_data_separated <- thermal and visible images separated into child subdirectories

### Model output
Output from a given model run is stored in the following Docker volume:

    MERGEN_FLIR_output

Each run is put into a child directory according to the naming convention *YYYY-MM-DD-HHhmm*, where this datetime denotes the start execution datetime of the code.

### Docker image
For this code, as well as the make-model classifier, the following Docker image was used:

    king0759/tf2_gpu_jupyter_mpl:v3

This (admittedly bloated) Docker image contains the packages listed in `requirements.txt`. Not all of the packages listed in this requirements file are strictly necessary for the code in this repository though.

# Train model using Docker (detached)
### Example 1: using Pix2Pix script
    
    docker run -it \
        --name p2p \
        -d --rm \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=FLIR_data_matched,target=/data \
        --mount source=MERGEN_FLIR_output,target=/output \
        --gpus device=GPU-3c51591d-cfdb-f87c-ece8-8dcfdc81e67a \
        king0759/tf2_gpu_jupyter_mpl:v3 python3 ./scripts/pix2pix.py \
        --train --data=data --output=output --lambda=100 --epochs=200 \
        --batch-size=8 --logging='true' --save-weights='true'

### Example 2: using CycleGAN script

    docker run -it \
        --name cgan \
        -d --rm \
        --mount type=bind,source=/home/kingj/scripts,target=/scripts \
        --mount source=FLIR_data_separated,target=/data \
        --mount source=MERGEN_FLIR_output,target=/output \
        --gpus device=GPU-0c5076b3-fe4a-0cd8-e4b7-71c2037933c0 \
        king0759/tf2_gpu_jupyter_mpl:v3 python3 ./scripts/cycle_gan.py \
        --train --input-images=data/therm --target-images=data/vis \
        --output=output --save-weights='true' --logging='true' \
        --epochs=50 --img-size=256 --batch-size=16

Both examples above presume the Pix2Pix or CycleGAN scripts you want to run are located in the directory at `/home/kingj/scripts`. Change this directory path as needed. The shell script `driver.sh` automates the copying and uploading of key scripts for this repository to the GPU cluster. Also see `HELPME.md` for generic Docker and Linux tips.
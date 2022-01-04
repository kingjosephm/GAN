# Introduction
This code develops a dataset of matched thermal and visible images and two conditional generative adversarial network (cGAN) models to convert thermal-spectrum images to the visible spectrum. The data originate from [Teledyne FLIR](https://www.flir.com/oem/adas/adas-dataset-form/) (see extended description below). The cGAN models, a [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix) model and [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan) model, are based on Tensorflow's excellent tutorial for each model. This model will serve as the first model in a three-part ML image pipeline: 1) images are output from a thermal camera and supplied to the trained cGAN model for conversion to the visible spectrum; 2) the YOLOv5 algorithm is deployed on converted visible images to generate bounding box coordinates around any passenger motor vehicles present in the image; 3) images are cropped to the YOLOv5 bounding box area and the make-model of the vehicle is classified using a [second model](https://github.boozallencsn.com/MERGEN/vehicle_make_model_classifier).


# Data
FLIR matched thermal-visible data are developed primarily for object detection exercises, as they include object bounding boxes and labels. However, we discard this information as it is not relevant to the current code. FLIR offers a [free version](https://www.flir.com/oem/adas/adas-dataset-form/) of their matched thermal-visible image data, which comprises 14,452 images taken in and around Santa Barbara, California. We use this dataset and combine it with FLIR's Europe dataset, which contains 14,353 matched images captured in London, Paris, Madrid, and several other Spanish cities. Both California and European original FLIR images are located on the MERGEN OneDrive Data folder in a directory titled `FLIR_ADAS_DATASET`.

Scripts located in this repository under `./create_training_imgs` curate these raw images. Specifically, `curate_FLIR_data.py` pairs matched thermal and visible images, and aligns these to the overlapping portion of the visible image. Matched images are converted from RGB to grayscale and concatenated horizontally with the thermal on the left, visible on the right. These are stored on the MERGEN OneDrive data folder in `FLIR_matched_rgb_thermal.zip`. Visible images are 1024h x 1224w pixels and were produced with a camera with a wider field-of-view than the 512h x 640w pixel thermal images, so alignment is necessary for (at least) the Pix2Pix model. In some cases, a matched image was not found, or it was corrupt. Correspondingly, the total number of matched thermal-visible images totals **28,279**. The script `separate_FLIR_data.py` takes as input the concatenated matched images and separates them into child thermal and visible directories. Matched thermal and visible images retain the same file name as the original image for linkage purposes. These are stored on OneDrive in `FLIR_separated.zip`. Each curated FLIR image is 512h x 640w pixels, which must be resized by both cGAN models to 256x256 or 512x512 on read-in.

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
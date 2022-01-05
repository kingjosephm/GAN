# Introduction
This Git branch contains code to run a Pix2Pix model using the code developed by [Jun-Yan Zhu and colleagues](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This code uses PyTorch, rather than TensorFlow as in the `main` branch of this repository. This code also expects a different data structure than the TensorFlow `main` branch. For a fuller discussion of the data and object of this project see the README in the `main` branch.

# Data
Image data for this branch are the same as `main`; however, images are organized into child 'train', 'val', and 'test' directories. The script used to randomly select files and move them into these subdirectory is `create_torch_img_dataset.py`. These files can be found on the MERGEN OneDrive in `FLIR_Matched_gray_thermal_pytorch.zip`. 

# GPU cluter & Docker
### Data 
Unzipped image data can be found on the GPU cluster in `/home/kingj/FLIR_matched_gray_thermal_pytorch`.

These data are located in the following Docker volume:
    
    FLIR_data_matched_pytorch

### Model output
Output from a given model run is stored in the following Docker volume:

    MERGEN_FLIR_output

Each run is put into a child directory according to the naming convention YYYY-MM-DD-HHhmm, where this datetime denotes the start execution datetime of the code.

### Docker image
For this code, as well as the make-model classifier, the following Docker image was used:

    king0759/tf2_gpu_jupyter_mpl:v3

# Train model within Docker container (interactive)
The shell script, `torch_driver.sh`, should be called from command line with the following parameters:

    -n      -> Number of epochs to train at initial learning rate
    -d      -> Number of epochs to train with LR decay
    -s      -> Frequency (# per epoch) to save model

# Train model using Docker (detached)

    docker run -it \
    --name p2p_torch --rm -d \
    --mount type=bind,source=/home/kingj/scripts,target=/scripts \
    --mount source=FLIR_data_matched_pytorch,target=/data \
    --mount source=MERGEN_FLIR_output,target=/output \
    --gpus device=GPU-0c5076b3-fe4a-0cd8-e4b7-71c2037933c0 \
    king0759/tf2_gpu_jupyter_mpl:v3 \
    sh ./scripts/torch_driver.sh -n 50 -d 50 -s 25
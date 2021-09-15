#!/bin/bash

# TODO - create own docker image to obviate this step
# Install any missing packages from container image
pip install -r ./scripts/requirements.txt

# TODO - transform below to accept CLI input, removing config file
python3 ./scripts/tf_pix2pix.py
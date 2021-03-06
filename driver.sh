#!/bin/bash

# Save credentials so don't have to enter twice
read -p "Welcome, please enter credentials: " credentials

# Move updated script onto server, replace old scripts in ./scripts dir
echo "Uploading scripts to server, please enter password"
scp pix2pix.py cycle_gan.py base_gan.py utils.py requirements.txt "$credentials":scripts # also prompts password
echo "Done uploading scripts. Now SSH into server, which requires your password again"

# SSH into GPU server
ssh $credentials # will again prompt password
#!/bin/bash

mkdir DIV2K
cd DIV2K

# Training.
echo "Downloading DIV2K train dataset."
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip
mv DIV2K_train_HR train
rm DIV2K_train_HR.zip

# Testing.
echo "Download Set5/Set14 test dataset see README.md#Test"

#!/bin/bash

wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
unzip DIV2K_train_LR_bicubic_X4.zip
mv DIV2K_train_LR_bicubic/X4 ./
rm -rf DIV2K_train_LR_bicubic
rm DIV2K_train_LR_bicubic_X4.zip

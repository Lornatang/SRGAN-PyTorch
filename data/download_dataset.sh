#!/bin/bash

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

mv DIV2K_train_HR train
mv DIV2K_valid_HR val

rm DIV2K_train_HR.zip
rm DIV2K_valid_HR.zip

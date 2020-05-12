#!/bin/bash

FILE=$1

if [[ ${FILE} != "apple2orange" && ${FILE} != "summer2winter_yosemite" &&  ${FILE} != "horse2zebra" && ${FILE} != "monet2photo" && ${FILE} != "cezanne2photo" && ${FILE} != "ukiyoe2photo" && ${FILE} != "vangogh2photo" && ${FILE} != "maps" && ${FILE} != "facades" && ${FILE} != "iphone2dslr_flower" && ${FILE} != "ae_photos" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, iphone2dslr_flower"
    exit 1
fi

URL=https://github.com/Lornatang/CycleGAN-PyTorch/releases/download/1.0/${FILE}.zip
ZIP_FILE=${FILE}.zip
TARGET_DIR=${FILE}
wget -N ${URL} -O ${ZIP_FILE}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}

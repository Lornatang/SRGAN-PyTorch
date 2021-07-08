# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import cv2
import logging
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("data", metavar="DIR", help="Path to dataset.")
args = parser.parse_args()

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)


def main():
    # Initializes the mean, standard deviation, and file of the dataset.
    mean = [0., 0., 0.]
    std = [0., 0., 0.]

    # To ensure the correctness, the absolute path of the dataset is obtained.
    dataset_path = os.path.abspath(args.data)

    # Initializes the number of pictures in the dataset.
    num_images = 0

    # Get all folder information in the dataset.
    for dataset_dir in os.listdir(dataset_path):
        # Gets the absolute address of the folder in the dataset.
        image_dir = os.path.join(dataset_path, dataset_dir)
        # Traverse all image files under file.
        for file in os.listdir(image_dir):
            # Get the absolute address of the image.
            filename = os.path.join(image_dir, file)
            logger.info(f"Process `{filename}`.")
            # Caution: OpenCV read operator is BGR image data!
            image = cv2.imread(filename)
            # Convert image data to [0, 1].
            image = image.astype(np.float32) / 255.
            # Calculate the mean and variance of the three channels in turn.
            for i in range(3):
                mean[i] += image[:, :, i].mean()
                std[i] += image[:, :, i].std()
            # Calculate the number of pictures in the dataset.
            num_images += 1

    # What get the mean and variance of BGR format, which needs to be converted to the mean and variance of RGB format.
    mean.reverse()
    std.reverse()

    # Calculate the mean and variance of datasets.
    mean = np.asarray(mean) / num_images
    std = np.asarray(std) / num_images

    print(f"\nmean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]"
          f"std=[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}].")


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.07.02")

    main()

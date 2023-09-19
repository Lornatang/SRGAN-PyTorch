# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import multiprocessing
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


def main():
    args = {
        "inputs_dir": "./data/SRGAN_ImageNet",  # Path to input image directory.
        "output_dir": "./data/SRGAN_ImageNet_train_GT_sub",  # Path to generator image directory.
        "crop_size": 128,  # Crop image size from raw image.
        "step": 64,  # Step size of sliding window.
        "thresh_size": 0,  # Threshold size. If the remaining image is less than the threshold, it will not be cropped.
        "num_workers": 10  # How many threads to open at the same time.
    }
    split_images(args)

    # args = {
    #     "inputs_dir": "./data/DIV2K_train_HR",  # Path to input image directory.
    #     "output_dir": "./data/DIV2K_train_GT_sub",  # Path to generator image directory.
    #     "crop_size": 128,  # Crop image size from raw image.
    #     "step": 64,  # Step size of sliding window.
    #     "thresh_size": 0,  # Threshold size. If the remaining image is less than the threshold, it will not be cropped.
    #     "num_workers": 10  # How many threads to open at the same time.
    # }
    # split_images(args)


def split_images(args: dict):
    """Split the image into multiple small images.

    Args:
        args (dict): Custom parameter dictionary.

    """

    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    num_workers = args["num_workers"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Create {output_dir} successful.")
    else:
        print(f"{output_dir} already exists.")
        sys.exit(1)

    # Get all image paths
    image_file_paths = os.listdir(inputs_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_paths), unit="image", desc="Split image")
    workers_pool = multiprocessing.Pool(num_workers)
    for image_file_path in image_file_paths:
        workers_pool.apply_async(worker, args=(image_file_path, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()
    print("Split image successful.")


def worker(image_file_path: str, args: dict):
    """Split the image into multiple small images.

    Args:
        image_file_path (str): Image file path.
        args (dict): Custom parameter dictionary.

    """

    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    crop_size = args["crop_size"]
    step = args["step"]
    thresh_size = args["thresh_size"]

    image_name, extension = os.path.splitext(os.path.basename(image_file_path))
    image = cv2.imread(os.path.join(inputs_dir, image_file_path), cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[0:2]
    image_height_space = np.arange(0, image_height - crop_size + 1, step)
    if image_height - (image_height_space[-1] + crop_size) > thresh_size:
        image_height_space = np.append(image_height_space, image_height - crop_size)
    image_width_space = np.arange(0, image_width - crop_size + 1, step)
    if image_width - (image_width_space[-1] + crop_size) > thresh_size:
        image_width_space = np.append(image_width_space, image_width - crop_size)

    index = 0
    for h in image_height_space:
        for w in image_width_space:
            index += 1
            # Crop
            crop_image = image[h: h + crop_size, w:w + crop_size, ...]
            crop_image = np.ascontiguousarray(crop_image)
            # Save image
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_{index:04d}{extension}"), crop_image)


if __name__ == "__main__":
    main()

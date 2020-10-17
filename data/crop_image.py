# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
"""Making training data set quickly."""
import glob
import os
import shutil

import cv2
import torchvision.transforms as transforms
from PIL import Image

__all__ = [
    "center_crop", "create_folder", "crop_candidate_region"
]

raw_dir_10x = "./raw_data/10x"  # The processed image address is 10x.
raw_dir_20x = "./raw_data/20x"  # The processed image address is 20x.
raw_dir_40x = "./raw_data/40x"  # The processed image address is 40x.

new_dir_10x = "./10x"  # Path saved after 10x image processing.
new_dir_20x = "./20x"  # Path saved after 20x image processing.
new_dir_40x = "./40x"  # Path saved after 40x image processing.

lr_dir_2x = "./2x/train/input"  # Low resolution image at 2x.
lr_dir_4x = "./4x/train/input"  # Low resolution image at 4x.
hr_dir_2x = "./2x/train/target"  # High resolution image at 2x.
hr_dir_4x = "./4x/train/target"  # High resolution image at 4x.


def center_crop(raw_img_dir=None, dst_img_dir=None, crop_img_size: int = 1944) -> None:
    """ Center crop image with torchvision Library.

    Args:
        raw_img_dir (str): The address of the original folder that needs to be processed.
        dst_img_dir (str): Destination folder address after processing.
        crop_img_size (int): The length and width of the central interception area are consistent.
    """
    for filename in glob.glob(f"{raw_img_dir}/*"):
        img = Image.open(filename)
        # Call pytorch function to intercept the intermediate region directly.
        img = transforms.CenterCrop(crop_img_size)(img)
        img.save(f"{dst_img_dir}/{filename.split('/')[-1]}")


def create_folder(path: str = "./output") -> None:
    """ Ensure that the folder is successfully created and that the folder is empty.

    Args:
        path (str, optional): The folder path address can be absolute or relative.
            (Default: ``./output``)

    Returns:
        If it cannot be created successfully, an error is thrown.
    """
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    except OSError:
        pass


def crop_candidate_region(raw_img_dir: str = None, dst_img_dir: str = None,
                          lr_dir: str = None, hr_dir: str = None,
                          lr_img_size: int = 972, hr_img_size: int = 1944,
                          scale_factor: int = 2, candidate_box: list = None) -> None:
    """ The region corresponding to HR image is extracted from LR image.

    Args:
        raw_img_dir (str): Original picture folder path.
        dst_img_dir (str): Target picture folder path.
        lr_dir (str): Low resolution image folder path.
        hr_dir (str): High resolution image folder path.
        lr_img_size (int): Low resolution image size. (default: 972).
        hr_img_size (int): High resolution image size. (default: 1944).
        scale_factor (int): Image magnification. (Default: 2).
        candidate_box (list): Low resolution image candidate region.
    """
    if hr_img_size is None or hr_img_size < 4:
        raise (
            "The target image size cannot be empty, it must be a number "
            "greater than 4."
        )
    if lr_img_size is None:
        lr_img_size = int(hr_img_size // scale_factor)

    if candidate_box is None:
        raise (
            "An candidate matrix range must be specified!"
        )
    # Start width, end width, start height, end height.
    w1, w2, h1, h2 = candidate_box

    for filename in glob.glob(f"{raw_img_dir}/*"):
        img = cv2.imread(filename)
        candidate_img = img[w1:w2, h1:h2]  # Intercept the location of the default area, do not change!
        # Ensure that the size of the image is consistent with that of the target image.
        new_candidate_img = cv2.resize(candidate_img, (lr_img_size, lr_img_size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{lr_dir}/{filename.split('/')[-1]}", new_candidate_img)
        # Move the target image to the specified directory.
        shutil.copyfile(f"{dst_img_dir}/{filename.split('/')[-1]}",
                        f"{hr_dir}/{filename.split('/')[-1]}")


if __name__ == "__main__":
    # Create the necessary folders.
    print("Step 1: Create the necessary folders.")
    create_folder(new_dir_10x)
    create_folder(new_dir_20x)
    create_folder(new_dir_40x)
    create_folder(lr_dir_2x)
    create_folder(lr_dir_4x)
    create_folder(hr_dir_2x)
    create_folder(hr_dir_4x)

    # Traverse all '.bmp' suffix images in all folder.
    print("Step 2: Traverse all '.bmp' suffix images in all folder.")
    center_crop(raw_dir_10x, new_dir_10x, 1944)
    center_crop(raw_dir_20x, new_dir_20x, 1944)
    center_crop(raw_dir_40x, new_dir_40x, 1944)

    print("Step 3: Similar regions of HR image are extracted from LR image for 2x.")
    # Similar regions of HR image are extracted from LR image for 2x.
    crop_candidate_region(raw_img_dir=new_dir_10x, dst_img_dir=new_dir_20x,
                          lr_dir=lr_dir_2x, hr_dir=hr_dir_2x,
                          lr_img_size=972, hr_img_size=1944,
                          scale_factor=2, candidate_box=[507, 1443, 524, 1454])
    # print("Step 4: Similar regions of HR image are extracted from LR image for 4x.")
    # Similar regions of HR image are extracted from LR image for 4x.
    # crop_candidate_region(raw_img_dir=new_dir_10x, dst_img_dir=new_dir_40x,
    #                       input_dir=input_dir_4x, target_dir=target_dir_4x,
    #                       input_img_size=img_size_4x, target_img_size=crop_img_size,
    #                       scale_factor=4, candidate_box=[507, 1443, 524, 1454])

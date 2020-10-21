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
import glob
import os

import cv2
import numpy as np
from PIL import Image


def split_for_slicling(image: np.array, row_number: int = 9, col_number: int = 9) -> list:
    r""" Use the simple numpy slicing function.

    Args:
        image (cv2.imread): Image format read by opencv.
        row_number (int): Split along the width of the image. (Default: 9).
        col_number (int): Split along the height of the image. (Default: 9).

    Shape:
        image: :math:`(N, *)` where :math:`*` means, any number of additional dimensions

    Returns:
       Split an array into multiple sub-arrays.
    """
    __constant__ = ["row_number", "col_number"]
    row_number: int
    col_number: int

    # Cut picture vertically, get a lot of horizontal strips
    block_row = np.array_split(image, row_number, axis=0)
    image_blocks = []
    for block in block_row:
        # Horizontal direction cutting, get a lot of image blocks
        block_col = np.array_split(block, col_number, axis=1)
        image_blocks += [block_col]

    return image_blocks


def save_split_image(img_dir: str, row_number: int = 9, col_number: int = 9, delete: bool = True) -> None:
    r""" Save the split image.

    Args:
        img_dir (str): Original image folder to be processed.
        row_number (int): Split along the width of the image. (Default: 9).
        col_number (int): Split along the height of the image. (Default: 9).
        delete (optional, bool): Do you want to delete the original image after processing. (Default:``True``).
    """
    __constant__ = ["delete"]
    delete: bool

    for filename in glob.glob(f"{img_dir}/*"):
        img = cv2.imread(filename)
        image_blocks = split_for_slicling(img, row_number=row_number, col_number=col_number)
        for row in range(row_number):
            for col in range(col_number):
                image_blocks[row][col] = Image.fromarray(cv2.cvtColor(image_blocks[row][col], cv2.COLOR_BGR2RGB))
                image_blocks[row][col].save(f"{img_dir}/{filename.split('/')[-1].split('.')[0]}_{row}_{col}.bmp")
        if delete:
            os.remove(filename)


if __name__ == "__main__":
    save_split_image("./4x/train/input")

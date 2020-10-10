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

input_dir_2x = "./2x/train/input"  # Low resolution image at 2x.
input_dir_4x = "./4x/train/input"  # Low resolution image at 4x.
target_dir_2x = "./2x/train/target"  # High resolution image at 2x.
target_dir_4x = "./4x/train/target"  # High resolution image at 4x.
row_number = 4  # How many rows an image split into
col_number = 4  # How many cols an image split into


def split_for_slicling(image, row_number, col_number):
    r"""Use the simple numpy slicing function."""
    # Cut picture vertically, get a lot of horizontal strips
    block_row = np.array_split(image, row_number, axis=0)
    image_blocks = []
    for block in block_row:
        # Horizontal direction cutting, get a lot of image blocks
        block_col = np.array_split(block, col_number, axis=1)
        image_blocks += [block_col]

    return image_blocks


def save_split_image(img_dir: str = None) -> None:
    for filename in glob.glob(f"{img_dir}/*"):
        img = cv2.imread(filename)
        image_blocks = split_for_slicling(img, row_number=row_number, col_number=col_number)
        for row in range(row_number):
            for col in range(col_number):
                image_blocks[row][col] = Image.fromarray(cv2.cvtColor(image_blocks[row][col], cv2.COLOR_BGR2RGB))
                image_blocks[row][col].save(f"{img_dir}/{filename.split('/')[-1].split('.')[0]}_{row}_{col}.bmp")
        os.remove(filename)


if __name__ == "__main__":
    save_split_image(input_dir_2x)
    save_split_image(target_dir_2x)

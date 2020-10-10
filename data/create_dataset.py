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
import os
import random
import shutil

# Define raw directory
input_dir_2x = "./2x/train/input"  # Low resolution image at 2x.
input_dir_4x = "./4x/train/input"  # Low resolution image at 4x.
target_dir_2x = "./2x/train/target"  # High resolution image at 2x.
target_dir_4x = "./4x/train/target"  # High resolution image at 4x.
# Define the directory address of the dataset
lr_val_dir_2x = "./2x/val/data"
hr_val_dir_2x = "./2x/val/target"
lr_val_dir_4x = "./4x/val/data"
hr_val_dir_4x = "./4x/val/target"

# Ratio of val set in total dataset.
rate = 0.1

# Check all the file directory exists.
if not os.path.exists(input_dir_2x):
    raise FileExistsError(
        "Low resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(target_dir_2x):
    raise FileExistsError(
        "High resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(input_dir_4x):
    raise FileExistsError(
        "Low resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(target_dir_4x):
    raise FileExistsError(
        "High resolution image address does not exist, "
        "please make the original dataset.")

# Delete old folder
if os.path.exists(lr_val_dir_2x):
    shutil.rmtree(lr_val_dir_2x)
if os.path.exists(hr_val_dir_2x):
    shutil.rmtree(hr_val_dir_2x)
if os.path.exists(lr_val_dir_4x):
    shutil.rmtree(lr_val_dir_4x)
if os.path.exists(hr_val_dir_4x):
    shutil.rmtree(hr_val_dir_4x)

# Create new folder
os.makedirs(lr_val_dir_2x)
os.makedirs(hr_val_dir_2x)
os.makedirs(lr_val_dir_4x)
os.makedirs(hr_val_dir_4x)


def split_dataset(input_dir: str = None, target_dir: str = None,
                  lr_val_dir: str = None, hr_val_dir: str = None) -> None:
    # The original data set is divided into 9:1 (train:val)
    for _, _, files in os.walk(input_dir):
        # The size of the number of files in the total dataset.
        total_file_number = len(files)
        # Number of files in val set
        val_number = int(rate * total_file_number)
        # Gets a list of file names selected by random functions.
        samples = random.sample(files, val_number)
        # Move the validation set to the specified location.
        for filename in samples:
            # Making low resolution image data set.
            shutil.copyfile(os.path.join(input_dir, filename), os.path.join(lr_val_dir, filename))
            shutil.copyfile(os.path.join(target_dir, filename), os.path.join(hr_val_dir, filename))


if __name__ == "__main__":
    split_dataset(input_dir=input_dir_2x, target_dir=target_dir_2x,
                  lr_val_dir=lr_val_dir_2x, hr_val_dir=hr_val_dir_2x)

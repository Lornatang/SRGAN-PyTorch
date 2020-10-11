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
lr_train_dir_2x = "./2x/train/input"  # Low resolution image at 2x.
lr_train_dir_4x = "./4x/train/input"  # Low resolution image at 4x.
hr_train_dir_2x = "./2x/train/target"  # High resolution image at 2x.
hr_train_dir_4x = "./4x/train/target"  # High resolution image at 4x.
# Define the directory address of the dataset
lr_val_dir_2x = "./2x/val/input"
hr_val_dir_2x = "./2x/val/target"
lr_val_dir_4x = "./4x/val/input"
hr_val_dir_4x = "./4x/val/target"

# Ratio of val set in total dataset.
rate = 0.1

# Check all the file directory exists.
if not os.path.exists(lr_train_dir_2x):
    raise FileExistsError(
        "Low resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(hr_train_dir_2x):
    raise FileExistsError(
        "High resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(lr_train_dir_4x):
    raise FileExistsError(
        "Low resolution image address does not exist, "
        "please make the original dataset.")
if not os.path.exists(hr_train_dir_4x):
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


def split_dataset(lr_train_dir: str = None, hr_train_dir: str = None,
                  lr_val_dir: str = None, hr_val_dir: str = None) -> None:
    """ Make training data set and validation data set.

    Args:
        lr_train_dir (str): Low resolution train image folder path.
        hr_train_dir (str): High resolution train image folder path.
        lr_val_dir (str): Low resolution val image folder path.
        hr_val_dir (str): High resolution val image folder path.
    """
    # The original data set is divided into 9:1 (train:val)
    for _, _, files in os.walk(lr_train_dir):
        # The size of the number of files in the total dataset.
        total_file_number = len(files)
        # Number of files in val set
        val_number = int(rate * total_file_number)
        # Gets a list of file names selected by random functions.
        samples = random.sample(files, val_number)
        # Move the validation set to the specified location.
        for filename in samples:
            # Making low resolution image data set.
            shutil.copyfile(os.path.join(lr_train_dir, filename), os.path.join(lr_val_dir, filename))
            shutil.copyfile(os.path.join(hr_train_dir, filename), os.path.join(hr_val_dir, filename))


if __name__ == "__main__":
    split_dataset(lr_train_dir=lr_train_dir_2x, hr_train_dir=hr_train_dir_2x,
                  lr_val_dir=lr_val_dir_2x, hr_val_dir=hr_val_dir_2x)

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

# Ratio of val set in total dataset.
rate = 0.1


def split_dataset(train_dir: str = None, val_dir: str = None) -> None:
    """ Make training data set and validation data set.

    Args:
        train_dir (str): Train image folder path.
        val_dir (str): Val image folder path.
    """
    # The original data set is divided into 9:1 (train:val)
    for _, _, files in os.walk(train_dir):
        # The size of the number of files in the total dataset.
        total_file_number = len(files)
        # Number of files in val set
        val_number = int(rate * total_file_number)
        # Gets a list of file names selected by random functions.
        samples = random.sample(files, val_number)
        # Move the validation set to the specified location.
        for filename in samples:
            # Making low resolution image data set.
            shutil.move(os.path.join("train", "input", filename), os.path.join("test", "input", filename))
            shutil.move(os.path.join("train", "target", filename), os.path.join("test", "target", filename))


if __name__ == "__main__":
    split_dataset(train_dir="./train/input", val_dir="./test/input")

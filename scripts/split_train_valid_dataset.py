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
import os
import random
import shutil

from tqdm import tqdm


def main(args) -> None:
    if not os.path.exists(args.train_images_dir):
        os.makedirs(args.train_images_dir)
    if not os.path.exists(args.valid_images_dir):
        os.makedirs(args.valid_images_dir)

    train_files = os.listdir(args.train_images_dir)
    valid_files = random.sample(train_files, int(len(train_files) * args.valid_samples_ratio))

    process_bar = tqdm(valid_files, total=len(valid_files), unit="image", desc="Split train/valid dataset")

    for image_file_name in process_bar:
        shutil.copyfile(f"{args.train_images_dir}/{image_file_name}", f"{args.valid_images_dir}/{image_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts.")
    parser.add_argument("--train_images_dir", type=str, help="Path to train image directory.")
    parser.add_argument("--valid_images_dir", type=str, help="Path to valid image directory.")
    parser.add_argument("--valid_samples_ratio", type=float, help="What percentage of the data is extracted from the training set into the validation set.")
    args = parser.parse_args()

    main(args)

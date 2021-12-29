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
import shutil

from PIL import Image
from torchvision.transforms import RandomCrop
from tqdm import tqdm


def main() -> None:
    image_dir = f"{args.output_dir}/train"

    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    file_names = os.listdir(args.inputs_dir)
    for file_name in tqdm(file_names, total=len(file_names)):
        # Use PIL to read high-resolution image
        image = Image.open(f"{args.inputs_dir}/{file_name}").convert("RGB")

        if image.width >= args.image_size and image.height >= args.image_size:
            for i in range(10):
                new_image = RandomCrop([args.image_size, args.image_size])(image)
                new_image.save(f"{image_dir}/{file_name.split('.')[-2]}_{i:03d}.{file_name.split('.')[-1]}")
    print("Data split successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--inputs_dir", type=str, default="ImageNet/original", help="Path to input image directory. (Default: `ImageNet/original`)")
    parser.add_argument("--output_dir", type=str, default="ImageNet/SRGAN", help="Path to generator image directory. (Default: `ImageNet/SRGAN`)")
    parser.add_argument("--image_size", type=int, default=96, help="Low-resolution image size from raw image. (Default: 96)")
    args = parser.parse_args()

    main()

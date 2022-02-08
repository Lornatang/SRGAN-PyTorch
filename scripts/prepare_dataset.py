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
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm


def main(args) -> None:
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Get all image paths
    image_file_names = os.listdir(args.images_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Split")
    workers_pool = Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    image = Image.open(f"{args.images_dir}/{image_file_name}").convert("RGB")

    index = 1
    if image.width >= args.image_size and image.height >= args.image_size:
        for pos_x in range(0, image.width - args.image_size + 1, args.step):
            for pos_y in range(0, image.height - args.image_size + 1, args.step):
                index += 1
                crop_image = image.crop([pos_x, pos_y, pos_x + args.image_size, pos_y + args.image_size])
                # Save all images
                crop_image.save(f"{args.output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to generator image directory.")
    parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.")
    parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    args = parser.parse_args()

    main(args)

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
import os

from PIL import Image


def main():
    image_sizex0  = 128
    image_sizex2  = 64
    image_sizex4  = 32

    filenames = os.listdir("HR")
    for filename in sorted(filenames):
        print(f"Process: `{filename}`.")

        imagex0 = Image.open(os.path.join("HR",          filename))
        imagex2 = Image.open(os.path.join("LRunknownx2", filename))
        imagex4 = Image.open(os.path.join("LRunknownx4", filename))

        crop_imagesx0 = crop_image(imagex0, image_sizex0)
        crop_imagesx2 = crop_image(imagex2, image_sizex2)
        crop_imagesx4 = crop_image(imagex4, image_sizex4)

        save_images(os.path.join("HR",          filename), crop_imagesx0)
        save_images(os.path.join("LRunknownx2", filename), crop_imagesx2)
        save_images(os.path.join("LRunknownx4", filename), crop_imagesx4)

        os.remove(os.path.join("HR",          filename))
        os.remove(os.path.join("LRunknownx2", filename))
        os.remove(os.path.join("LRunknownx4", filename))


def crop_image(image, crop_size: int):
    assert image.size[0] == image.size[1]
    # Get the split image size.
    crop_number = image.size[0] // crop_size
    # (left, upper, right, lower)
    box_list = []
    for width_index in range(0, crop_number):
        for height_index in range(0, crop_number):
            box = ((height_index + 0) * crop_size, (width_index + 0) * crop_size,
                   (height_index + 1) * crop_size, (width_index + 1) * crop_size)
            box_list.append(box)

    # Save all split image data.
    crop_images = [image.crop(box) for box in box_list]
    return crop_images


def save_images(raw_filename, image_list):
    index = 1
    for image in image_list:
        image.save(raw_filename.split(".")[0] + f"_{index:08d}.bmp")
        index += 1


if __name__ == "__main__":
    main()

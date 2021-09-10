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

# ==============================================================================
# File description: Realize the function of data set preparation.
# ==============================================================================
import os
from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode as IMode

from imgproc import center_crop
from imgproc import image2tensor
from imgproc import random_crop
from imgproc import random_horizontally_flip
from imgproc import random_rotate

__all__ = ["BaseDataset", "CustomDataset"]


class BaseDataset(Dataset):
    """The basic data set loading function only needs to prepare high-resolution image data.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(BaseDataset, self).__init__()
        lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
        hr_image_size = (image_size, image_size)
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        # Low-resolution images and high-resolution images have different processing methods.
        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(hr_image_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.hr_transforms = transforms.Compose([
                transforms.CenterCrop(hr_image_size),
                transforms.ToTensor()
            ])
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(lr_image_size, interpolation=IMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])

        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)


class CustomDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(CustomDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and low-resolution folder
        # under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_dir_path = os.path.join(dataroot, f"LRunknownx{upscale_factor}")
        hr_dir_path = os.path.join(dataroot, f"HR")
        self.filenames = os.listdir(lr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in self.filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in self.filenames]

        self.image_size = image_size  # HR image size.
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.lr_filenames[index])
        hr = Image.open(self.hr_filenames[index])

        # Data enhancement methods.
        if self.mode == "train":
            lr, hr = random_crop(lr, hr, self.image_size, self.upscale_factor)
            lr, hr = random_rotate(lr, hr, 90)
            lr, hr = random_horizontally_flip(lr, hr, 0.5)
        else:
            lr, hr = center_crop(lr, hr, self.image_size, self.upscale_factor)

        # `PIL.Image` image data is converted to `Tensor` format data.
        lr = image2tensor(lr)
        hr = image2tensor(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)

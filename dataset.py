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

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode as IMode

from imgproc import image2tensor
from imgproc import random_crop
from imgproc import random_horizontally_flip
from imgproc import random_vertically_flip

__all__ = ["BaseDataset", "CustomDataset"]


class BaseDataset(data.Dataset):
    def __init__(self, dataroot: str, image_size: int, downscale_factor: int) -> None:
        super(BaseDataset, self).__init__()
        lr_image_size = (image_size // downscale_factor, image_size // downscale_factor)
        hr_image_size = (image_size, image_size)
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(lr_image_size, IMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop(hr_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])

        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(hr)

        hr = (hr / 0.5) - 1.

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)


class CustomDataset(data.Dataset):
    def __init__(self, dataroot: str, image_size: int, downscale_factor: int) -> None:
        super(CustomDataset, self).__init__()
        lr_dir_path = os.path.join(dataroot, "LRcubicx4")
        hr_dir_path = os.path.join(dataroot, "HR")
        filenames = os.listdir(lr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in filenames]

        self.image_size = image_size  # HR image size.
        self.downscale_factor = downscale_factor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.lr_filenames[index])
        hr = Image.open(self.hr_filenames[index])

        lr, hr = random_crop(lr, hr, self.image_size, self.downscale_factor)
        lr, hr = random_horizontally_flip(lr, hr)
        lr, hr = random_vertically_flip(lr, hr)

        lr = image2tensor(lr)
        hr = image2tensor(hr, norm=True)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)

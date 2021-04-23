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
import random

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from .utils.common import check_image_file
from .utils.data_augmentation import random_horizontally_flip
from .utils.data_augmentation import random_vertically_flip

__all__ = [
    "BaseTrainDataset", "BaseTestDataset",
    "CustomTrainDataset", "CustomTestDataset"
]


class BaseTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96)
            upscale_factor (optional, int): Image magnification. (Default: 4)
        """
        super(BaseTrainDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        image = Image.open(self.filenames[index]).convert("RGB")

        hr = self.hr_transforms(image)
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self):
        return len(self.filenames)


class BaseTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96)
            upscale_factor (optional, int): Image magnification. (Default: 4)
        """
        super(BaseTestDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        image = Image.open(self.filenames[index]).convert("RGB")

        hr = self.hr_transforms(image)
        lr = self.lr_transforms(hr)
        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr

    def __len__(self):
        return len(self.filenames)


class CustomTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (int): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTrainDataset, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr = Image.open(self.lr_filenames[index]).convert("RGB")
        hr = Image.open(self.hr_filenames[index]).convert("RGB")

        lr, hr = random_horizontally_flip(lr, hr)
        lr, hr = random_vertically_flip(lr, hr)

        lr = self.transforms(lr)
        hr = self.transforms(hr)

        return lr, hr

    def __len__(self):
        return len(self.sampler_filenames)


class CustomTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 256, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 256)
            sampler_frequency (list): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTestDataset, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr = Image.open(self.lr_filenames[index]).convert("RGB")
        hr = Image.open(self.hr_filenames[index]).convert("RGB")

        lr = self.transforms(lr)
        bicubic = self.bicubic_transforms(lr)
        hr = self.transforms(hr)

        return lr, bicubic, hr

    def __len__(self):
        return len(self.sampler_filenames)

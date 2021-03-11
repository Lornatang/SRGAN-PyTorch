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

import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode as mode

__all__ = [
    "check_image_file",
    "BaseTrainDataset", "BaseTestDataset",
    "CustomTrainDataset", "CustomTestDataset"
]


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    return any(filename.endswith(extension) for extension in [".jpg", ".JPG",
                                                              ".jpeg", ".JPEG",
                                                              ".png", ".PNG",
                                                              ".bmp", ".BMP"])


class BaseTrainDataset(torch.utils.data.dataset.Dataset):
    """An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        """
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96).
            upscale_factor (optional, int): Image magnification. (Default: 4).
        """
        super(BaseTrainDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=mode.BICUBIC),
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
        hr = self.hr_transforms(Image.open(self.filenames[index]).convert("RGB"))
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self):
        return len(self.filenames)


class BaseTestDataset(torch.utils.data.dataset.Dataset):
    """An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        """
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96).
            upscale_factor (optional, int): Image magnification. (Default: 4).
        """
        super(BaseTestDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=mode.BICUBIC),
            transforms.ToTensor()
        ])
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=mode.BICUBIC),
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
        hr = self.hr_transforms(Image.open(self.filenames[index]).convert("RGB"))
        lr = self.lr_transforms(hr)
        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr

    def __len__(self):
        return len(self.filenames)


class CustomTrainDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str, sampler_frequency: int = 1):
        """

        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (list): If there are many datasets, this method can be used to increase
                the number of epochs. (Default: 1).
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
        lr = self.transforms(Image.open(self.lr_filenames[index]))
        hr = self.transforms(Image.open(self.hr_filenames[index]))

        return lr, hr

    def __len__(self):
        return len(self.sampler_filenames)


class CustomTestDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str, image_size: int, sampler_frequency: int = 1):
        """

        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image.
            sampler_frequency (list): If there are many datasets, this method can be used to increase
                the number of epochs. (Default: 1).
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
            transforms.Resize((image_size, image_size), interpolation=mode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr = self.transforms(Image.open(self.lr_filenames[index]))
        bicubic = self.bicubic_transforms(lr)
        hr = self.transforms(Image.open(self.hr_filenames[index]))

        return lr, bicubic, hr

    def __len__(self):
        return len(self.sampler_filenames)

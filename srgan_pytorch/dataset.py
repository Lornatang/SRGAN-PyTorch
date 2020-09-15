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

import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.

    """
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    r"""Auto crop image size.

    Args:
        crop_size (int): Minimum size of image block.
        upscale_factor (int): Image magnification factor.

    Returns:
        The minimum length and width of the cropped image.

    """
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size, upscale_factor):
    r"""Processing format of training high resolution image.

    Args:
        crop_size (int): Minimum size of image block.
        upscale_factor (int): Image magnification factor.

    Returns:
        Composes several transforms together.

    """
    return transforms.Compose([
        transforms.RandomCrop(crop_size * upscale_factor),
        transforms.ToTensor(),
    ])


def train_lr_transform(crop_size):
    r"""Processing format of training low resolution image.

    Args:
        crop_size (int): Minimum size of image block.

    Returns:
        Composes several transforms together.

    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


def display_transform(image_size=384):
    r"""Display the processed image.

    Args:
        image_size (int): Displays the size of the image. Default: 384.

    Returns:
        Composes several transforms together.

    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])


class TrainDatasetFromFolder(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, dataset_dir, crop_size, upscale_factor):
        r"""

        Args:
            dataset_dir (str): Directory address of the file.
            crop_size (int): Minimum size of image block.
            upscale_factor (int): Image magnification factor.
        """
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if check_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size, upscale_factor)
        self.lr_transform = train_lr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, dataset_dir, upscale_factor):
        r"""

        Args:
            dataset_dir (str): Directory address of the file.
            upscale_factor (int): Image magnification factor.
        """
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if check_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = transforms.Resize((crop_size // self.upscale_factor, crop_size // self.upscale_factor),
                                     interpolation=Image.BICUBIC)
        hr_scale = transforms.Resize((crop_size, crop_size), interpolation=Image.BICUBIC)
        hr_image = transforms.CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, dataset_dir, upscale_factor):
        r"""

        Args:
            dataset_dir (str): Directory address of the file.
            upscale_factor (int): Image magnification factor.
        """
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + 'result/X' + str(upscale_factor) + '/data'
        self.hr_path = dataset_dir + 'result/X' + str(upscale_factor) + '/target'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [os.path.join(self.lr_path, x) for x in os.listdir(self.lr_path) if check_image_file(x)]
        self.hr_filenames = [os.path.join(self.hr_path, x) for x in os.listdir(self.hr_path) if check_image_file(x)]

    def __getitem__(self, index):
        filename = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = transforms.Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return filename, transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(
            hr_image)

    def __len__(self):
        return len(self.lr_filenames)

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
"""Realize the function of dataset preparation."""
import io
import os

import lmdb
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

import imgproc

__all__ = ["ImageDataset", "LMDBDataset"]


class ImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address
        image_size       (int): High resolution image size
        upscale_factor   (int): Image magnification
        mode             (str): Data set loading method, the training data set is for data enhancement,
                             and the verification data set is not for data enhancement
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        self.image_file_names = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomRotation([0, 90]),
                transforms.RandomHorizontalFlip(0.5),
            ])
        else:
            self.hr_transforms = transforms.CenterCrop(image_size)

        self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC)

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        image = Image.open(self.image_file_names[batch_index])

        # Transform image
        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.image_file_names)


class LMDBDataset(Dataset):
    """Load the data set as a data set in the form of LMDB.

    Attributes:
        lr_datasets (list): Low-resolution image data in the dataset
        hr_datasets (list): High-resolution image data in the dataset
    """

    def __init__(self, lr_lmdb_path, hr_lmdb_path) -> None:
        super(LMDBDataset, self).__init__()
        # Create low/high resolution image array
        self.lr_datasets = []
        self.hr_datasets = []

        # Initialize the LMDB database file address
        self.lr_lmdb_path = lr_lmdb_path
        self.hr_lmdb_path = hr_lmdb_path

        # Write image data in LMDB database to memory
        self.read_lmdb_dataset()

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        lr_image = self.lr_datasets[batch_index]
        hr_image = self.hr_datasets[batch_index]

        # Data augment
        lr_image, hr_image = imgproc.random_rotate(lr_image, hr_image, angle=90)
        lr_image, hr_image = imgproc.random_horizontally_flip(lr_image, hr_image, p=0.5)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.hr_datasets)

    def read_lmdb_dataset(self) -> [list, list]:
        # Open two LMDB database writing environments to read low/high image data
        lr_lmdb_env = lmdb.open(self.lr_lmdb_path)
        hr_lmdb_env = lmdb.open(self.hr_lmdb_path)

        # Write the image data in the low-resolution LMDB data set to the memory
        for _, image_bytes in lr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.lr_datasets.append(image)

        # Write the image data in the high-resolution LMDB data set to the memory
        for _, image_bytes in hr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.hr_datasets.append(image)

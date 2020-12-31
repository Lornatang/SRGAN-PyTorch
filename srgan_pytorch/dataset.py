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

        self.input_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.target_transforms = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        target = self.target_transforms(Image.open(self.filenames[index]))
        input = self.input_transforms(target)

        return input, target

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

        self.input_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.target_transforms = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        target = self.target_transforms(Image.open(self.filenames[index]))
        input = self.input_transforms(target)
        bicubic = self.bicubic_transforms(input)

        return input, bicubic, target

    def __len__(self):
        return len(self.filenames)


class CustomTrainDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str):
        """

        Args:
            root (str): The directory address where the data image is stored.
        """
        super(CustomTrainDataset, self).__init__()
        input_dir = os.path.join(root, "input")
        target_dir = os.path.join(root, "target")
        self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        input = self.transforms(Image.open(self.input_filenames[index]))
        target = self.transforms(Image.open(self.target_filenames[index]))

        return input, target

    def __len__(self):
        return len(self.input_filenames)


class CustomTestDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, root: str, image_size: int):
        """

        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image.
        """
        super(CustomTestDataset, self).__init__()
        input_dir = os.path.join(root, "input")
        target_dir = os.path.join(root, "target")
        self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        input = self.transforms(Image.open(self.input_filenames[index]))
        bicubic = self.bicubic_transforms(input)
        target = self.transforms(Image.open(self.target_filenames[index]))

        return input, bicubic, target

    def __len__(self):
        return len(self.input_filenames)

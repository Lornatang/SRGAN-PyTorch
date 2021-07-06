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

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from .utils.common import check_image_file
from .utils.data_augmentation import random_horizontally_flip
from .utils.data_augmentation import random_vertically_flip

__all__ = ["BaseDataset", "CustomDataset"]


class BaseDataset(torch.utils.data.dataset.Dataset):
    r""" Base dataset loader constructed using bicubic down-sampling method.

    Args:
        root (str): The directory address where the data image is stored.
        image_size (optional, int): The size of image block is randomly cut out from the original image.
        upscale_factor (optional, int): Image magnification.
    """

    def __init__(self, root, image_size, upscale_factor):
        super(BaseDataset, self).__init__()
        lr_image_size = int(image_size / upscale_factor)
        # Get the index of all images in the directory that meet the suffix format conditions.
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((lr_image_size, lr_image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.filenames[index]).convert("RGB")

        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(hr)

        # Norm HR image [0, 1] to [-1, 1]
        hr = (hr / 0.5) - 1.

        return lr, hr

    def __len__(self):
        return len(self.filenames)


class CustomDataset(torch.utils.data.dataset.Dataset):
    r""" Load through the pre-dataset.

    Args:
        root (str): The directory address where the data image is stored.
        image_size (optional, int): The size of image block is randomly cut out from the original image.
        upscale_factor (optional, int): Image magnification.
        use_da (optional, bool): Do you want to use data enhancement for training dataset. (Default: `True`)
    """

    def __init__(self, root, image_size, upscale_factor, use_da=True):
        super(CustomDataset, self).__init__()
        lr_image_size = int(image_size / upscale_factor)
        self.use_da = use_da

        # Get the index of all images in the directory that meet the suffix format conditions.
        self.filenames = os.path.join(root, "input")
        self.lr_filenames = [os.path.join(root, "input", x) for x in self.filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(root, "target", x) for x in self.filenames if check_image_file(x)]

        self.lr_transforms = transforms.Compose(
            [transforms.CenterCrop((lr_image_size, lr_image_size)),
             transforms.ToTensor()])
        self.hr_transforms = transforms.Compose(
            [transforms.CenterCrop((image_size, image_size)),
             transforms.ToTensor()])

    def __getitem__(self, index):
        lr = Image.open(self.lr_filenames[index]).convert("RGB")
        hr = Image.open(self.hr_filenames[index]).convert("RGB")

        if self.use_da:
            lr, hr = random_horizontally_flip(lr, hr)
            lr, hr = random_vertically_flip(lr, hr)

        # Norm HR image [0, 1] to [-1, 1]
        hr = (hr / 0.5) - 1.

        return lr, hr

    def __len__(self):
        return len(self.filenames)

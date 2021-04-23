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
"""Mainly apply the data augment operation on `Torchvision` and `PIL` to the super-resolution field"""
import PIL.BmpImagePlugin
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional_pil as F

__all__ = ["adjust_brightness", "adjust_contrast", "adjust_saturation",
           "random_horizontally_flip", "random_vertically_flip", "random_rotate"]


def adjust_brightness(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile, p: float = 0.5) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Adjust brightness of an image.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Adjust the probability value, only process when the probability value is greater than 0.5.

    Returns:
        Adjusted brightness low-resolution image and adjusted brightness high-resolution image.
    """

    if p > 0.5:
        lr = F.adjust_brightness(lr, brightness_factor=p)
        hr = F.adjust_brightness(hr, brightness_factor=p)

    return lr, hr


def adjust_contrast(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile, p: float = 0.5) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Adjust contrast of an image.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Adjusted contrast low-resolution image and adjusted contrast high-resolution image.
    """

    lr = F.adjust_contrast(lr, brightness_factor=p)
    hr = F.adjust_contrast(hr, brightness_factor=p)

    return lr, hr


def adjust_saturation(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile, p: float = 0.5) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Adjust saturation of an image.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Adjusted saturation low-resolution image and adjusted saturation high-resolution image.
    """

    lr = F.adjust_saturation(lr, brightness_factor=p)
    hr = F.adjust_saturation(hr, brightness_factor=p)

    return lr, hr


def random_horizontally_flip(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile,
                             p: float = 0.5) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Flip horizontally randomly.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Flipped low-resolution image and flipped high-resolution image.
    """
    if torch.rand(1) < p:
        lr = F.hflip(lr)
        hr = F.hflip(hr)

    return lr, hr


def random_vertically_flip(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile,
                           p: float = 0.5) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Flip horizontally randomly.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Flipped low-resolution image and flipped high-resolution image.
    """
    if torch.rand(1) < p:
        lr = F.hflip(lr)
        hr = F.hflip(hr)

    return lr, hr


def random_rotate(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile) -> PIL.BmpImagePlugin.BmpImageFile:
    r"""Randomly select the rotation angle.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.

    Returns:
        Rotated low-resolution image and rotated high-resolution image.
    """
    angle = transforms.RandomRotation.get_params([-180, 180])
    lr = lr.rotate(angle)
    hr = hr.rotate(angle)

    return lr, hr

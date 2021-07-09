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
# ============================================================================
import torch
from torchvision.transforms.functional_pil import hflip
from torchvision.transforms.functional_pil import vflip

__all__ = ["random_horizontally_flip", "random_vertically_flip"]


def random_horizontally_flip(lr, hr, p=0.5):
    r""" Realize the image random horizontal flip function.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution image data read using PIL library.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution image data read using PIL library.
        p (optional, float): Rollover probability. (Default: 0.5)

    Returns:
        If the random probability is greater than 0.5, the image data after the horizontal flip is returned, 
        otherwise the original image data is returned.
    """
    if torch.rand(1) > p:
        lr = hflip(lr)
        hr = hflip(hr)

    return lr, hr


def random_vertically_flip(lr, hr, p=0.5):
    r""" Realize the function of randomly flipping up and down images.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution image data read using PIL library.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution image data read using PIL library.
        p (optional, float): Rollover probability. (Default: 0.5)

    Returns:
        If the random probability is greater than 0.5, the image data after the vertical flip is returned, 
        otherwise the original image data is returned.
    """
    if torch.rand(1) > p:
        lr = vflip(lr)
        hr = vflip(hr)

    return lr, hr

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
import torch
import torchvision.transforms.functional_pil as F

__all__ = ["random_horizontally_flip", "random_vertically_flip"]


def random_horizontally_flip(lr, hr, p=0.5):
    r"""Flip horizontally randomly.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Flipped low-resolution image and flipped high-resolution image.
    """
    if torch.rand(1) > p:
        lr = F.hflip(lr)
        hr = F.hflip(hr)

    return lr, hr


def random_vertically_flip(lr, hr, p=0.5):
    r"""Flip horizontally randomly.

    Args:
        lr (PIL.BmpImagePlugin.BmpImageFile): Low-resolution images loaded with PIL.
        hr (PIL.BmpImagePlugin.BmpImageFile): High-resolution images loaded with PIL.
        p (float): Probability of the image being flipped. (Default: 0.5)

    Returns:
        Flipped low-resolution image and flipped high-resolution image.
    """
    if torch.rand(1) > p:
        lr = F.vflip(lr)
        hr = F.vflip(hr)

    return lr, hr

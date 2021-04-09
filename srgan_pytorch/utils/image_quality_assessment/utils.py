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
import numpy as np
import torch
import torch.nn.functional

__all__ = [
    "gaussian_filter", "fspecial_gauss"
]


def gaussian_filter(x: torch.Tensor, filter_weight: torch.Tensor) -> torch.Tensor:
    r"""Gaussian filtering using two dimensional convolution.

    Args:
        x (torch.Tensor): Input tensor.
        filter_weight (torch.Tensor): Gaussian filter weight.

    Returns:
        torch.Tensor.
    """
    out = torch.nn.functional.conv2d(x, filter_weight, stride=1, padding=0, groups=x.shape[1])
    return out


def fspecial_gauss(size: int, sigma: float):
    r"""Implementation of Gaussian filter(MATLAB) in Python.

    Args:
        size (int): Wave filter size.
        sigma (float): Standard deviation of filter.

    Returns:
        Picture after using filter.
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    gauss = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    gauss = torch.from_numpy(gauss / gauss.sum()).float().unsqueeze(0).unsqueeze(0)
    out = gauss.repeat(3, 1, 1, 1)
    return out

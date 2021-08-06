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

# ==============================================================================
# File description: Realize the function of processing the data set before training.
# ==============================================================================

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.functional_pil as F_pil
from torch import Tensor


def normalize(image) -> np.ndarray:
    return image.astype(np.float64) / 255.0


def unnormalize(image) -> np.ndarray:
    return image.astype(np.float64) * 255.0


def image2tensor(image, norm: bool = False) -> Tensor:
    tensor = F.to_tensor(image)
    if norm:
        tensor = (tensor / 0.5) - 1

    return tensor


def tensor2image(tensor, norm: bool = False) -> np.ndarray:
    image = F.to_pil_image(tensor)
    if norm:
        image = (image + 1) * 127.5

    return image


def random_crop(lr, hr, image_size: int, downscale_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    w, h = hr.size
    left = torch.randint(0, w - image_size + 1, size=(1,)).item()
    top = torch.randint(0, h - image_size + 1, size=(1,)).item()
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left   // downscale_factor,  
                  top    // downscale_factor, 
                  right  // downscale_factor, 
                  bottom // downscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_rotate(lr, hr) -> Tuple[np.ndarray, np.ndarray]:
    angle = random.choice((90, 180, 270))
    lr = F_pil.rotate(lr, angle)
    hr = F_pil.rotate(hr, angle)

    return lr, hr


def random_horizontally_flip(lr, hr, p=0.5) -> Tuple[np.ndarray, np.ndarray]:
    if torch.rand(1).item() > p:
        lr = F_pil.hflip(lr)
        hr = F_pil.hflip(hr)

    return lr, hr


def random_vertically_flip(lr, hr, p=0.5) -> Tuple[np.ndarray, np.ndarray]:
    if torch.rand(1).item() > p:
        lr = F_pil.vflip(lr)
        hr = F_pil.vflip(hr)

    return lr, hr


def random_adjust_brightness(lr, hr) -> Tuple[np.ndarray, np.ndarray]:
    factor = random.uniform(0.25, 4)
    lr = F_pil.adjust_brightness(lr, factor)
    hr = F_pil.adjust_brightness(hr, factor)

    return lr, hr


def random_adjust_contrast(lr, hr) -> Tuple[np.ndarray, np.ndarray]:
    factor = random.uniform(0.25, 4)
    lr = F_pil.adjust_contrast(lr, factor)
    hr = F_pil.adjust_contrast(hr, factor)

    return lr, hr

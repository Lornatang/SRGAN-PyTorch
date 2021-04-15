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
import random

import PIL.BmpImagePlugin
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

__all__ = [
    "opencv2pil", "opencv2tensor", "pil2opencv", "process_image", "train_transform"
]


def opencv2pil(image: np.ndarray) -> PIL.BmpImagePlugin.BmpImageFile:
    """ OpenCV Convert to PIL.Image format.

    Returns:
        PIL.Image.
    """

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def opencv2tensor(image: np.ndarray, gpu: int) -> torch.Tensor:
    """ OpenCV Convert to torch.Tensor format.

    Returns:
        torch.Tensor.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nhwc_image = torch.from_numpy(rgb_image).div(255.0).unsqueeze(0)
    input_tensor = nhwc_image.permute(0, 3, 1, 2)
    if gpu is not None:
        input_tensor = input_tensor.cuda(gpu, non_blocking=True)
    return input_tensor


def pil2opencv(image: PIL.BmpImagePlugin.BmpImageFile) -> np.ndarray:
    """ PIL.Image Convert to OpenCV format.

    Returns:
        np.ndarray.
    """

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def process_image(image: PIL.BmpImagePlugin.BmpImageFile, gpu: int = None) -> torch.Tensor:
    """ PIL.Image Convert to PyTorch format.

    Args:
        image (PIL.BmpImagePlugin.BmpImageFile): File read by PIL.Image.
        gpu (int): Graphics card model.

    Returns:
        torch.Tensor.
    """
    tensor = transforms.ToTensor()(image)
    input_tensor = tensor.unsqueeze(0)
    if gpu is not None:
        input_tensor = input_tensor.cuda(gpu, non_blocking=True)
    return input_tensor


def train_transform(lr: PIL.BmpImagePlugin.BmpImageFile, hr: PIL.BmpImagePlugin.BmpImageFile):
    r"""Unified transformation of training dataset."""
    # Rotate the image by angle.
    angle = transforms.RandomRotation.get_params([-180, 180])
    lr = lr.rotate(angle)
    hr = hr.rotate(angle)

    # Horizontally flip the given image randomly with a given probability.
    p = random.random()
    if p > 0.5:
        lr = F.hflip(lr)
        lr = F.vflip(lr)
        hr = F.hflip(hr)
        hr = F.vflip(hr)

    # Randomly change the brightness, contrast, saturation and hue of an image.
    lr = F.adjust_brightness(lr, brightness_factor=p)
    lr = F.adjust_contrast(lr, contrast_factor=p)
    lr = F.adjust_saturation(lr, saturation_factor=p)
    hr = F.adjust_brightness(hr, brightness_factor=p)
    hr = F.adjust_contrast(hr, contrast_factor=p)
    hr = F.adjust_saturation(hr, saturation_factor=p)

    # PIL to tensor.
    lr = F.to_tensor(lr)
    hr = F.to_tensor(hr)

    return lr, hr

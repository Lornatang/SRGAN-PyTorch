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
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

__all__ = ["opencv2pil", "opencv2tensor", "pil2opencv", "process_image"]


def opencv2pil(image):
    """ OpenCV Convert to PIL.Image format.

    Returns:
        PIL.Image.
    """

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def opencv2tensor(image, gpu=None):
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


def pil2opencv(image):
    """ PIL.Image Convert to OpenCV format.

    Returns:
        np.ndarray.
    """

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def process_image(image, norm=False, gpu=None) -> torch.Tensor:
    """ RGB data convert to tensor data(PyTorch format).

    Args:
        image (image): File read by PIL.Image.
        norm (bool): Is it necessary to normalize the input to [-1, 1]. (Default: `False`)
        gpu (int): Graphics card index.

    Examples:
        >>> tensor_image = process_image(Image.open("lena.png"), norm=True, gpu=1)

    Returns:
        torch.Tensor.
    """
    # Convert C*H*W data to N*C*H*W data. Example: (3, 64, 64) -> (1, 3, 64, 64).
    tensor = transforms.ToTensor()(image).unsqueeze(0)

    # If set to `True`, the input scaling is normalized to between [-1, 1].
    if norm:
        tensor = (tensor - 0.5) / 2

    # If the GPU model is specified, the data will be transferred to the GPU, otherwise it will be processed on the CPU.
    if gpu is not None:
        tensor = tensor.cuda(gpu, non_blocking=True)

    return tensor

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
import PIL.BmpImagePlugin
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

__all__ = [
    "opencv2pil", "opencv2tensor", "pil2opencv", "process_image"
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

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


def opencv2tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """ OpenCV Convert to torch.Tensor format.

    Returns:
        torch.Tensor.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nhwc_image = torch.from_numpy(rgb_image).div(255.0).unsqueeze(0)
    nchw_image = nhwc_image.permute(0, 3, 1, 2)
    input_tensor = nchw_image.to(device)
    return input_tensor


def pil2opencv(image: PIL.BmpImagePlugin.BmpImageFile) -> np.ndarray:
    """ PIL.Image Convert to OpenCV format.

    Returns:
        np.ndarray.
    """

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def process_image(image: PIL.BmpImagePlugin.BmpImageFile, device: torch.device) -> torch.Tensor:
    """ PIL.Image Convert to PyTorch format.

    Args:
        image (PIL.BmpImagePlugin.BmpImageFile): File read by PIL.Image.
        device (torch.device): Location of data set processing.

    Returns:
        torch.Tensor.
    """
    tensor = transforms.ToTensor()(image)
    input_tensor = tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    return input_tensor

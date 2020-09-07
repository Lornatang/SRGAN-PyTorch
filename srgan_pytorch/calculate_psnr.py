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
import math

import torch
import torchvision.transforms as transforms
from PIL import Image


def cal_psnr(prediction, target):
    r"""Python simply calculates the maximum signal noise ratio.

    Args:
        prediction (tensor): Low resolution image.
        target (tensor): high resolution image.

    ..math:
        10 \cdot \log _{10}\left(\frac{MAX_{I}^{2}}{MSE}\right)

    Returns:
        Maximum signal to noise ratio between two images.
    """
    pil_to_tensor = transforms.ToTensor()
    # Convert pictures to tensor format
    with torch.no_grad():
        prediction = pil_to_tensor(Image.open(prediction)).unsqueeze(0)
        target = pil_to_tensor(Image.open(target)).unsqueeze(0)

    mse = ((prediction - target) ** 2).data.mean()

    return 10 * math.log10((target.max() ** 2) / mse)

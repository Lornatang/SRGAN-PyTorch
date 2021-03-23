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
import torch
import torch.nn.functional

from .utils import fspecial_gauss
from .utils import gaussian_filter

__all__ = [
    "ssim", "SSIM"
]


# Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
def ssim(image1: torch.Tensor, image2: torch.Tensor, filter_weight: torch.Tensor, cs=False) -> float:
    """Python implements structural similarity.

    Args:
        image1 (np.array): Original tensor picture.
        image2 (np.array): Target tensor picture.
        filter_weight (torch.Tensor): Gaussian filter weight.
        cs (bool): It is mainly used to calculate ms-ssim value. (default: False)

    Returns:
        Structural similarity value.
    """
    k1 = 0.01 ** 2
    k2 = 0.03 ** 2

    filter_weight = filter_weight.to(image1.device)

    mu1 = gaussian_filter(image1, filter_weight)
    mu2 = gaussian_filter(image2, filter_weight)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(image1 * image1, filter_weight) - mu1_sq
    sigma2_sq = gaussian_filter(image2 * image2, filter_weight) - mu2_sq
    sigma12 = gaussian_filter(image1 * image2, filter_weight) - mu1_mu2

    cs_map = (2 * sigma12 + k2) / (sigma1_sq + sigma2_sq + k2)
    cs_map = torch.nn.functional.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + k1) / (mu1_sq + mu2_sq + k1)) * cs_map
    out = ssim_map.mean([1, 2, 3])

    if cs:
        cs_out = cs_map.mean([1, 2, 3])
        return out, cs_out

    return out


class SSIM(torch.nn.Module):
    def __init__(self) -> None:
        super(SSIM, self).__init__()
        self.filter_weight = fspecial_gauss(11, 1.5)

    def forward(self, image1_tensor: torch.Tensor, image2_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image1_tensor (torch.Tensor): Original tensor picture.
            image2_tensor (torch.Tensor): Target tensor picture.

        Returns:
            torch.Tensor.
        """
        assert image1_tensor.shape == image2_tensor.shape
        out = torch.mean(ssim(image1_tensor, image2_tensor, filter_weight=self.filter_weight))

        return out

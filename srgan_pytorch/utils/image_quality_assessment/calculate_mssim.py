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

from .calculate_ssim import ssim
from .utils import fspecial_gauss


def ms_ssim(image1: torch.Tensor, image2: torch.Tensor, filter_weight: torch.Tensor) -> float:
    """ Multi scale structural similarity

    Args:
        image1 (np.array): Original tensor picture.
        image2 (np.array): Target tensor picture.
        filter_weight (torch.Tensor): Gaussian filter weight.

    Returns:
        MS_SSIM value.
    """
    assert image1.shape == image2.shape

    ssim_value = ssim(image1, image2, filter_weight=filter_weight)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(image1.device, dtype=image2.dtype)

    mcs = []
    for _ in range(weights.shape[0]):
        _, cs_value = ssim(image1, image2, filter_weight=filter_weight, cs=True)
        mcs.append(cs_value)
        padding = (image1.shape[2] % 2, image2.shape[3] % 2)
        image1 = torch.nn.functional.avg_pool2d(image1, kernel_size=2, padding=padding)
        image2 = torch.nn.functional.avg_pool2d(image2, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)
    out = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_value ** weights[-1]), dim=0)
    return out


class MS_SSIM(torch.nn.Module):
    def __init__(self) -> None:
        super(MS_SSIM, self).__init__()
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
        out = torch.mean(ms_ssim(image1_tensor, image2_tensor, filter_weight=self.filter_weight))

        return out

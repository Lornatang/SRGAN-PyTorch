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


def ms_ssim(source: torch.Tensor, target: torch.Tensor, filter_weight: torch.Tensor) -> float:
    """ Multi scale structural similarity

    Args:
        source (np.array): Original tensor picture.
        target (np.array): Target tensor picture.
        filter_weight (torch.Tensor): Gaussian filter weight.

    Returns:
        MS_SSIM value.
    """
    assert source.shape == target.shape

    ssim_value = ssim(source, target, filter_weight=filter_weight)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(source.device, dtype=target.dtype)

    mcs = []
    for _ in range(weights.shape[0]):
        _, cs_value = ssim(source, target, filter_weight=filter_weight, cs=True)
        mcs.append(cs_value)
        padding = (source.shape[2] % 2, target.shape[3] % 2)
        source = torch.nn.functional.avg_pool2d(source, kernel_size=2, padding=padding)
        target = torch.nn.functional.avg_pool2d(target, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)
    out = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_value ** weights[-1]), dim=0)
    return out


class MS_SSIM(torch.nn.Module):
    def __init__(self) -> None:
        super(MS_SSIM, self).__init__()
        self.filter_weight = fspecial_gauss(size=11, sigma=1.5)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert source.shape == target.shape
        out = torch.mean(ms_ssim(source, target, filter_weight=self.filter_weight))

        return out

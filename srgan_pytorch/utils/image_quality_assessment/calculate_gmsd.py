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
from torch import nn
from torch.nn import functional as F

__all__ = [
    "GMSD"
]


# Source code reference from `http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm`.
class GMSD(nn.Module):
    """Gradient map similarity deviation"""

    def __init__(self) -> None:
        super(GMSD, self).__init__()
        source = (torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        target = (torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.image1 = nn.Parameter(source, requires_grad=False)
        self.image2 = nn.Parameter(target, requires_grad=False)
        self.average_kernel = nn.Parameter(torch.ones(3, 1, 2, 2) / 4., requires_grad=False)

    def gmsd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        image1 = F.conv2d(source, weight=self.average_kernel, stride=2, padding=0, groups=3)
        image2 = F.conv2d(target, weight=self.average_kernel, stride=2, padding=0, groups=3)

        image1_image1 = F.conv2d(image1, weight=self.image1, stride=1, padding=1, groups=3)
        image1_image2 = F.conv2d(image1, weight=self.image2, stride=1, padding=1, groups=3)
        gradient_map1 = torch.sqrt(image1_image1 ** 2 + image1_image2 ** 2 + 1e-12)

        image2_image1 = F.conv2d(image2, weight=self.image1, stride=1, padding=1, groups=3)
        image2_image2 = F.conv2d(image2, weight=self.image2, stride=1, padding=1, groups=3)
        gradient_map2 = torch.sqrt(image2_image1 ** 2 + image2_image2 ** 2 + 1e-12)

        quality_map = (2 * gradient_map1 * gradient_map2 + 170) / (gradient_map1 ** 2 + gradient_map2 ** 2 + 170)
        out = torch.std(quality_map.view(quality_map.shape[0], -1), dim=1)

        return out

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert source.shape == target.shape
        out = torch.mean(self.gmsd(source * 255, target * 255))

        return out

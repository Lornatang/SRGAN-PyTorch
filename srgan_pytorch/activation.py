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
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FReLU", "HSigmoid", "HSwish", "Mish", "Sine", "Swish"
]


class FReLU(nn.Module):
    r""" Applies the FReLU function element-wise.

    `"Funnel Activation for Visual Recognition" <https://arxiv.org/pdf/2007.11824.pdf>`_

    Examples:
        >>> channels = 64
        >>> frelu = FReLU(channels)
        >>> x = torch.randn(1, channels, 64, 64)
        >>> output = frelu(x)
    """

    def __init__(self, channels: int = 64) -> None:
        r"""

        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super().__init__()
        self.FReLU = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor):
        out = self.FReLU(x)
        return torch.max(x, out)


class HSigmoid(nn.Module):
    r""" Applies the Hard-Sigmoid function element-wise.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3, inplace=True) / 6.


class HSwish(nn.Module):
    r""" Applies the Hard-Swish function element-wise.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6.


class Mish(nn.Module):
    r""" Applies the Mish function element-wise.

    `"Mish: A Self Regularized Non-Monotonic Activation Function" <https://arxiv.org/pdf/1908.08681.pdf>`_

    .. math:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - x: (N, *) where * means, any number of additional.
          dimensions
        - Output: (N, *), same shape as the x.

    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * (torch.tanh(F.softplus(x)))


class Sine(nn.Module):
    r""" Applies the sine function element-wise.

    `"Implicit Neural Representations with Periodic Activation Functions" <https://arxiv.org/pdf/2006.09661.pdf>`_

    Examples:
        >>> m = Sine()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30.
        return torch.sin(30 * x)


class Swish(nn.Module):
    r""" Applies the swish function element-wise.

    `"Searching for Activation Functions" <https://arxiv.org/pdf/1710.05941.pdf>`_ paper.

    Examples:
        >>> m = Swish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(x)

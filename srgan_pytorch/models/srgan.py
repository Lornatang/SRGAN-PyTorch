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
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "srgan_2x2_16": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/0.1.0/mnist-5539a1a7.pth",
    "srgan_4x4_16": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/0.1.2/mnist-5539a1a7.pth",
    "srgan_2x2_23": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/0.1.2/tfd-4e44e2ca.pth",
    "srgan_4x4_23": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/0.1.2/tfd-4e44e2ca.pth"

}


class Generator(nn.Module):
    r"""The main architecture of the generator."""

    def __init__(self, upscale_factor=4, num_residual_block=16):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 16 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 23 RRDB blocks.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
            num_residual_block (int): How many residual blocks are combined. (Default: 16).
        """
        num_upsample_block = int(math.log(upscale_factor, 2))

        super(Generator, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.PReLU()
        )

        # 16 Residual blocks
        residual_blocks = []
        for _ in range(num_residual_block):
            residual_blocks.append(ResidualBlock(64))
        self.Trunk = nn.Sequential(*residual_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        out = self.Trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        return out


class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out + input


def _srgan(arch, upscale_factor, num_residual_block, pretrained, progress):
    model = Generator(upscale_factor, num_residual_block)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def srgan_2x2_16(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _srgan("srgan_2x2_16", 2, 16, pretrained, progress)


def srgan_4x4_16(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _srgan("srgan_4x4_16", 4, 16, pretrained, progress)


def srgan_2x2_23(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _srgan("srgan_2x2_23", 2, 23, pretrained, progress)


def srgan_4x4_23(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _srgan("srgan_4x4_23", 4, 23, pretrained, progress)

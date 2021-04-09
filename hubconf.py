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
"""File for accessing GAN via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load("Lornatang/SRGAN-PyTorch", "srgan", pretrained=True, progress=True, verbose=False)
"""
import torch
from torch.hub import load_state_dict_from_url

from srgan_pytorch.models.generator import Generator

model_urls = {
    "srgan_2x2": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.1/SRGAN_2x2_DIV2K-9ec9dd11.pth",
    "srgan": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.1/SRGAN_DIV2K-6b7848ca.pth",
    "srgan_8x8": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.1/SRGAN_8x8_DIV2K-5ded8368.pth"
}

dependencies = ["torch"]


def create(arch: str, upscale_factor: int, pretrained: bool, progress: bool) -> Generator:
    model = Generator(upscale_factor)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def srgan_2x2(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("srgan_2x2", 2, pretrained, progress)


def srgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("srgan", 4, pretrained, progress)


def srgan_8x8(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1609.04802>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("srgan_8x8", 8, pretrained, progress)

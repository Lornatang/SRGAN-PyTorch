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
from torch.hub import load_state_dict_from_url

from .utils import ResidualBlock
from .utils import SubpixelConvolutionLayer

model_urls = {
    "srgan": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.2/SRGAN_ImageNet2012-992702908bcbce3b6e2bc2d15eb5b4eb7a5c816468654819c6efbbd79ce671ea.pth"
}


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # First layer.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # 16 Residual blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(channels=64))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks.
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # 2 Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(2):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(64))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # Final output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = torch.add(conv1, conv2)
        out = self.subpixel_conv(out)
        out = self.conv3(out)
        out = torch.tanh(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _gan(arch: str, pretrained: bool, progress: bool) -> Generator:
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def srgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1609.04802>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("srgan", pretrained, progress)

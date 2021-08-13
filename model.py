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

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch import Tensor


__all__ = [
    "ResidualBlock", "Discriminator", "Generator", "PerceptualLoss",
]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv_block(x)
        out = out + identity

        return out


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 96) -> None:
        super(Discriminator, self).__init__()
        feature_size = image_size // 16

        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_size * feature_size, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )

        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(64))
        self.trunk = nn.Sequential(*trunk)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )

        self.conv_block3 = nn.Conv2d(64, 3, 9, 1, 4)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out = self.upsampling(out)
        out = self.conv_block3(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.fill_(0)


class PerceptualLoss(nn.Module):
    """Constructed a perceptual loss function based on the VGG19 network.
    The loss defined on the feature map of higher-level features from deeper network layers is more likely to focus on the content of the image.

    `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network   `<https://arxiv.org/pdf/1609.04802.pdf>` paper.
    `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                      `<https://arxiv.org/pdf/1809.00219.pdf>` paper.
    `Perceptual Extreme Super Resolution Network with Receptive Field Block                 `<https://arxiv.org/pdf/2005.12597.pdf>` paper.
    """

    def __init__(self) -> None:
        super(PerceptualLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # The feature extraction layer in the VGG19 model is extracted as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the preprocessing method recommended by the VGG19 model.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = transforms.Resize([224, 224])

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # Normalize operation.
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        # Scale to the input size of the VGG19 model.
        sr = self.resize(sr)
        hr = self.resize(hr)

        # Find the feature map difference between two images.
        loss = F.mse_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss

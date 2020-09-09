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
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            in_channels (int): Number of channels in the input image. Default: 64.
            out_channels (int): Number of channels produced by the convolution. Default: 64.
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.

        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        shortcut = inputs

        out = self.conv1(inputs)
        out = self.bn(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn(out)

        out = out + shortcut

        return out


class UpsampleBlock(nn.Module):
    r"""Up-sampling method block"""

    def __init__(self, in_channels, out_channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffler(x)

        out = F.relu(x, inplace=True)
        return out


class Generator(nn.Module):
    r"""The main architecture of the generator."""

    def __init__(self, n_residual_blocks, upsample_factor):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            n_residual_blocks (int): Number of consecutive residual blocks.
            upsample_factor (int): Coefficient of image magnification.
        """
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module("residual_block_" + str(i + 1), ResidualBlock())

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor // 2):
            self.add_module("upsample_block_" + str(i + 1), UpsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__("residual_block_" + str(i + 1))(y)

        shortcut = x

        x = self.conv2(y)
        x = self.bn(x)
        x = x + shortcut

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__("upsample_block_" + str(i + 1))(x)

        out = self.conv3(x)

        return out


class Discriminator(nn.Module):
    r"""The main architecture of the discriminator."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x, inplace=True)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x, inplace=True)

        x = self.conv9(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.sigmoid(x)
        out = torch.flatten(x, 1)
        return out

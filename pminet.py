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
"""文件说明: 实现模型定义功能."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

__all__ = [
    "SqueezeExcitationModule", "LightAttentionConvBlock",
    "DiscriminatorForVGG", "Generator",
    "ContentLoss",
]


class SqueezeExcitationModule(nn.Module):
    """注意力卷积模块. 自动提取一幅图像中感兴趣区域, 在这里实现的是一种软注意力方法,
    通过反向传播更新注意力卷积模块机制内部的权重.

    Attributes:
        se_module (nn.Sequential): 定义注意力卷积方法.

    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """

        Args:
            channels (int): 输入图像的通道数.
            reduction (optional, int): 通道数降维因子.

        """
        super(SqueezeExcitationModule, self).__init__()
        hidden_channels = channels // reduction

        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.Hardsigmoid(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Tensor(NCHW)格式图像数据.

        Returns:
            torch.Tensor: 注意力卷积处理后Tensor(NCHW)格式图像数据.

        """
        out = self.se_module(x)
        out = torch.mul(out, x)

        return out


class LightAttentionConvBlock(nn.Module):
    def __init__(self, channels: int, multiplier_ratio: int = 4) -> None:
        super(LightAttentionConvBlock, self).__init__()
        multiplier_channels = int(channels * multiplier_ratio)

        self.lac_block = nn.Sequential(
            nn.Conv2d(channels, multiplier_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(multiplier_channels, multiplier_channels, (3, 3), (1, 1), (1, 1), groups=multiplier_channels),

            SqueezeExcitationModule(multiplier_channels),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(multiplier_channels, channels, (1, 1), (1, 1), (0, 0)),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.lac_block(x)
        out = torch.add(out, identity)

        return out


class DiscriminatorForVGG(nn.Module):
    def __init__(self) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(4)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, (3, 3), (1, 1), (1, 1))

        # 特征提取层
        trunk = []
        for _ in range(16):
            trunk.append(LightAttentionConvBlock(128))
        self.trunk = nn.Sequential(*trunk)

        # 特征抽象层
        self.conv2 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))

        # 上采样卷积层
        self.upsampling = nn.Sequential(
            nn.Conv2d(128, 512, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
        )

        # 输出层
        self.conv3 = nn.Conv2d(128, 3, (3, 3), (1, 1), (1, 1))

        # 初始化模型权重
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # 支持Torch.script方法.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ContentLoss(nn.Module):
    """构建了基于VGG19网络的感知损失函数.
    使用来自后几层的高级别的特征映射层会更专注于图像的纹理.

    论文参考列表:
        - `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        - `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        - `Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
    """

    def __init__(self, model_path: str = "") -> None:
        """初始化内容损失模型

        Args:
            model_path (str): 内容损失模型地址

        """
        super(ContentLoss, self).__init__()
        # 根据模型权重选择合适模型架构以及预处理方式
        if model_path != "":
            # 加载基于BreakHis数据集训练的VGG19模型
            model = models.vgg19(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 8),
            )
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

            # 针对BreakHis数据集的预处理参数
            self.register_buffer("mean", torch.Tensor([0.805, 0.657, 0.776]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.088, 0.113, 0.087]).view(1, 3, 1, 1))
        else:
            # 加载基于ImageNet数据集训练的VGG19模型
            model = models.vgg19(pretrained=True)

            # 针对ImageNet数据集的预处理参数
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 提取VGG19模型中第三十五层输出作为内容损失
        self.feature_extractor = nn.Sequential(*list(model.features.children())[:35])

        # 冻结模型参数
        self.feature_extractor.eval()
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        """用VGG模型求两张图片的内容差异

        Args:
            sr_tensor (torch.Tensor): 超分辨图像，PyTorch支持的数据格式
            hr_tensor (torch.Tensor): 高分辨图像，PyTorch支持的数据格式

        Returns:
            (torch.Tensor): 内容损失数值

        """
        # 图像标准化操作
        sr_tensor = sr_tensor.sub(self.mean).div(self.std)
        hr_tensor = hr_tensor.sub(self.mean).div(self.std)

        # 求两张图像之间的特征图差异
        loss = F.l1_loss(self.feature_extractor(sr_tensor), self.feature_extractor(hr_tensor))

        return loss


# if __name__ == "__main__":
#     ours_model = Generator()
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#
#     esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
#     x = torch.randn([1, 3, 128, 128])
#     import time
#     from fvcore.nn import FlopCountAnalysis, parameter_count
#
#     # 计算参数和FLOPs
#     parameters = parameter_count(esrgan_model)
#     flops = FlopCountAnalysis(esrgan_model, x)
#
#     # 计算时间
#     start_time = time.time()
#     for i in range(2):
#         _ = esrgan_model(x)
#     use_time = time.time() - start_time
#
#     print(f"Use time: ({use_time:.3f}s/{use_time / 2:.3f}s).\n"
#           f"Parameters: {parameters[''] / 1e6:.2f}MB.\n"
#           f"FLOPs: {flops.total() / 1e9:.2f}G")

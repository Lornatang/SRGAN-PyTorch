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
from torch.nn import functional
from torchvision import models

__all__ = ["ContentLoss"]


class ContentLoss(nn.Module):
    r""" The content loss function based on vgg19 network is constructed.
    According to the suggestion of the paper, the 36th layer of feature extraction layer is used.

    The explanation of the paper is as follows:
        * "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" `<https://arxiv.org/pdf/1609.04802v5.pdf>` paper.
        * "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" `<https://arxiv.org/pdf/1809.00219.pdf>` paper.
        * "Perceptual Extreme Super Resolution Network with Receptive Field Block" `<https://arxiv.org/pdf/2005.12597.pdf>` paper.

    A loss defined on feature maps of higher level features from deeper network layers
    with more potential to focus on the content of the images. We refer to this network
    as SRGAN in the following.

    Examples:
        >>> # Loading pre training vgg19 model weight based on Imagenet dataset as content loss.
        >>> content_loss = ContentLoss()
        >>> # According to the input size of VGG19 model, an image with a resolution of 224*224 is randomly constructed.
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> target = torch.randn(1, 3, 224, 224)
        >>> loss = content_loss(inputs, target)

    Notes:
        features(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
          (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (6): ReLU(inplace=True)
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (11): ReLU(inplace=True)
          (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU(inplace=True)
          (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU(inplace=True)
          (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU(inplace=True)
          (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU(inplace=True)
          (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU(inplace=True)
          (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU(inplace=True)
          (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU(inplace=True)
          (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (29): ReLU(inplace=True)
          (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (31): ReLU(inplace=True)
          (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (33): ReLU(inplace=True)
          (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (35): ReLU(inplace=True)  ---> use this layer
          (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    """

    def __init__(self):
        super(ContentLoss, self).__init__()
        # If you will `use_pretrained` is set to `True`, the model weight based
        # on Imagenet dataset will be loaded,
        # otherwise, the custom dataset model weight will be loaded.
        vgg19 = models.vgg19(pretrained=True).eval()

        # Extract the 36th layer of vgg19 model feature extraction layer.
        self.model = nn.Sequential(*list(vgg19.features.children())[:36])

        # Freeze model all parameters. Don't train.
        for _, parameters in self.model.named_parameters():
            parameters.requires_grad = False

    def forward(self, source, target):
        # Convert the image value range to [0, 1].
        source = (source + 1) / 2
        target = (target + 1) / 2

        # Use VGG19_36th loss as the euclidean distance between the feature
        # representations of a reconstructed image and the reference image.
        loss = functional.mse_loss(self.model(source), self.model(target))

        return loss

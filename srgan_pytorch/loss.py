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
from torch import nn
from torchvision.models import vgg19


class ContentLoss_VGG22(nn.Module):
    """A loss defined on feature maps representing lower-level features.
    """

    def __init__(self, feature_layer=8):
        """ Constructing characteristic loss function of VGG network. For VGG2.2.

        Args:
            feature_layer (int): How many layers in VGG19. (Default:8).
        """
        super(ContentLoss_VGG22, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]).eval()

    def forward(self, x):
        return self.features(x)


class ContentLoss_VGG54(nn.Module):
    """ a loss defined on feature maps of higher level features from deeper network layers
        with more potential to focus on the content of the images. We refer to this network
        as SRGAN in the following.
    """

    def __init__(self, feature_layer=30):
        """ Constructing characteristic loss function of VGG network. For VGG5.4.

        Args:
            feature_layer (int): How many layers in VGG19. (Default:30).
        """
        super(ContentLoss_VGG54, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]).eval()

    def forward(self, x):
        return self.features(x)


class TVLoss(nn.Module):
    """Regularization loss based on Li FeiFei."""

    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, input):
        batch_size = input.size()[0]
        h_x = input.size()[2]
        w_x = input.size()[3]
        count_h = self.tensor_size(input[:, :, 1:, :])
        count_w = self.tensor_size(input[:, :, :, 1:])
        h_tv = torch.pow((input[:, :, 1:, :] - input[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((input[:, :, :, 1:] - input[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

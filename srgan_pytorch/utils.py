# Copyright 2020 Lorna Authors. All Rights Reserved.
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

import cv2
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features)[:31]).eval()

    def forward(self, x):
        return self.features(x)


def evaluate_performance(real_image, fake_image):
    prediction = cv2.imread(fake_image)
    target = cv2.imread(real_image)

    error_value = []
    for i in range(len(target)):
        error_value.append((target[i] - prediction[i]) ** 2)

    mse = sum(error_value) / len(error_value)

    if mse < 1.0e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))

    return mse, psnr


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

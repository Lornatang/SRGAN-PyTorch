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
import os

import cv2
import torch.nn as nn
from sewar.full_ref import mse
from sewar.full_ref import psnr


class FeatureExtractor(nn.Module):
    def __init__(self, model, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


def evaluate_performance(input_folder, target_folder):
    fake_images = os.listdir(input_folder)
    fake_images.sort()
    real_images = os.listdir(target_folder)
    real_images.sort()

    assert len(fake_images) == len(real_images), "Number of pictures does not match!"

    mse_list = []
    psnr_list = []

    for index in range(len(fake_images)):
        fake_image = cv2.imread(os.path.join(input_folder, fake_images[index]))
        real_image = cv2.imread(os.path.join(target_folder, real_images[index]))

        mse_list.append(mse(real_image, fake_image))
        psnr_list.append(psnr(real_image, fake_image))

    return sum(mse_list) / len(fake_images), sum(psnr_list) / len(fake_images)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

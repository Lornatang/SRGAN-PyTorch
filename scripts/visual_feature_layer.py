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
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

image = transforms.ToTensor()(Image.open("visual/SR/104_27.png"))

vgg19 = models.vgg19(pretrained=True).eval()
model = nn.Sequential(*list(vgg19.features.children())[:36])

for _, parameters in model.named_parameters():
    parameters.requires_grad = False

features = model(image).squeeze(0)

for i in range(1, 2):
    plt.subplot(1, 1, i)
    new_img_PIL = transforms.ToPILImage()(features[i - 1]).convert()
    plt.savefig("features.png")

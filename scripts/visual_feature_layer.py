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
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

torch.manual_seed(666)

image = transforms.ToTensor()(Image.open("visual/SR/104_27.png"))

mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

model = models.vgg19(pretrained=True)
model = torch.nn.Sequential(*list(model.features.children())[:35]).eval()

for name, parameters in model.named_parameters():
    parameters.requires_grad = False

image = (image - mean) / std

features = model(image).squeeze(0)

for i in range(1, 2):
    plt.subplot(1, 1, i)
    new_img_PIL = transforms.ToPILImage()(features[i - 1]).convert()
    plt.imshow(np.asarray(new_img_PIL))
    print(i)
    plt.savefig("features.png")

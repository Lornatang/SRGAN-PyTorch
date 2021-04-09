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
import os

import torch
import torch.nn.functional as F
import torchvision.models as models

__all__ = [
    "LPIPS"
]


# Source code reference from `https://github.com/richzhang/PerceptualSimilarity`.
class LPIPS(torch.nn.Module):
    def __init__(self, gpu: int = None) -> None:
        super(LPIPS, self).__init__()
        model = models.vgg19(pretrained=True).eval()

        # Freeze parameters. Don't train.
        for param in self.parameters():
            param.requires_grad = False

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), model.features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), model.features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), model.features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), model.features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), model.features[x])

        # Image normalization method.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.channels = [64, 128, 256, 512, 512]
        if gpu is None:
            self.weights = torch.load(os.path.join(os.path.abspath("weights"), "lpips_vgg.pth"))
        else:
            # Map model to be loaded to specified single gpu.
            self.weights = torch.load(os.path.join(os.path.abspath("weights"), "lpips_vgg.pth"), map_location=f"cuda:{gpu}")
        self.weights = list(self.weights.items())

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.mean) / self.std
        relu1_2 = self.stage1(out)
        relu2_2 = self.stage2(relu1_2)
        relu3_3 = self.stage3(relu2_2)
        relu4_3 = self.stage4(relu3_3)
        relu5_3 = self.stage5(relu4_3)
        out = [relu1_2, relu2_2, relu3_3, relu4_3, relu5_3]

        for k in range(len(out)):
            out[k] = F.normalize(out[k])

        return out

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert source.shape == target.shape
        source_features = self.forward_once(source)
        target_features = self.forward_once(target)
        out = 0
        for k in range(len(self.channels)):
            out = out + (self.weights[k][1] * (source_features[k] - target_features[k]) ** 2).mean([2, 3]).sum(1)

        out = torch.mean(out)

        return out

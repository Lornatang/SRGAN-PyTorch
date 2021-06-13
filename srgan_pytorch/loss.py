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
"""It mainly implements all the losses used in the model."""
import lpips
import torch
import torch.nn.functional
import torchvision

__all__ = [
    "CharbonnierLoss", "ContentLoss", "LPIPSLoss"
]


class CharbonnierLoss(torch.nn.Module):
    r""" The charbonnier loss(one variant of Robust L1Loss) function optimizes
    the error between the minimum residual image and the real image by one level.

    The explanation of the paper is as follows:
        * `"Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution" <https://arxiv.org/pdf/1710.01992.pdf>` paper.

    Learn a mapping function for generating an HR image that is as similar to the ground truth HR image as possible.

    Args:
        eps (float): Prevent value equal to 0. (Default: 1e-12)

    Examples:
        >>> charbonnier_loss = CharbonnierLoss()
        >>> # Create a resolution of 224*224 image.
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> target = torch.randn(1, 3, 224, 224)
        >>> loss = charbonnier_loss(inputs, target)
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate charbonnier loss.
        loss = torch.mean(torch.sqrt((source - target) ** 2 + self.eps))

        return loss


class ContentLoss(torch.nn.Module):
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
        >>> # According to the input size of VGG19 model, an image with a resolution of 224*224 is randomly constructed
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

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # If you will `use_pretrained` is set to `True`, the model weight based on Imagenet dataset will be loaded,
        # otherwise, the custom dataset model weight will be loaded.
        vgg19 = torchvision.models.vgg19(pretrained=True)

<<<<<<< Updated upstream
        # If the weight of pre training model is used, the normalized value on Imagenet will be loaded,
        # otherwise, the normalized value on custom data set will be loaded.
        if use_pretrained:
            # Normalized values for ImageNet datasets.
            self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            assert model_path is not None, "You must define the model weight address."
            model.load_state_dict(torch.load(model_path))
            # Normalized values for specified datasets.
            self.mean = torch.Tensor([0.864, 0.565, 0.625]).view(1, 3, 1, 1)
            self.std = torch.Tensor([0.089, 0.104, 0.049]).view(1, 3, 1, 1)

        # Extract the 35th layer of vgg19 model feature extraction layer.
        self.model = torch.nn.Sequential(*list(model.features.children())[:35]).eval()
=======
        # Extract the 36th layer of vgg19 model feature extraction layer.
        self.feature_extract = torch.nn.Sequential(*list(vgg19.features.children())[:36]).eval()
>>>>>>> Stashed changes

        # Freeze model all parameters. Don't train.
        for name, parameters in self.feature_extract.named_parameters():
            parameters.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
<<<<<<< Updated upstream
        # Keep all parameters in same device.
        self.mean = self.mean.cuda(source.device, non_blocking=True)
        self.std = self.std.cuda(source.device, non_blocking=True)

        # Normalize the all input image.
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std

=======
>>>>>>> Stashed changes
        # Use VGG19_35th loss as the euclidean distance between the feature representations of a reconstructed image and the reference image.
        loss = torch.nn.functional.mse_loss(self.feature_extract(source), self.feature_extract(target))

        return loss


# TODO: Source code reference from `https://github.com/richzhang/PerceptualSimilarity`.
class LPIPSLoss(torch.nn.Module):
    r""" Learned Perceptual Image Patch Similarity (LPIPS) metric.

    The explanation of the paper is as follows:
        * `"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" <https://arxiv.org/pdf/1801.03924.pdf>` paper.

    For a specific convolution layer, the cosine distance (in the channel dimension)
    and the average value between the network space dimension and layers are calculated.

    Examples:
        >>> # Loading pre training vgg19 model weight based on Imagenet dataset as content loss.
        >>> lpips_loss = LPIPSLoss()
        >>> # According to the input size of VGG19 model, an image with a resolution of 224*224 is randomly constructed
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> target = torch.randn(1, 3, 224, 224)
        >>> loss = lpips_loss(inputs, target)
    """

    def __init__(self) -> None:
        super(LPIPSLoss, self).__init__()
        self.feature_extract = lpips.LPIPS(net="vgg", verbose=False).eval()

<<<<<<< Updated upstream
        # Normalize the input image. Caution: input tensor range [0, 1].
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Keep all parameters in same device.
        self.mean = self.mean.cuda(source.device, non_blocking=True)
        self.std = self.std.cuda(source.device, non_blocking=True)

        # Normalize the input image. Default: `ImageNet` dataset.
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std
=======
        # Freeze model all parameters. Don't train.
        for name, parameters in self.feature_extract.named_parameters():
            parameters.requires_grad = False
>>>>>>> Stashed changes

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use lpips_vgg loss as the euclidean distance between the feature representations of a reconstructed image and the reference image.
        loss = torch.nn.functional.mse_loss(self.feature_extract(source), self.feature_extract(target))

        return loss

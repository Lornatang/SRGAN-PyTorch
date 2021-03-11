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
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from .image_quality_assessment import GMSD
from .image_quality_assessment import LPIPS
from .image_quality_assessment import MS_SSIM
from .image_quality_assessment import SSIM

__all__ = [
    "iqa", "test_psnr", "test_gan"
]


def iqa(image1_tensor: torch.Tensor, image2_tensor: torch.Tensor, gpu: int = None) -> torch.Tensor:
    """Image quality evaluation function.

    Args:
        image1_tensor (torch.Tensor): Original tensor picture.
        image2_tensor (torch.Tensor): Target tensor picture.
        gpu (int): Graphics card index.

    Returns:
        MSE, RMSE, PSNR, SSIM, MS-SSIM, LPIPS, GMSD
    """
    mse_loss = nn.MSELoss().cuda(gpu)
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS().cuda(gpu)
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu)
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    msssim_loss = MS_SSIM().cuda(gpu)

    # Complete estimate.
    mse_value = mse_loss(image1_tensor, image2_tensor)
    rmse_value = torch.sqrt(mse_value)
    psnr_value = 10 * torch.log10(1. / mse_loss(image1_tensor, image2_tensor))
    ssim_value = ssim_loss(image1_tensor, image2_tensor)
    msssim_value = msssim_loss(image1_tensor, image2_tensor)
    lpips_value = lpips_loss(image1_tensor, image2_tensor)
    return mse_value, rmse_value, psnr_value, ssim_value, msssim_value, lpips_value


def test_psnr(model: nn.Module, dataloader: torch.utils.data.DataLoader, gpu: int = None) -> [torch.Tensor,
                                                                                              torch.Tensor]:
    mse_loss = nn.MSELoss().cuda(gpu)
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu)

    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_psnr_value = 0.
    total_ssim_value = 0.
    total = len(dataloader)
    for i, (lr, _, hr) in progress_bar:
        # Move data to special device.
        if gpu is not None:
            lr = lr.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            hr = hr.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            sr = model(lr)

        # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
        psnr_value = 10 * torch.log10(1. / mse_loss(sr, hr))
        ssim_value = ssim_loss(sr, hr)

        total_psnr_value += psnr_value
        total_ssim_value += ssim_value

        progress_bar.set_description(f"PSNR: {psnr_value:.2f}dB SSIM: {ssim_value:.2f}.")

    out = total_psnr_value / total, total_ssim_value / total

    return out


def test_gan(model: nn.Module, dataloader: torch.utils.data.DataLoader, gpu: int = None) -> [torch.Tensor,
                                                                                             torch.Tensor,
                                                                                             torch.Tensor]:
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS().cuda(gpu)
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu)
    # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    gmsd_loss = GMSD().cuda(gpu)

    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_ssim_value = 0.
    total_lpips_value = 0.
    total_gmsd_value = 0.
    total = len(dataloader)
    for i, (lr, _, hr) in progress_bar:
        # Move data to special device.
        if gpu is not None:
            lr = lr.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            hr = hr.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            sr = model(lr)

        # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
        ssim_value = ssim_loss(sr, hr)
        # The LPIPS of the generated fake high-resolution image and real high-resolution image is calculated.
        lpips_value = lpips_loss(sr, hr)
        # The GMSD of the generated fake high-resolution image and real high-resolution image is calculated.
        gmsd_value = gmsd_loss(sr, hr)

        total_ssim_value += ssim_value
        total_lpips_value += lpips_value
        total_gmsd_value += gmsd_value

        progress_bar.set_description(f"SSIM: {ssim_value:.2f} LPIPS: {lpips_value:.2f} GMSD: {gmsd_value:.2f}.")

    out = total_ssim_value / total, total_lpips_value / total, total_gmsd_value / total

    return out

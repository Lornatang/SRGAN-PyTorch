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
from .image_quality_assessment import SSIM

__all__ = [
    "iqa", "test"
]


def iqa(source: torch.Tensor, target: torch.Tensor, gpu: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Image quality evaluation function.

    Args:
        source (torch.Tensor): Original tensor picture.
        target (torch.Tensor): Target tensor picture.
        gpu (int): Graphics card index.

    Returns:
        MSE, RMSE, PSNR, SSIM, LPIPS, GMSD.
    """
    mse_loss = nn.MSELoss().cuda(gpu).eval()
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS(gpu).cuda(gpu).eval()
    # Reference sources from https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu).eval()
    # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    gmsd_loss = GMSD().cuda(gpu).eval()

    # Complete estimate.
    with torch.no_grad():
        mse_value = mse_loss(source, target)
        rmse_value = torch.sqrt(mse_value)
        psnr_value = 10 * torch.log10(1. / mse_value)
        ssim_value = ssim_loss(source, target)
        lpips_value = lpips_loss(source, target)
        gmsd_value = gmsd_loss(source, target)

    return mse_value, rmse_value, psnr_value, ssim_value, lpips_value, gmsd_value


def test(dataloader: torch.utils.data.DataLoader, model: nn.Module, gpu: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_loss = nn.MSELoss().cuda(gpu).eval()
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu).eval()
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS(gpu).cuda(gpu).eval()
    # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    gmsd_loss = GMSD().cuda(gpu).eval()

    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_psnr_value = 0.
    total_ssim_value = 0.
    total_lpips_value = 0.
    total_gmsd_value = 0.
    total = len(dataloader)

    with torch.no_grad():
        for i, (lr, _, hr) in progress_bar:
            # Move data to special device.
            if gpu is not None:
                lr = lr.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                hr = hr.cuda(gpu, non_blocking=True)

            sr = model(lr)

            # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
            total_psnr_value += 10 * torch.log10(1. / mse_loss(sr, hr))
            # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
            total_ssim_value += ssim_loss(sr, hr)
            # The LPIPS of the generated fake high-resolution image and real high-resolution image is calculated.
            total_lpips_value += lpips_loss(sr, hr)
            # The GMSD of the generated fake high-resolution image and real high-resolution image is calculated.
            total_gmsd_value += gmsd_loss(sr, hr)

            progress_bar.set_description(f"PSNR: {total_psnr_value / (i + 1):.2f} "
                                         f"SSIM: {total_ssim_value / (i + 1):.4f} "
                                         f"LPIPS: {total_lpips_value / (i + 1):.4f} "
                                         f"GMSD: {total_gmsd_value / (i + 1):.4f}")

    out = total_psnr_value / total, total_ssim_value / total, total_lpips_value / total, total_gmsd_value / total

    return out

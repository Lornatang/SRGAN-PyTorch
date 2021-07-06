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
from .image_quality_assessment import PSNR
from .image_quality_assessment import SSIM

__all__ = ["iqa", "test"]


def iqa(source, target, gpu):
    """Image quality evaluation function.

    Args:
        source (torch.Tensor): Original tensor image.
        target (torch.Tensor): Target tensor image.
        gpu (int): Graphics card index.

    Returns:
        MSE, RMSE, PSNR(RGB channel), SSIM, MS-SSIM, LPIPS, GMSD.
    """
    mse_loss = nn.MSELoss().cuda(gpu).eval()
    psnr_loss = PSNR(y_channel=False, gpu=gpu).cuda(gpu).eval()
    # Source code reference from `https://github.com/richzhang/PerceptualSimilarity`
    lpips_loss = LPIPS(gpu).cuda(gpu).eval()
    # Source code reference from `https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py`
    ssim_loss = SSIM().cuda(gpu).eval()
    # Source code reference from `https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MS_SSIM.py`
    msssim_loss = MS_SSIM().cuda(gpu).eval()
    # Source code reference from `http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm`
    gmsd_loss = GMSD().cuda(gpu).eval()

    # Convert the image value range to [0, 1].
    source = (source + 1) / 2
    target = (target + 1) / 2

    # Complete estimate.
    with torch.no_grad():
        mse_value = mse_loss(source, target)
        rmse_value = torch.sqrt(mse_value)
        psnr_value = psnr_loss(source, target)
        ssim_value = ssim_loss(source, target)
        msssim_value = msssim_loss(source, target)
        lpips_value = lpips_loss(source, target)
        gmsd_value = gmsd_loss(source, target)

    return mse_value, rmse_value, psnr_value, ssim_value, msssim_value, lpips_value, gmsd_value


def test(dataloader, model, gpu):
    """Image quality evaluation function (dataloader).

    Args:
        dataloader (torch.utils.data.DataLoader): Test dataset loader.
        model (nn.Module): Super-resolution model.
        gpu (int): Graphics card index.

    Returns:
        PSNR(RGB channel), SSIM.
    """
    psnr_loss = PSNR(y_channel=False, gpu=gpu).cuda(gpu).eval()
    # Source code reference from `https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py`
    ssim_loss = SSIM().cuda(gpu).eval()

    # switch eval mode.
    model.eval()

    total = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), total=total)
    total_psnr_value = 0.
    total_ssim_value = 0.

    with torch.no_grad():
        for i, (lr, hr) in progress_bar:
            # Move data to special device.
            if gpu is not None:
                lr = lr.cuda(gpu, non_blocking=True)
                hr = hr.cuda(gpu, non_blocking=True)

            sr = model(lr)

            # Convert the image value range to [0, 1].
            sr = (sr + 1) / 2
            hr = (hr + 1) / 2

            # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
            total_psnr_value += psnr_loss(sr, hr)
            # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
            total_ssim_value += ssim_loss(sr, hr)

            progress_bar.set_description(f"PSNR: {total_psnr_value / (i + 1):.2f} "
                                         f"SSIM: {total_ssim_value / (i + 1):.4f}")

    out = total_psnr_value / total, total_ssim_value / total

    return out

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
import math

import cv2
import lpips
import torch.nn as nn
import torch.utils.data
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp
from tqdm import tqdm

from .calculate_niqe import niqe
from .transform import opencv2tensor

__all__ = [
    "image_quality_evaluation", "test_psnr", "test_gan"
]


def image_quality_evaluation(sr_filename: str, hr_filename: str, device: torch.device = "cpu"):
    """Image quality evaluation function.

    Args:
        sr_filename (str): Image file name after super resolution.
        hr_filename (str): Original high resolution image file name.
        device (optional, torch.device): Selection of data processing equipment in PyTorch. (Default: ``cpu``).

    Returns:
        If the `simple` variable is set to ``False`` return `mse, rmse, psnr, ssim, msssim, niqe, sam, vifp, lpips`,
        else return `psnr, ssim`.
    """
    # Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
    lpips_loss = lpips.LPIPS(net="vgg", verbose=False).to(device)
    # Evaluate performance
    sr = cv2.imread(sr_filename)
    hr = cv2.imread(hr_filename)

    # For LPIPS evaluation
    sr_tensor = opencv2tensor(sr, device)
    hr_tensor = opencv2tensor(hr, device)

    # Complete estimate.
    mse_value = mse(sr, hr)
    rmse_value = rmse(sr, hr)
    psnr_value = psnr(sr, hr)
    ssim_value = ssim(sr, hr)
    msssim_value = msssim(sr, hr)
    niqe_value = niqe(sr_filename)
    sam_value = sam(sr, hr)
    vifp_value = vifp(sr, hr)
    lpips_value = lpips_loss(sr_tensor, hr_tensor)
    return mse_value, rmse_value, psnr_value, ssim_value, msssim_value, niqe_value, sam_value, vifp_value, lpips_value


def test_psnr(model: nn.Module, psnr_criterion: nn.MSELoss, dataloader: torch.utils.data.DataLoader,
              device: torch.device = "cpu"):
    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_psnr = 0.
    for i, data in progress_bar:
        # Move data to special device.
        lr = data[0].to(device)
        hr = data[2].to(device)

        with torch.no_grad():
            sr = model(lr)

        # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
        psnr = 10 * math.log10(1. / psnr_criterion(sr, hr).item())
        total_psnr += psnr

        progress_bar.set_description(f"PSNR: {psnr:.2f}dB.")

    return total_psnr / len(dataloader)


def test_gan(model: nn.Module, psnr_criterion: nn.MSELoss, lpips_criterion: lpips.LPIPS,
             dataloader: torch.utils.data.DataLoader, device: torch.device = "cpu"):
    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_psnr = 0.
    total_lpips = 0.
    for i, data in progress_bar:
        # Move data to special device.
        lr = data[0].to(device)
        hr = data[2].to(device)

        with torch.no_grad():
            sr = model(lr)

        # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
        psnr = 10 * math.log10(1. / psnr_criterion(sr, hr).item())
        # The LPIPS of the generated fake high-resolution image and real high-resolution image is calculated.
        lpips = torch.mean(lpips_criterion(sr, hr)).item()

        total_psnr += psnr
        total_lpips += lpips

        progress_bar.set_description(f"PSNR: {psnr:.2f}dB LPIPS: {lpips:.4f}.")

    return total_psnr / len(dataloader), total_lpips / len(dataloader)

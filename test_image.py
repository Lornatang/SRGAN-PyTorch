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
import argparse
import os
import time

import cv2
import lpips
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp

from srgan_pytorch import Generator
from srgan_pytorch import cal_niqe
from srgan_pytorch import select_device

parser = argparse.ArgumentParser(description="Photo-Realistic Single Image Super-Resolution Using "
                                             "a Generative Adversarial Network.")
parser.add_argument("--lr", type=str,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str,
                    help="Raw high resolution image name.")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weight/SRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weight/SRGAN_4x.pth``).")
parser.add_argument("--device", default="cpu",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")

args = parser.parse_args()

try:
    os.makedirs("benchmark")
except OSError:
    pass

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=1)

# Construct SRGAN model.
model = Generator(upscale_factor=args.upscale_factor).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))

# Set model eval mode
model.eval()

# Just convert the data to Tensor format
pre_process = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load image
lr = Image.open(args.lr)
hr = Image.open(args.hr)
lr = pre_process(lr).unsqueeze(0)
hr = pre_process(hr).unsqueeze(0)
lr = lr.to(device)
hr = hr.to(device)

start_time = time.time()
with torch.no_grad():
    sr = model(lr)
end_time = time.time()

vutils.save_image(lr, "lr.png", normalize=True)
vutils.save_image(sr, "sr.png", normalize=True)
vutils.save_image(hr, "hr.png", normalize=True)

# Evaluate performance
src_img = cv2.imread("sr.png")
dst_img = cv2.imread("hr.png")

# Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
lpips_loss = lpips.LPIPS(net="vgg").to(device)

mse_value = mse(src_img, dst_img)
rmse_value = rmse(src_img, dst_img)
psnr_value = psnr(src_img, dst_img)
ssim_value = ssim(src_img, dst_img)
ms_ssim_value = msssim(src_img, dst_img)  # 30.00+000j
niqe_value = cal_niqe("sr.png")
sam_value = sam(src_img, dst_img)
vif_value = vifp(src_img, dst_img)
lpips_value = lpips_loss(sr, hr)

print("\n")
print("====================== Performance summary ======================")
print(f"MSE: {mse_value:.2f}\n"
      f"RMSE: {rmse_value:.2f}\n"
      f"PSNR: {psnr_value:.2f}\n"
      f"SSIM: {ssim_value[0]:.4f}\n"
      f"MS-SSIM: {ms_ssim_value.real:.4f}\n"
      f"NIQE: {niqe_value:.2f}\n"
      f"SAM: {sam_value:.4f}\n"
      f"VIF: {vif_value:.4f}\n"
      f"LPIPS: {lpips_value.item():.4f}"
      f"Use time: {(end_time - start_time) * 1000:.2f}ms/{(end_time - start_time)}s.")
print("============================== End ==============================")
print("\n")

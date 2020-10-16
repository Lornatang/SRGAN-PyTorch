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

import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp
from tqdm import tqdm

from srgan_pytorch import DatasetFromFolder
from srgan_pytorch import Generator
from srgan_pytorch import cal_niqe
from srgan_pytorch import select_device

parser = argparse.ArgumentParser(description="Photo-Realistic Single Image Super-Resolution Using "
                                             "a Generative Adversarial Network.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weights/SRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weights/SRGAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")

args = parser.parse_args()

try:
    os.makedirs("benchmark")
except OSError:
    pass

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct SRGAN model.
model = Generator(upscale_factor=args.upscale_factor).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

# Evaluate algorithm performance
total_mse_value = 0.0
total_rmse_value = 0.0
total_psnr_value = 0.0
total_ssim_value = 0.0
total_ms_ssim_value = 0.0
total_niqe_value = 0.0
total_sam_value = 0.0
total_vif_value = 0.0

# Start evaluate model performance
with torch.no_grad():
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for iteration, (input, target) in progress_bar:
        # Set model gradients to zero
        lr = input.to(device)
        hr = target.to(device)

        sr = model(lr)

        vutils.save_image(lr, f"./benchmark/lr_{iteration}.bmp", normalize=True)
        vutils.save_image(sr, f"./benchmark/sr_{iteration}.bmp", normalize=True)
        vutils.save_image(hr, f"./benchmark/hr_{iteration}.bmp", normalize=True)

        mse_value = mse(sr, hr)
        rmse_value = rmse(sr, hr)
        psnr_value = psnr(sr, hr)
        ssim_value = ssim(sr, hr)
        ms_ssim_value = msssim(sr, hr)
        niqe_value = cal_niqe(f"./benchmark/sr_{iteration}.bmp")
        sam_value = sam(sr, hr)
        vif_value = vifp(sr, hr)

        total_mse_value += mse_value
        total_rmse_value += rmse_value
        total_psnr_value += psnr_value
        total_ssim_value += ssim_value
        total_ms_ssim_value += ms_ssim_value
        total_niqe_value += niqe_value
        total_sam_value += sam_value
        total_vif_value += vif_value

        progress_bar.set_description(f"[{iteration + 1}/{len(dataloader)}] "
                                     f"PSNR: {psnr_value:.2f}dB SSIM: {ssim_value:.4f}")

    avg_mse_value = total_mse_value / len(dataloader)
    avg_rmse_value = total_rmse_value / len(dataloader)
    avg_psnr_value = total_psnr_value / len(dataloader)
    avg_ssim_value = total_ssim_value / len(dataloader)
    avg_ms_ssim_value = total_ms_ssim_value / len(dataloader)
    avg_niqe_value = total_niqe_value / len(dataloader)
    avg_sam_value = total_sam_value / len(dataloader)
    avg_vif_value = total_vif_value / len(dataloader)

print("\n")
print("====================== Performance summary ======================")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg NIQE: {avg_niqe_value:.2f}\n"
      f"Avg SAM: {avg_sam_value:.4f}\n"
      f"Avg VIF: {avg_vif_value:.4f}")
print("============================== End ==============================")
print("\n")

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
import random

import cv2
import torch.autograd.profiler as profiler
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from sewar.full_ref import msssim
from sewar.full_ref import sam
from sewar.full_ref import vifp

from srgan_pytorch import Generator
from srgan_pytorch import ValDatasetFromFolder
from srgan_pytorch import cal_mse
from srgan_pytorch import cal_niqe
from srgan_pytorch import cal_psnr
from srgan_pytorch import cal_rmse
from srgan_pytorch import cal_ssim
from srgan_pytorch import calculate_valid_crop_size

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data/DIV2K",
                    help="Path to datasets. (default:`./data/DIV2K`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 4, 8, 16],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")
parser.add_argument("--weights", default="./weights/srgan_X4.pth", required=True,
                    help="Path to weights. (default:`./weights/srgan_X4.pth`).")
parser.add_argument("--outf", default="./result",
                    help="folder to output images. (default:`./result`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = ValDatasetFromFolder(dataset_dir=f'{args.dataroot}/val',
                               upscale_factor=args.scale_factor)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

model = Generator(scale_factor=args.scale_factor).to(device)

model.load_state_dict(torch.load(args.weights, map_location=device))

# Memory call of analysis model
lr_image_size = calculate_valid_crop_size(args.image_size, args.scale_factor)
lr = torch.randn(args.batch_size, 3, lr_image_size, lr_image_size, device=device)

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(lr)
print(" # Generator # ")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

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
    model.eval()
    for i, data in enumerate(dataloader):
        lr_real_image = data[0].to(device)
        hr_restore_image = data[1].to(device)
        hr_real_image = data[2].to(device)

        hr_fake_image = model(lr_real_image)

        vutils.save_image(hr_real_image, f"{args.outf}/hr_real_epoch_{i}.png", normalize=True)
        vutils.save_image(hr_restore_image, f"{args.outf}/hr_restore_epoch_{i}.png", normalize=True)
        vutils.save_image(hr_fake_image, f"{args.outf}/hr_fake_epoch_{i}.png", normalize=True)

        src_img = cv2.imread(f"{args.outf}/hr_fake_epoch_{i}.png")
        dst_img = cv2.imread(f"{args.outf}/hr_real_epoch_{i}.png")

        mse_value = cal_mse(src_img, dst_img)
        rmse_value = cal_rmse(src_img, dst_img)
        psnr_value = cal_psnr(src_img, dst_img)
        ssim_value = cal_ssim(src_img, dst_img)
        ms_ssim_value = msssim(src_img, dst_img)
        niqe_value = cal_niqe(f"{args.outf}/hr_fake_epoch_{i}.png")
        sam_value = sam(src_img, dst_img)
        vif_value = vifp(src_img, dst_img)

        total_mse_value += mse_value
        total_rmse_value += rmse_value
        total_psnr_value += psnr_value
        total_ssim_value += ssim_value
        total_ms_ssim_value += ms_ssim_value
        total_niqe_value += niqe_value
        total_sam_value += sam_value
        total_vif_value += vif_value

print("\n")
print("====================== Performance summary ======================")
print(f"Avg MSE: {total_mse_value / len(dataloader):.2f}\n"
      f"Avg RMSE: {total_rmse_value / len(dataloader):.2f}\n"
      f"Avg PSNR: {total_psnr_value / len(dataloader):.2f}\n"
      f"Avg SSIM: {total_ssim_value / len(dataloader):.4f}\n"
      f"Avg MS-SSIM: {total_ms_ssim_value / len(dataloader):.4f}\n"
      f"Avg NIQE: {total_niqe_value / len(dataloader):.2f}\n"
      f"Avg SAM: {total_sam_value / len(dataloader):.4f}\n"
      f"Avg VIF: {total_vif_value / len(dataloader):.4f}")
print("============================== End ==============================")
print("\n")

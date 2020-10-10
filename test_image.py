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

import cv2
import torch.backends.cudnn as cudnn
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

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. "
                         "(default:`./assets/baby.png`)")
parser.add_argument("--weights", type=str, default="weights/SRGAN_4x.pth",
                    help="Generator model name.  "
                         "(default:`weights/SRGAN_4x.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--image-size", type=int, default=96,
                    help="size of the data crop (squared assumed). (default:96)")
parser.add_argument("--scale-factor", default=4, type=int,
                    help="Super resolution upscale factor")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator(upscale_factor=args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model mode
model.eval()

# Load image
image = Image.open(args.file)

lr_process = transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()])
hr_process = transforms.Compose([transforms.Resize(args.image_size * args.scale_factor), transforms.ToTensor()])

hr_real_image = hr_process(image).unsqueeze(0)
lr_real_image = lr_process(image).unsqueeze(0)
lr_real_image = lr_real_image.to(device)

hr_fake_image = model(lr_real_image)
vutils.save_image(hr_real_image, "target.png", normalize=False)
vutils.save_image(hr_fake_image, "result.png", normalize=False)

# Evaluate performance
src_img = cv2.imread("result.png")
dst_img = cv2.imread("target.png")

mse_value = mse(src_img, dst_img)
rmse_value = rmse(src_img, dst_img)
psnr_value = psnr(src_img, dst_img)
ssim_value = ssim(src_img, dst_img)
ms_ssim_value = msssim(src_img, dst_img)
niqe_value = cal_niqe("result.png")
sam_value = sam(src_img, dst_img)
vif_value = vifp(src_img, dst_img)

print("\n")
print("====================== Performance summary ======================")
print(f"MSE: {mse_value:.2f}\n"
      f"RMSE: {rmse_value:.2f}\n"
      f"PSNR: {psnr_value:.2f}\n"
      f"SSIM: {ssim_value:.4f}\n"
      f"MS-SSIM: {ms_ssim_value:.4f}\n"
      f"NIQE: {niqe_value:.2f}\n"
      f"SAM: {sam_value:.4f}\n"
      f"VIF: {vif_value:.4f}")
print("============================== End ==============================")

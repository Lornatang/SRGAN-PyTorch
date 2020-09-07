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

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from srgan_pytorch import Generator
from srgan_pytorch import cal_psnr
from srgan_pytorch import cal_ssim

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to dataset. (default:`./data`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("--scale-factor", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--weights", default="./weights/netG_X4.pth",
                    help="Path to netG (default:`./weights/netG_X4.pth`).")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default:`./outputs`).")
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
    print("WARNING: You have a CUDA device, "
          "so you should probably run with --cuda")

# Load dataset
dataset = datasets.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(
                                       args.image_size * args.scale_factor),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, pin_memory=True,
                                         num_workers=int(args.workers))

# Setting device
device = torch.device("cuda:0" if args.cuda else "cpu")

# Load model
model = Generator(8, args.scale_factor).to(device)
model.load_state_dict(torch.load(args.weights))

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]),
                             ])

total_psnr_value = 0.0
total_ssim_value = 0.0

for i, data in enumerate(dataloader):
    hr_real_image = data[0].to(device)
    batch_size = hr_real_image.size(0)

    lr_real_image = torch.randn(batch_size, 3, args.image_size,
                                args.image_size, device=device)

    # Down sample images to low resolution
    for batch_index in range(batch_size):
        lr_real_image[batch_index] = resize(hr_real_image[batch_index].cpu())

    # Generate real and fake inputs
    hr_fake_image = model(lr_real_image)

    vutils.save_image(hr_real_image,
                      f"{args.outf}/hr_real_{i}.png",
                      normalize=True)
    vutils.save_image(hr_fake_image,
                      f"{args.outf}/hr_fake_{i}.png", normalize=True)
    vutils.save_image(lr_real_image,
                      f"{args.outf}/lr_real_{i}.png",
                      normalize=True)

    total_psnr_value += cal_psnr(f"{args.outf}/hr_real_{i}.png",
                                 f"{args.outf}/hr_fake_{i}.png")
    total_ssim_value += cal_ssim(f"{args.outf}/hr_real_{i}.png",
                                 f"{args.outf}/hr_fake_{i}.png")

print(f"Avg PSNR: {total_psnr_value / len(dataloader):.2f} "
      f"Avg SSIM: {total_ssim_value / len(dataloader):.4f}")

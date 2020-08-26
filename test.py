# Copyright 2020 Lorna Authors. All Rights Reserved.
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
from tqdm import tqdm

from srgan_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to dataset. (default:`./data`)")
parser.add_argument("--image-size", type=int, default=88,
                    help="Size of the data crop (squared assumed). (default:88)")
parser.add_argument("--upscale-factor", type=int, default=4,
                    help="Super resolution upscale factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default:`./results`).")
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

dataset = datasets.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(args.image_size * args.upscale_factor),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator(scale_factor=args.upscale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load("weights/netG.pth", map_location=device))

# Set model mode
model.eval()

resize = transforms.Compose([transforms.Resize(args.image_size),
                             transforms.ToTensor()])

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, data in progress_bar:
    high_resolution_real_image = data[0].to(device)
    batch_size = high_resolution_real_image.size(0)

    # Down sample images to low resolution
    low_resolution_image = resize(high_resolution_real_image)

    # Generate real and fake inputs
    high_resolution_fake_image = model(low_resolution_image)

    vutils.save_image(high_resolution_real_image, f"{args.outf}/high_res_real_{i}.png", normalize=False)
    vutils.save_image(high_resolution_fake_image.detach(), f"{args.outf}/high_res_fake_{i}.png", normalize=True)
    vutils.save_image(low_resolution_image, f"{args.outf}/low_res_real_{i}.png", normalize=True)

    progress_bar.set_description(f"[Number of currently processed pictures: {i} images]")

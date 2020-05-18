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

from srgan_pytorch import Discriminator
from srgan_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to dataset. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--image-size", type=int, default=96,
                    help="Size of the data crop (squared assumed). (default:96)")
parser.add_argument("--up-sampling", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG", default="./weights/netG.pth", help="Path to netG (default:`./weights/netG.pth`).")
parser.add_argument("--netD", default="./weights/netD.pth", help="Path to netD (default:`./weights/netD.pth`).")
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
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = datasets.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomResizedCrop(args.image_size * args.up_sampling),
                                   transforms.ToTensor()]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, pin_memory=True, num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

generator = Generator(n_residual_blocks=8, upsample_factor=args.up_sampling).to(device)
discriminator = Discriminator().to(device)

generator = torch.nn.DataParallel(generator).to(device)
discriminator = torch.nn.DataParallel(discriminator).to(device)

if args.netG != "":
    generator.load_state_dict(torch.load(args.netG))
if args.netD != "":
    discriminator.load_state_dict(torch.load(args.netD))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             normalize,
                             ])

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, data in progress_bar:
    high_resolution_real_image = data[0].to(device)
    batch_size = high_resolution_real_image.size(0)

    low_resolution_image = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)

    # Down sample images to low resolution
    for batch_index in range(batch_size):
        low_resolution_image[batch_index] = resize(high_resolution_real_image[batch_index].cpu())

    # Generate real and fake inputs
    high_resolution_fake_image = generator(low_resolution_image)

    vutils.save_image(high_resolution_real_image, f"{args.outf}/high_res_real_{i}.png", normalize=True)
    vutils.save_image(high_resolution_fake_image.detach(), f"{args.outf}/high_res_fake_{i}.png", normalize=True)
    vutils.save_image(low_resolution_image, f"{args.outf}/low_res_real_{i}.png", normalize=True)

    progress_bar.set_description(f"[Number of currently processed pictures: {i} images]")

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
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from srgan_pytorch import Discriminator
from srgan_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Super Resolution GAN.")
parser.add_argument("--file", type=str, default="./data/test.png",
                    help="Path to image. (default:`./data/test.png`)")
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

device = torch.device("cuda:0" if args.cuda else "cpu")

resize = transforms.Compose([transforms.Resize(args.image_size),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                             ])

image = Image.open(args.file)
image = resize(image)

generator = Generator(8, args.up_sampling).to(device)
discriminator = Discriminator().to(device)

high_resolution_fake_image = generator(image)
vutils.save_image(high_resolution_fake_image.detach(),
                  f"{args.outf}/fake.png",
                  normalize=True)
